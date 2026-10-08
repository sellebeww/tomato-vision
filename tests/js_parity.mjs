// Parity test: browser inference core (site/tomato-core.js) vs the Python predictor, for every exported model.
// Fixtures: tests/fixtures/web_parity_<model>.json (src/export_web.py).
// Run: node tests/js_parity.mjs [--site site] [--fixtures tests/fixtures] [--write-info]
// --write-info stores the measured numbers in <site>/model_info.json (shown in the demo, quoted by README/report).
// Without it the test only checks that model_info.json describes the same models and weights as models.json.
import assert from 'node:assert/strict';
import {readFileSync, writeFileSync, existsSync, renameSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const require = createRequire(import.meta.url);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const arg = (name, fallback) => { const i = process.argv.indexOf(name); return i > 0 ? path.resolve(process.argv[i + 1]) : path.join(root, fallback); };
const siteDir = arg('--site', 'site'), fixtureDir = arg('--fixtures', 'tests/fixtures');
const Core = require(path.join(root, 'site/tomato-core.js'));
const b64 = s => new Uint8Array(Buffer.from(s, 'base64'));
const maxAbs = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);
const MIN_CASES = 10;
const NETWORK_TOLERANCE = 1e-4, PROBABILITY_TOLERANCE = 0.01;   // browser network vs Keras; end-to-end incl. image resampling
const writeInfo = process.argv.includes('--write-info');

const manifest = JSON.parse(readFileSync(path.join(siteDir, 'data/models.json')));
assert.ok(manifest.models.some(m => m.id === manifest.default), 'default model must exist');
const summary = [];
for (const entry of manifest.models) {
  const dir = path.join(siteDir, entry.path);
  const spec = JSON.parse(readFileSync(path.join(dir, 'model.json')));
  const raw = readFileSync(path.join(dir, spec.weights_file));
  assert.equal(createHash('sha256').update(raw).digest('hex'), spec.weights_sha256, entry.id + ': weights hash');
  assert.equal(raw.length, entry.weights_bytes, entry.id + ': size in models.json');
  assert.ok(spec.keras_parity_max_abs_error <= 1e-4, entry.id + ': export gate');
  const t0 = performance.now();
  const model = Core.createModel(spec, raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
  const loadMs = performance.now() - t0;
  const fixture = JSON.parse(readFileSync(path.join(fixtureDir, `web_parity_${entry.id}.json`)));
  assert.ok(fixture.cases.length >= MIN_CASES, `${entry.id}: needs >= ${MIN_CASES} cases`);
  let worstInput = 0, worstNet = 0, worstProb = 0, slowest = 0;
  for (const c of fixture.cases) {
    const rgb = b64(c.rgb), expectedInput = b64(c.expected_input);
    assert.equal(rgb.length, c.width * c.height * 3, c.name + ': fixture size');
    // 1) Pillow-compatible letterbox reproduces the tensor (<= 1 level on < 1% of values).
    const tensor = Core.letterbox(rgb, c.width, c.height, 3, spec.input_size, spec.pad_color);
    let off = 0, worst = 0;
    for (let i = 0; i < tensor.length; i++) { const d = Math.abs(tensor[i] - expectedInput[i]); if (d) off++; worst = Math.max(worst, d); }
    assert.ok(worst <= 1 && off / tensor.length < 0.01, `${entry.id}/${c.name}: preprocessing (max ${worst}, ${off} values)`);
    worstInput = Math.max(worstInput, worst);
    // 2) Network on Python's own tensor matches Keras.
    const fromPython = Core.predictTensor(model, expectedInput);
    const netError = Math.max(...fromPython._raw_views.map((v, i) => maxAbs(Array.from(v), c.expected_views[i])));
    assert.ok(netError < NETWORK_TOLERANCE, `${entry.id}/${c.name}: network deviates from Keras by ${netError}`);
    worstNet = Math.max(worstNet, netError);
    // 3) End to end, including review logic and quality diagnostics.
    const t1 = performance.now();
    const r = Core.predictPixels(model, rgb, c.width, c.height, 3);
    slowest = Math.max(slowest, performance.now() - t1);
    const e = c.expected;
    const probError = Math.max(...Object.keys(e.probabilities).map(k => Math.abs(r.probabilities[k] - e.probabilities[k])));
    assert.ok(probError < PROBABILITY_TOLERANCE, `${entry.id}/${c.name}: probability error ${probError}`);
    worstProb = Math.max(worstProb, probError);
    assert.equal(r.label, e.label, `${entry.id}/${c.name}: label`);
    assert.equal(r.needs_review, e.needs_review, `${entry.id}/${c.name}: needs_review`);
    assert.deepEqual(r.review_reasons, e.review_reasons, `${entry.id}/${c.name}: review reasons`);
    assert.equal(r.stability.agreement, e.agreement, `${entry.id}/${c.name}: agreement`);
    assert.ok(Math.abs(r.probability_margin - e.probability_margin) < 0.02, `${entry.id}/${c.name}: margin`);
    for (const key of ['brightness', 'contrast', 'edge_variance']) {
      const tol = Math.max(1e-6, Math.abs(e.quality[key]) * 0.02);
      assert.ok(Math.abs(r.quality[key] - e.quality[key]) <= tol, `${entry.id}/${c.name}: quality.${key}`);
    }
    assert.deepEqual(r.quality.warnings, e.quality.warnings, `${entry.id}/${c.name}: quality warnings`);
  }
  summary.push({model: entry.id, sha256: spec.weights_sha256, architecture: spec.architecture, cases: fixture.cases.length, weights_mb: +(raw.length / 1e6).toFixed(2),
    max_input_level_diff: worstInput, max_network_error_vs_keras: worstNet, max_end_to_end_probability_error: worstProb,
    model_init_ms: Math.round(loadMs), slowest_prediction_ms: Math.round(slowest)});
}

// Guard rails
const anySpec = JSON.parse(readFileSync(path.join(siteDir, manifest.models[0].path, 'model.json')));
const anyRaw = readFileSync(path.join(siteDir, manifest.models[0].path, anySpec.weights_file));
const anyBuffer = anyRaw.buffer.slice(anyRaw.byteOffset, anyRaw.byteOffset + anyRaw.byteLength);
assert.throws(() => Core.createModel({...anySpec, weights_floats: anySpec.weights_floats + 1}, anyBuffer), /Ukuran bobot/);
assert.throws(() => Core.createModel({...anySpec, nodes: [...anySpec.nodes, {op: 'lstm', name: 'x'}]}, anyBuffer), /tidak didukung: lstm/);
assert.throws(() => Core.createModel({...anySpec, format: 'other'}, anyBuffer), /Format/);
assert.throws(() => Core.letterbox(new Uint8Array(4), 6000, 5000, 3, 128, 127), /25 megapiksel/);
assert.equal(Core.pyRound(0.5), 0); assert.equal(Core.pyRound(1.5), 2); assert.equal(Core.pyRound(2.5), 2); assert.equal(Core.pyRound(2.6), 3);
assert.ok(Core.temperatureScale([0.2, 0.3, 0.5], 1).every((v, i) => Math.abs(v - [0.2, 0.3, 0.5][i]) < 1e-9), 'T=1 is identity');

// model_info.json: must describe exactly the models that were just verified
const infoPath = path.join(siteDir, 'model_info.json');
if (existsSync(infoPath)) {
  const info = JSON.parse(readFileSync(infoPath));
  assert.deepEqual(info.models.map(m => m.id).sort(), summary.map(m => m.model).sort(), 'model_info.json lists different models than models.json');
  for (const item of summary) {
    const recorded = info.models.find(m => m.id === item.model);
    assert.equal(recorded.weights_sha256, item.sha256, item.model + ': model_info.json is stale (weights changed); re-run with --write-info');
    if (writeInfo) {
      recorded.js_parity = {
        max_network_error_vs_keras: item.max_network_error_vs_keras,
        max_end_to_end_probability_error: item.max_end_to_end_probability_error,
        max_input_level_diff: item.max_input_level_diff,
        cases: item.cases,
        tolerance: {network: NETWORK_TOLERANCE, end_to_end_probability: PROBABILITY_TOLERANCE},
        measured_at: new Date().toISOString().replace(/\.\d+Z$/, 'Z'),
        runtime: 'node ' + process.version,
        method: 'tests/js_parity.mjs: browser inference code (site/tomato-core.js) vs outputs of the Keras model stored in tests/fixtures'};
    } else if (recorded.js_parity) {
      assert.ok(recorded.js_parity.max_network_error_vs_keras < NETWORK_TOLERANCE, item.model + ': recorded network error exceeds tolerance');
    }
  }
  if (writeInfo) { writeFileSync(infoPath + '.tmp', JSON.stringify(info, null, 2) + '\n'); renameSync(infoPath + '.tmp', infoPath); }
} else assert.ok(!writeInfo, '--write-info needs ' + infoPath + ' (run python -m src.export_web first)');
console.log(JSON.stringify({passed: true, model_info: existsSync(infoPath) ? (writeInfo ? 'written' : 'consistent') : 'absent', models: summary.map(({sha256, ...rest}) => rest)}, null, 2));
