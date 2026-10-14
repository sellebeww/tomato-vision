// Browser test for the static GitHub Pages demo (site/). Expectations come from the published data files
// (data/models.json, data/report.json, data/site-config.json), never from hand-typed numbers.
// Usage: python3 -m http.server 8765 -d site &   chrome --headless=new --remote-debugging-port=9235 about:blank &
//        node tests/site_smoke.mjs [baseUrl] [--site folder-being-served]
import assert from 'node:assert/strict';
import {mkdir, writeFile} from 'node:fs/promises';
import {existsSync} from 'node:fs';
import path from 'node:path';
const argv = process.argv.slice(2), siteIndex = argv.indexOf('--site');
const siteDir = path.resolve(siteIndex >= 0 ? argv[siteIndex + 1] : 'site');   // local copy of the served folder (for uploads)
const base = argv.find((a, i) => !a.startsWith('--') && i !== siteIndex + 1 || (siteIndex < 0 && !a.startsWith('--'))) || 'http://127.0.0.1:8765/';
const getJSON = async name => { const r = await fetch(new URL(name, base)); assert.ok(r.ok, name + ' must be served'); return r.json(); };
const models = await getJSON('data/models.json');
const report = await getJSON('data/report.json');
const siteConfig = await getJSON('data/site-config.json');
const examples = siteConfig.examples ? report.examples : [];
const defaultModel = models.models.find(m => m.id === models.default);
const pct = v => (Number(v) * 100).toFixed(1).replace('.', ',') + '%';

const pages = await (await fetch('http://127.0.0.1:9235/json/list')).json();
const page = pages.find(p => p.type === 'page');
assert(page, 'No browser page');
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
let counter = 0;
const pending = new Map(), errors = [], failedRequests = [];
ws.onmessage = event => {
  const message = JSON.parse(event.data);
  if (message.id) {
    const item = pending.get(message.id);
    if (item) { clearTimeout(item.timer); pending.delete(message.id); message.error ? item.reject(Error(JSON.stringify(message.error))) : item.resolve(message.result); }
  } else if (message.method === 'Runtime.exceptionThrown') errors.push(JSON.stringify(message.params.exceptionDetails.exception?.description || message.params.exceptionDetails.text));
  else if (message.method === 'Runtime.consoleAPICalled' && message.params.type === 'error') errors.push('console.error: ' + message.params.args.map(a => a.value ?? a.description).join(' '));
  else if (message.method === 'Network.loadingFailed' && !message.params.canceled) failedRequests.push(message.params.errorText);
  else if (message.method === 'Network.responseReceived' && message.params.response.status >= 400) failedRequests.push(message.params.response.status + ' ' + message.params.response.url);
  else if (message.method === 'Log.entryAdded' && message.params.entry.level === 'error') errors.push('log: ' + message.params.entry.text + ' ' + (message.params.entry.url || ''));
};
const command = (method, params = {}) => new Promise((resolve, reject) => {
  const id = ++counter;
  const timer = setTimeout(() => { pending.delete(id); reject(Error('CDP timeout: ' + method)); }, 60000);
  pending.set(id, {resolve, reject, timer}); ws.send(JSON.stringify({id, method, params}));
});
async function evaluate(expression) {
  const result = await command('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true});
  if (result.exceptionDetails) throw Error(JSON.stringify(result.exceptionDetails));
  return result.result.value;
}
async function until(expression, tries = 600) {
  for (let i = 0; i < tries; i++) { if (await evaluate(expression)) return; await new Promise(r => setTimeout(r, 200)); }
  throw Error('Timed out: ' + expression);
}
const count = selector => evaluate(`document.querySelectorAll(${JSON.stringify(selector)}).length`);
const text = selector => evaluate(`document.querySelector(${JSON.stringify(selector)})?.textContent||''`);
await mkdir('outputs/browser', {recursive: true});
async function screenshot(name) {
  const result = await command('Page.captureScreenshot', {format: 'png', captureBeyondViewport: true});
  await writeFile('outputs/browser/site-' + name + '.png', Buffer.from(result.data, 'base64'));
}
await Promise.all(['Runtime', 'Page', 'Network', 'Log', 'DOM'].map(d => command(d + '.enable')));
await command('Network.setCacheDisabled', {cacheDisabled: true});
await command('Emulation.setDeviceMetricsOverride', {width: 1280, height: 900, deviceScaleFactor: 1, mobile: false});
await command('Page.navigate', {url: base});

// Default model loads in the browser; model card and selector reflect models.json
await until("document.querySelector('#status').textContent.includes('dimuat')");
assert.match(await text('#status'), new RegExp(models.default));
assert.equal(await count('#model-select option'), models.models.length);
assert.match(await text('#model-card'), new RegExp(defaultModel.parameters.toLocaleString('id-ID').replace('.', '\\.')));
assert.equal(await count('.example-button'), examples.length);
await screenshot('desktop-prediction');

// Example photos: every one is analysed without error; agreement with the owner label is recorded, not required
const agreement = [];
for (let i = 0; i < examples.length; i++) {
  await until("!document.querySelector('#upload').disabled&&!document.querySelectorAll('.example-button')[0].disabled");
  await evaluate(`document.querySelectorAll('.example-button')[${i}].click()`);
  await until("document.querySelector('.prediction-card h3')||document.querySelector('.prediction-card .error')");
  await until("!document.querySelector('.prediction-card .pending')");
  const card = await text('.prediction-card');
  assert.doesNotMatch(card, /tidak dapat|gagal/i, 'example ' + i + ': ' + card);
  assert.match(await text('.prediction-card h3'), /Dugaan: (Segar|Tidak segar|Busuk)/);
  assert.equal(await count('.prediction-card progress'), 3);
  assert.match(card, /Konsistensi/);
  assert.match(card, /Label pemilik: .*(sesuai|berbeda)/);
  agreement.push(/sesuai dengan dugaan/.test(card));
}

// Upload through the real file input: two published example photos and one non-image file
const uploads = examples.slice(0, 2).map(e => path.join(siteDir, e.file)).filter(existsSync);
const doc = await command('DOM.getDocument');
const input = await command('DOM.querySelector', {nodeId: doc.root.nodeId, selector: '#upload'});
if (uploads.length) {
  await command('DOM.setFileInputFiles', {nodeId: input.nodeId, files: uploads});
  await until("!document.querySelector('#predict').disabled");
  assert.match(await text('#file-info'), new RegExp(uploads.length + ' foto dipilih'));
  await evaluate("document.querySelector('#predict').click()");
  await until("!document.querySelector('#download-predictions').disabled");
  assert.equal(await count('.prediction-card'), uploads.length);
  assert.equal(await count('.prediction-card .error'), 0);
  await screenshot('desktop-results');
}
await command('DOM.setFileInputFiles', {nodeId: input.nodeId, files: [path.resolve('README.md')]});
await evaluate("document.querySelector('#upload').dispatchEvent(new Event('change'))");
await until("!document.querySelector('#predict').disabled");
await evaluate("document.querySelector('#predict').click()");
await until("document.querySelector('.prediction-card .error')");
assert.match(await text('.prediction-card .error'), /Format tidak didukung/);

// Switching to another exported model
if (models.models.length > 1) {
  const other = models.models.find(m => m.id !== models.default);
  await evaluate(`(()=>{const s=document.querySelector('#model-select');s.value=${JSON.stringify(other.id)};s.dispatchEvent(new Event('change'))})()`);
  await until(`document.querySelector('#status').textContent.includes('dimuat')&&document.querySelector('#status').textContent.includes(${JSON.stringify(other.id)})`);
  await evaluate(`(()=>{const s=document.querySelector('#model-select');s.value=${JSON.stringify(models.default)};s.dispatchEvent(new Event('change'))})()`);
  await until(`document.querySelector('#status').textContent.includes(${JSON.stringify(models.default)})&&document.querySelector('#status').textContent.includes('dimuat')`);
}

// Dataset / training / evaluation tabs rendered from report.json
for (const id of ['dataset', 'training', 'evaluation']) {
  await evaluate(`document.querySelector('[data-panel="${id}"]').click()`);
  await until(`!document.querySelector('#${id}').hidden`);
  await screenshot('desktop-' + id);
}
assert.equal(await count('#data-stats .stat'), 4);
assert.equal(await count('#gallery .photo'), examples.length);
assert.ok(await count('#protocol li') >= 4);
assert.equal(await count('#curves img'), models.models.length);
await until("[...document.querySelectorAll('#curves img')].every(i=>i.complete)");
assert.equal(await evaluate("[...document.querySelectorAll('#curves img')].every(i=>i.naturalWidth>0)"), true, 'learning curves load');
assert.equal(await count('#eval-stats .stat'), 4);
assert.equal(await count('#class-metrics tbody tr'), 3);
assert.doesNotMatch(await text('#evaluation-notice'), /tidak dapat|tidak tersedia/);
if (report.study_type === 'single_split') {
  const selected = report.selection.selected, best = report.candidates.find(c => c.name === selected);
  assert.equal(await count('#split-table tbody tr'), 3);
  assert.equal(await count('#ablation tbody tr'), report.candidates.length);
  assert.equal(await count('#ablation tbody tr.highlight'), 1);
  assert.match(await text('#selection-note'), new RegExp('Terpilih: ' + selected));
  assert.ok((await text('#eval-stats')).includes(pct(best.validation.accuracy)), 'validation accuracy from report.json');
  assert.match(await text('#eval-stats'), /Belum dinilai/);
  assert.equal(await count('#model-compare tbody tr'), report.candidates.length);
  assert.equal(await count('#confusion table'), report.candidates.length);
  assert.equal(await count('#val-predictions tbody tr'), best.predictions.length);
} else {
  const selected = report.selection.selected, cvRows = report.cv.filter(r => r.complete), testModels = Object.keys(report.test.models);
  assert.equal(await count('#split-table tbody tr'), 2);
  assert.equal(await count('#ablation tbody tr'), cvRows.length);
  assert.equal(await count('#ablation tbody tr.highlight'), 1);
  assert.match(await text('#selection-note'), new RegExp('Terpilih: ' + selected));
  assert.ok((await text('#eval-stats')).includes(pct(report.test.models[selected].accuracy.mean)), 'test accuracy from report.json');
  assert.ok((await text('#eval-stats')).includes(pct(cvRows.find(r => r.name === selected).accuracy.mean)), 'CV accuracy from report.json');
  assert.equal(await count('#model-compare tbody tr'), testModels.length);
  assert.equal(await count('#confusion tbody tr'), 3);
  assert.equal(await count('#confusion-test tbody tr'), 3);
  assert.equal(await count('#robustness tbody tr'), 1 + Object.keys(report.test.models[selected].robustness).length);
}
if (report.legacy_own_v1) assert.ok(await count('#legacy tbody tr') >= 4);

// Responsive layout (phone widths), every tab
for (const [width, height] of [[390, 844], [360, 640]]) {
  await command('Emulation.setDeviceMetricsOverride', {width, height, deviceScaleFactor: 1, mobile: true});
  for (const id of ['prediction', 'dataset', 'training', 'evaluation']) {
    await evaluate(`document.querySelector('[data-panel="${id}"]').click();window.scrollTo(0,0)`);
    await until(`!document.querySelector('#${id}').hidden`);
    assert.equal(await evaluate('document.documentElement.scrollWidth<=window.innerWidth+1'), true, `Mobile overflow ${width}px: ${id}`);
    await screenshot(`mobile${width}-${id}`);
  }
}

// Nothing may leave the origin and nothing may fail
const external = await evaluate("performance.getEntriesByType('resource').map(e=>e.name).filter(n=>!n.startsWith(location.origin))");
assert.deepEqual(external, [], 'External requests found');
assert.deepEqual(failedRequests, [], 'Failed requests');
assert.deepEqual(errors, [], 'Browser errors');
console.log(JSON.stringify({site_smoke: 'passed', default_model: models.default, models: models.models.length,
  examples_matching_owner_label: `${agreement.filter(Boolean).length}/${agreement.length} (recorded, not asserted)`,
  checks: ['model load + selector', 'examples analysed', 'file upload', 'invalid file rejected', 'model switch', 'dataset/training/evaluation from report.json',
    'mobile 390 and 360 px', 'no external requests', 'no console errors'], screenshots: 'outputs/browser/site-*.png'}, null, 2));
ws.close();
