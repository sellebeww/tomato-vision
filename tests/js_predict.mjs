// Helper for tests/test_predict_vs_web.py: runs the browser inference core (site/tomato-core.js) on raw RGB pixels.
// Usage: node tests/js_predict.mjs <request.json>   request: {site, model, images: [{name, width, height, rgb(base64)}]}
// Prints a JSON array with one prediction (or error) per image.
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const require = createRequire(import.meta.url);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const Core = require(path.join(root, 'site/tomato-core.js'));
const request = JSON.parse(readFileSync(process.argv[2]));
const manifest = JSON.parse(readFileSync(path.join(request.site, 'data/models.json')));
const entry = manifest.models.find(m => m.id === (request.model || manifest.default));
const dir = path.join(request.site, entry.path);
const spec = JSON.parse(readFileSync(path.join(dir, 'model.json')));
const raw = readFileSync(path.join(dir, spec.weights_file));
const model = Core.createModel(spec, raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
const out = request.images.map(image => {
  try {
    const result = Core.predictPixels(model, new Uint8Array(Buffer.from(image.rgb, 'base64')), image.width, image.height, 3);
    return {name: image.name, label: result.label, probabilities: result.probabilities, needs_review: result.needs_review,
      review_reasons: result.review_reasons, threshold: spec.confidence_threshold};
  } catch (error) { return {name: image.name, error: String(error.message || error)}; }
});
console.log(JSON.stringify(out));
