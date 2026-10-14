// Browser test for the "Tentang demo ini" box of the static demo (site/): numbers come from model_info.json, layout on a
// 360x640 phone, keyboard access, contrast, no external requests, graceful fallback without model_info.json.
// Usage: python3 -m http.server 8765 -d site &   chrome --headless=new --remote-debugging-port=9235 about:blank &
//        node tests/site_about_smoke.mjs [baseUrl] [--allow-missing-report]   (the flag is for a partial bundle without data/report.json)
import assert from 'node:assert/strict';
import {mkdir, writeFile} from 'node:fs/promises';
const base = process.argv.slice(2).find(a => !a.startsWith('--')) || 'http://127.0.0.1:8765/';
const allowMissingReport = process.argv.includes('--allow-missing-report');
const pages = await (await fetch('http://127.0.0.1:9235/json/list')).json();
const page = pages.find(p => p.type === 'page');
assert(page, 'No browser page');
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
let counter = 0;
const pending = new Map();
let errors = [], failedRequests = [];
ws.onmessage = event => {
  const message = JSON.parse(event.data);
  if (message.id) {
    const item = pending.get(message.id);
    if (item) { pending.delete(message.id); message.error ? item.reject(Error(JSON.stringify(message.error))) : item.resolve(message.result); }
  } else if (message.method === 'Runtime.exceptionThrown') errors.push(JSON.stringify(message.params.exceptionDetails.exception?.description || message.params.exceptionDetails.text));
  else if (message.method === 'Runtime.consoleAPICalled' && message.params.type === 'error') errors.push('console.error: ' + message.params.args.map(a => a.value ?? a.description).join(' '));
  else if (message.method === 'Network.loadingFailed' && !message.params.canceled) failedRequests.push(message.params.errorText);
  else if (message.method === 'Network.responseReceived' && message.params.response.status >= 400) failedRequests.push(message.params.response.status + ' ' + message.params.response.url);
};
const command = (method, params = {}) => new Promise((resolve, reject) => {
  const id = ++counter;
  pending.set(id, {resolve, reject}); ws.send(JSON.stringify({id, method, params}));
  setTimeout(() => { if (pending.delete(id)) reject(Error('CDP timeout: ' + method)); }, 30000);
});
async function evaluate(expression) {
  const result = await command('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true});
  if (result.exceptionDetails) throw Error(JSON.stringify(result.exceptionDetails));
  return result.result.value;
}
async function until(expression, tries = 200) {
  for (let i = 0; i < tries; i++) { if (await evaluate(expression)) return; await new Promise(r => setTimeout(r, 200)); }
  throw Error('Timed out: ' + expression);
}
await mkdir('outputs/browser', {recursive: true});
const screenshot = async name => writeFile('outputs/browser/about-' + name + '.png', Buffer.from((await command('Page.captureScreenshot', {format: 'png', captureBeyondViewport: true})).data, 'base64'));
await Promise.all(['Runtime', 'Page', 'Network', 'Log'].map(d => command(d + '.enable')));
await command('Network.setCacheDisabled', {cacheDisabled: true});
await command('Emulation.setDeviceMetricsOverride', {width: 360, height: 640, deviceScaleFactor: 2, mobile: true});
await command('Page.navigate', {url: base});
await until("document.querySelector('#status').textContent.includes('dimuat')");

// Collapsed by default, one box, native <details>
assert.equal(await evaluate("document.querySelectorAll('details#about').length"), 1);
assert.equal(await evaluate("document.querySelector('#about').open"), false);
assert.equal(await evaluate("document.querySelector('#about .about-body').checkVisibility({contentVisibilityAuto:true,visibilityProperty:true})"), false, 'content hidden while collapsed');

// Keyboard: focus the summary and press Enter
await evaluate("document.querySelector('#about summary').focus()");
assert.equal(await evaluate("document.activeElement.tagName"), 'SUMMARY');
for (const type of ['keyDown', 'keyUp']) await command('Input.dispatchKeyEvent', {type, key: 'Enter', code: 'Enter', windowsVirtualKeyCode: 13, text: type === 'keyDown' ? '\r' : undefined});
await until("document.querySelector('#about').open");
assert.equal(await evaluate("getComputedStyle(document.querySelector('#about summary')).outlineStyle!=='none'"), true, 'visible focus indicator');
assert.ok(await evaluate("document.querySelector('#about summary').getBoundingClientRect().height>=44"), 'touch target >= 44px');

// Numbers are exactly those of model_info.json (nothing typed by hand in HTML/JS)
const info = await (await fetch(new URL('model_info.json', base))).json();
const entry = info.models.find(m => m.id === info.default);
const sci = v => Number(v).toExponential(1).replace('.', ',');
const text = await evaluate("document.querySelector('#about').textContent");
assert.ok(entry.js_parity, 'model_info.json should carry the measured JavaScript parity (node tests/js_parity.mjs --write-info)');
for (const expected of [sci(entry.js_parity.max_network_error_vs_keras), sci(entry.js_parity.max_end_to_end_probability_error), String(entry.js_parity.cases),
  Number(entry.parameters).toLocaleString('id-ID'), (entry.weights_bytes / 1e6).toFixed(2).replace('.', ','), info.export_environment.tensorflow, info.export_environment.keras, entry.architecture]) {
  assert.ok(text.includes(expected), 'missing in about box: ' + expected);
}
assert.match(text, /TensorFlow\/Keras/);
assert.match(text, /JavaScript/);
assert.match(text, /tidak meninggalkan perangkat/);
assert.match(text, /GitHub Pages/);

// Links: plain anchors to README section and Colab, no automatic requests
const links = await evaluate("[...document.querySelectorAll('#about a')].map(a=>({href:a.href,rel:a.rel,target:a.target}))");
assert.equal(links.length, 2);
assert.match(links[0].href, /github\.com\/.+\/tomato-vision#jalankan-versi-tensorflow$/);
assert.match(links[1].href, /^https:\/\/colab\.research\.google\.com\/github\/.+\/notebooks\/tomato_vision_colab\.ipynb$/);
for (const link of links) assert.match(link.rel, /noopener/);

// Phone layout 360x640: no horizontal scroll, box inside the viewport, readable text
await evaluate("window.scrollTo(0,0)");
assert.equal(await evaluate('document.documentElement.scrollWidth<=window.innerWidth+1'), true, 'horizontal overflow at 360px');
assert.equal(await evaluate("(()=>{const r=document.querySelector('#about').getBoundingClientRect();return r.left>=0&&r.right<=window.innerWidth})()"), true);
assert.equal(await evaluate("[...document.querySelectorAll('#about p, #about dd, #about dt')].every(n=>parseFloat(getComputedStyle(n).fontSize)>=13)"), true, 'font size >= 13px');
await evaluate("document.querySelector('#about').scrollIntoView()");
await screenshot('mobile-open');

// Contrast (WCAG AA 4.5:1) of the text colours used in the box against its white background
const contrast = await evaluate(`(()=>{
  const lum=c=>{const [r,g,b]=c.match(/\\d+(\\.\\d+)?/g).slice(0,3).map(Number).map(v=>{v/=255;return v<=.03928?v/12.92:((v+.055)/1.055)**2.4});return .2126*r+.7152*g+.0722*b};
  const ratio=(a,b)=>{const [x,y]=[lum(a),lum(b)].sort((p,q)=>q-p);return (x+.05)/(y+.05)};
  const bg=getComputedStyle(document.querySelector('#about')).backgroundColor;
  return [...document.querySelectorAll('#about summary, #about p, #about dt, #about dd, #about a')].map(n=>ratio(getComputedStyle(n).color,bg));
})()`);
assert.ok(Math.min(...contrast) >= 4.5, 'contrast too low: ' + Math.min(...contrast).toFixed(2));

// Desktop snapshot
await command('Emulation.setDeviceMetricsOverride', {width: 1280, height: 900, deviceScaleFactor: 1, mobile: false});
await screenshot('desktop-open');

// No external requests, nothing failed, no console errors
const external = await evaluate("performance.getEntriesByType('resource').map(e=>e.name).filter(n=>!n.startsWith(location.origin))");
assert.deepEqual(external, [], 'External requests found');
if (allowMissingReport) failedRequests = failedRequests.filter(r => !/ \/?.*data\/report\.json$/.test(r) && !/data\/site-config\.json$/.test(r));
assert.deepEqual(failedRequests, [], 'Failed requests');
assert.deepEqual(errors, [], 'Browser errors');

// Fallback: without model_info.json the box still renders and says why, with no script errors
await command('Network.setBlockedURLs', {urls: ['*model_info.json*']});
errors = []; failedRequests = [];
await command('Page.navigate', {url: base + '?fallback=1'});
await until("document.querySelector('#status').textContent.includes('dimuat')");
assert.match(await evaluate("document.querySelector('#about-parity').textContent"), /tidak tersedia/);
assert.equal(await evaluate("document.querySelectorAll('#about-facts dd').length"), 0);
assert.deepEqual(errors, [], 'Browser errors without model_info.json');
console.log(JSON.stringify({site_about_smoke: 'passed', model: info.default, min_contrast: +Math.min(...contrast).toFixed(2),
  checks: ['collapsed by default', 'keyboard toggle + focus ring', 'numbers equal model_info.json', 'links', '360x640 layout', 'contrast >= 4.5', 'no external requests', 'fallback without model_info.json']}, null, 2));
ws.close();
