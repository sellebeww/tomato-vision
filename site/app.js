'use strict';
// Tomato Vision: static demo for GitHub Pages. Inference runs in the browser (tomato-core.js / worker.js).
// Every number on the page comes from data/models.json, data/report.json and model_info.json (src/export_web.py).
const $ = id => document.getElementById(id);
const labels = {segar: 'Segar', tidak_segar: 'Tidak segar', busuk: 'Busuk'};
const classKeys = Object.keys(labels);
const MAX_BYTES = 10_000_000, MAX_PIXELS = 25_000_000, MAX_CANVAS_PIXELS = 16_000_000, MAX_FILES = 20;
const ALLOWED = new Set(['image/jpeg', 'image/png', 'image/webp']);
const el = (tag, text, css) => { const node = document.createElement(tag); if (text !== undefined) node.textContent = text; if (css) node.className = css; return node; };
const pct = value => (Number(value) * 100).toFixed(1).replace('.', ',') + '%';
const num = (value, digits = 3) => Number(value).toFixed(digits).replace('.', ',');
const idNumber = value => Number(value).toLocaleString('id-ID');
const meanStd = (s, asPct = true) => asPct ? pct(s.mean) + ' ± ' + (s.std * 100).toFixed(1).replace('.', ',') : num(s.mean) + ' ± ' + num(s.std);
const interval = ci => ci ? pct(ci[0]) + ' – ' + pct(ci[1]) : '—';
const mb = bytes => (bytes / 1e6).toFixed(2).replace('.', ',') + ' MB';
const sum = values => values.reduce((a, b) => a + b, 0);
const swatch = key => el('span', undefined, 'swatch ' + key);

let report = null, models = null, modelInfo = null, current = null, engine = null, siteConfig = {examples: false};
let modelReady = false, busy = false, selected = [], predictions = [], objectUrls = [], previewURL = '';
const isSingleSplit = () => report?.study_type === 'single_split';

function download(value, name) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], {type: 'application/json'}));
  const a = el('a'); a.href = url; a.download = name; document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function stats(target, items) {
  $(target).replaceChildren(...items.map(([name, value, note, css]) => {
    const card = el('div', undefined, 'stat' + (css ? ' ' + css : '')); card.append(el('span', name), el('strong', String(value)));
    if (note) card.append(el('small', note)); return card;
  }));
}
function table(target, headers, rows, highlight) {
  const t = el('table'), head = el('thead'), tr = el('tr'), body = el('tbody');
  headers.forEach(x => { const th = el('th', x); th.scope = 'col'; tr.append(th); }); head.append(tr);
  rows.forEach((values, i) => {
    const row = el('tr'); if (highlight?.(i)) row.className = 'highlight';
    values.forEach(x => { const td = el('td'); if (x instanceof Node) td.append(x); else td.textContent = String(x); row.append(td); });
    body.append(row);
  });
  t.append(head, body); (typeof target === 'string' ? $(target) : target).replaceChildren(t);
}
function confusionTable(target, matrix) {
  const names = classKeys.map(k => labels[k]), t = el('table', undefined, 'cm'), head = el('thead'), tr = el('tr'), body = el('tbody');
  ['Label \\ Prediksi', ...names].forEach(x => { const th = el('th', x); th.scope = 'col'; tr.append(th); }); head.append(tr);
  const max = Math.max(1, ...matrix.flat());
  matrix.forEach((row, i) => {
    const r = el('tr'), th = el('th', names[i], 'row'); th.scope = 'row'; r.append(th);
    row.forEach((v, j) => {
      const td = el('td', String(v), 'cell' + (v ? (i === j ? ' diag-ok' : ' off') : ''));
      if (v) td.style.background = (i === j ? 'rgba(46,125,75,' : 'rgba(192,57,43,') + (0.08 + 0.3 * v / max).toFixed(2) + ')';
      r.append(td);
    });
    body.append(r);
  });
  t.append(head, body); (typeof target === 'string' ? $(target) : target).replaceChildren(t);
}
// Builds a panel inside #eval-body (or another container) and returns the element for its content.
function panel(container, title, description, id) {
  const box = el('div', undefined, 'panel'); box.append(el('h3', title));
  if (description) box.append(el('p', description, 'muted'));
  const content = el('div', undefined, 'table-wrap'); if (id) content.id = id;
  box.append(content); $(container).append(box);
  return content;
}
function setPanel(id) {
  if (!$(id)?.classList.contains('tab-panel')) id = 'prediction';
  document.querySelectorAll('.tab-panel').forEach(n => n.hidden = n.id !== id);
  document.querySelectorAll('.tab').forEach(n => { const active = n.dataset.panel === id; n.classList.toggle('active', active); n.setAttribute('aria-pressed', String(active)); });
}
document.querySelectorAll('.tab').forEach(button => button.onclick = () => { history.replaceState(null, '', '#' + button.dataset.panel); setPanel(button.dataset.panel); });
window.addEventListener('hashchange', () => setPanel(location.hash.slice(1)));
function setStatus(text, state) {
  $('status-text').textContent = text;
  $('status').className = 'status ' + (state || 'loading');
}
async function getJSON(path, optional) {
  const response = await fetch(path);
  if (!response.ok) { if (optional) return null; throw Error(path + ' tidak dapat dimuat'); }
  return response.json();
}

// ---------------- inference engine (worker, with main-thread fallback) ----------------
function createEngine() {
  const pending = new Map();
  let worker = null, local = null, counter = 0, workerFailed = false;
  function startWorker(specUrl, weightsUrl) {
    return new Promise((resolve, reject) => {
      if (!worker) {
        try { worker = new Worker('worker.js'); } catch (error) { reject(error); return; }
        worker.onmessage = event => {
          const m = event.data, task = pending.get(m.id);
          if (m.type === 'progress') { task?.onProgress?.(m.done, m.total); return; }
          if (!task) return;
          pending.delete(m.id);
          if (m.type === 'error') task.reject(Error(m.message)); else task.resolve(m.type === 'ready' ? m.spec : m.result);
        };
        worker.onerror = event => { event.preventDefault?.(); pending.forEach(t => t.reject(Error('Worker berhenti.'))); pending.clear(); };
      }
      const id = ++counter;
      pending.set(id, {resolve, reject});
      worker.postMessage({type: 'init', id, specUrl, weightsUrl});
    });
  }
  async function startLocal(specUrl, weightsUrl) {
    const [specResponse, weightsResponse] = await Promise.all([fetch(specUrl), fetch(weightsUrl)]);
    if (!specResponse.ok || !weightsResponse.ok) throw Error('Model tidak dapat diunduh.');
    const spec = await specResponse.json();
    local = TomatoCore.createModel(spec, await weightsResponse.arrayBuffer());
    return spec;
  }
  return {
    async load(entry) {
      const specUrl = new URL(entry.path + 'model.json', location.href).href;
      const weightsUrl = new URL(entry.path + 'weights.bin', location.href).href;
      if (!workerFailed) {
        try { return await startWorker(specUrl, weightsUrl); }
        catch (error) { if (/checksum|Ukuran bobot|tidak didukung|Format/.test(error.message)) throw error; workerFailed = true; worker?.terminate(); worker = null; console.warn('Worker gagal, memakai thread utama:', error); }
      }
      return startLocal(specUrl, weightsUrl);
    },
    predict(image, onProgress) {
      if (worker && !workerFailed) {
        const id = ++counter;
        return new Promise((resolve, reject) => {
          pending.set(id, {resolve, reject, onProgress});
          worker.postMessage({type: 'predict', id, rgba: image.rgba, width: image.width, height: image.height}, [image.rgba]);
        });
      }
      return new Promise(resolve => setTimeout(resolve, 20)).then(() => {
        const result = TomatoCore.predictPixels(local, new Uint8ClampedArray(image.rgba), image.width, image.height, 4, onProgress);
        delete result._raw_views; return result;
      });
    },
  };
}

// ---------------- image decoding ----------------
function isHeic(file) { return /image\/hei[cf]/i.test(file.type) || /\.(heic|heif)$/i.test(file.name || ''); }
async function decodeImage(blob) {
  if (isHeic(blob)) throw Error('Format HEIC/HEIF (foto iPhone) belum didukung. Ubah ke JPG: Pengaturan → Kamera → Format → “Paling Kompatibel”, atau ekspor/bagikan foto sebagai JPG.');
  if (!ALLOWED.has(blob.type)) throw Error('Format tidak didukung. Gunakan JPG, PNG, atau WebP.');
  if (blob.size > MAX_BYTES) throw Error('Ukuran maksimal 10 MB per foto.');
  const url = URL.createObjectURL(blob);
  try {
    const img = new Image();
    img.decoding = 'async'; img.src = url;
    try { await img.decode(); } catch { throw Error('Foto tidak dapat dibaca atau rusak.'); }
    const w = img.naturalWidth, h = img.naturalHeight;   // the browser applies EXIF orientation
    if (!w || !h) throw Error('Foto kosong.');
    if (w * h > MAX_PIXELS) throw Error('Gambar melebihi 25 megapiksel (' + w + ' × ' + h + ').');
    let cw = w, ch = h;
    if (w * h > MAX_CANVAS_PIXELS) { const s = Math.sqrt(MAX_CANVAS_PIXELS / (w * h)); cw = Math.floor(w * s); ch = Math.floor(h * s); }
    const canvas = document.createElement('canvas'); canvas.width = cw; canvas.height = ch;
    const ctx = canvas.getContext('2d', {willReadFrequently: true});
    ctx.imageSmoothingQuality = 'high'; ctx.drawImage(img, 0, 0, cw, ch);
    return {rgba: ctx.getImageData(0, 0, cw, ch).data.buffer, width: cw, height: ch, original: [w, h], downscaled: cw !== w};
  } finally { URL.revokeObjectURL(url); }
}

// ---------------- "Tentang demo ini": every number comes from model_info.json (written by export_web + js_parity) ----------------
const sci = value => Number(value).toExponential(1).replace('.', ',');
const exportDate = iso => { const d = new Date(iso); return isNaN(d) ? '—' : d.toLocaleDateString('id-ID', {day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC'}); };
function renderAbout(id) {
  const parity = $('about-parity'), facts = $('about-facts');
  const entry = modelInfo?.models?.find(m => m.id === id) || modelInfo?.models?.find(m => m.id === modelInfo.default);
  if (!entry) { parity.textContent = 'Angka paritas tidak tersedia (model_info.json tidak dapat dimuat). Bobot yang dijalankan di sini berasal dari model TensorFlow/Keras yang sama, tetapi selisihnya tidak dapat ditampilkan.'; facts.replaceChildren(); return; }
  const js = entry.js_parity;
  parity.replaceChildren();
  parity.append('Hasil di browser dan di TensorFlow identik secara praktis (tidak harus identik bit demi bit). ');
  if (js) parity.append('Pada ' + js.cases + ' kasus uji, selisih maksimum keluaran jaringan antara JavaScript dan Keras adalah ', el('b', sci(js.max_network_error_vs_keras)),
    ' (batas uji ' + sci(js.tolerance.network) + '). Termasuk pengubahan ukuran foto, selisih probabilitas maksimum ', el('b', sci(js.max_end_to_end_probability_error)), ' (batas ' + sci(js.tolerance.end_to_end_probability) + ').');
  else parity.append('Selisih bobot yang diekspor terhadap Keras: ', el('b', sci(entry.keras_parity_max_abs_error)), '. Paritas JavaScript belum tercatat untuk model ini.');
  const rows = [['Model', entry.id + ' · ' + entry.architecture + ' CNN · ' + idNumber(entry.parameters) + ' parameter'],
    ['Ukuran bobot', mb(entry.weights_bytes)],
    ['Diekspor', exportDate(modelInfo.exported_at) + ' · TensorFlow ' + modelInfo.export_environment.tensorflow + ', Keras ' + modelInfo.export_environment.keras]];
  facts.replaceChildren(...rows.flatMap(([k, v]) => [el('dt', k), el('dd', v)]));
}

// ---------------- model selection ----------------
function renderModelCard(entry, loadMs) {
  const rows = [['Arsitektur', entry.architecture + ' CNN · ' + idNumber(entry.parameters) + ' parameter'],
    ['Ukuran bobot', mb(entry.weights_bytes) + (loadMs !== undefined ? ' · dimuat dalam ' + Math.round(loadMs) + ' ms' : '')]];
  if (entry.validation) {
    const v = entry.validation, correct = Math.round(v.accuracy * v.n_images);
    rows.push(['Akurasi validation', pct(v.accuracy) + ' (' + correct + '/' + v.n_images + ' foto) · macro-F1 ' + num(v.macro_f1, 2)]);
  }
  if (entry.cv) rows.push(['Akurasi validasi silang', meanStd(entry.cv.accuracy) + ' (95%: ' + interval(entry.cv.accuracy_wilson95_mean_seed) + ', ' + entry.cv.oof_images_per_seed + ' foto/seed)']);
  if (entry.test) rows.push(['Akurasi test terkunci', meanStd(entry.test.accuracy) + ' (95%: ' + interval(entry.test.accuracy_wilson95_mean_seed) + ', hanya ' + entry.test.n_images + ' foto)']);
  const rt = entry.review_threshold;
  rows.push(['Ambang “perlu tinjauan”', pct(entry.confidence_threshold) + (rt ? ' · dikalibrasi dari prediksi out-of-fold' : ' · bawaan, belum dikalibrasi')]);
  rows.push(['Status', entry.role === 'selected' ? 'Model terpilih menurut aturan seleksi' : 'Model pembanding']);
  $('model-card').replaceChildren(...rows.flatMap(([k, v]) => [el('dt', k), el('dd', v)]));
}
async function loadModel(id) {
  const entry = models.models.find(m => m.id === id) || models.models[0];
  modelReady = false; setControls();
  setStatus('Memuat model ' + entry.id + ' (' + mb(entry.weights_bytes) + ')…');
  const t0 = performance.now();
  try {
    await engine.load(entry);
    current = entry; modelReady = true;
    const loadMs = performance.now() - t0;
    renderModelCard(entry, loadMs); renderAbout(entry.id);
    setStatus('Model ' + entry.id + ' (' + entry.architecture + ' CNN) dimuat dalam ' + Math.round(loadMs) + ' ms. Berjalan di browser Anda; foto tidak diunggah.', 'ready');
    document.documentElement.dataset.modelLoadMs = String(Math.round(loadMs));
  } catch (error) {
    setStatus('Model tidak dapat dimuat: ' + error.message + ' Muat ulang halaman atau coba browser lain.', 'failed');
  }
  setControls();
}

// ---------------- prediction UI ----------------
function clearObjectUrls() { objectUrls.forEach(u => URL.revokeObjectURL(u)); objectUrls = []; }
function renderPrediction(card, result, truth, image) {
  card.querySelector('.pending')?.remove(); card.querySelector('.bar-progress')?.remove();
  const meta = card.querySelector('.meta');
  const review = result.needs_review;
  const h3 = el('h3'); h3.append(el('span', 'Dugaan: ' + labels[result.label], 'verdict ' + result.label), el('span', review ? 'Perlu tinjauan' : 'Kandidat', review ? 'badge warn' : 'badge ok'));
  meta.append(h3, el('p', 'Probabilitas model ' + pct(result.confidence) + ' · bukan kepastian', 'confidence'));
  const probs = el('div', undefined, 'probs');
  for (const key of classKeys) {
    const value = result.probabilities[key], wrap = el('div');
    const row = el('div', undefined, 'prob-label'); row.append(el('span', labels[key]), el('span', pct(value)));
    const bar = el('progress', undefined, key); bar.max = 1; bar.value = value; bar.setAttribute('aria-label', 'Probabilitas ' + labels[key] + ' ' + pct(value));
    wrap.append(row, bar); probs.append(wrap);
  }
  card.append(probs, el('p', result.notice, review ? 'review' : 'review ok'));
  if (truth) {
    const ok = truth === result.label;
    card.append(el('p', 'Label pemilik: ' + labels[truth] + (ok ? ' · sesuai dengan dugaan model.' : ' · berbeda dari dugaan model.'), ok ? 'truth ok' : 'truth miss'));
  }
  const details = el('details', undefined, 'diag'), list = el('ul');
  details.append(el('summary', 'Detail diagnostik'));
  if (result.stability) list.append(el('li', 'Konsistensi pada variasi ringan (flip, ±10% kecerahan): ' + pct(result.stability.agreement)));
  list.append(el('li', 'Selisih dua kelas teratas: ' + pct(result.probability_margin)));
  const q = result.quality;
  if (q) list.append(el('li', 'Kecerahan ' + num(q.brightness, 2) + ' · kontras ' + num(q.contrast, 3) + ' · porsi warna tomat ' + pct(q.tomato_color_fraction)));
  const reasons = [...(result.review_reasons || [])];
  if (image?.downscaled) reasons.push('Foto ' + image.original.join(' × ') + ' diperkecil di browser sebelum diproses; hasil bisa sedikit berbeda dari aplikasi Python.');
  reasons.forEach(r => list.append(el('li', r)));
  details.append(list);
  if (review) details.open = true;
  card.append(details);
}
function renderSummary() {
  const ok = predictions.filter(p => !p.error), chip = $('result-summary');
  if (ok.length < 2) { chip.hidden = true; return; }
  const counts = classKeys.map(k => [k, ok.filter(p => p.label === k).length]).filter(([, n]) => n);
  chip.textContent = ok.length + ' foto · ' + counts.map(([k, n]) => n + ' ' + labels[k].toLowerCase()).join(' · ') +
    ' · ' + ok.filter(p => p.needs_review).length + ' perlu tinjauan';
  chip.hidden = false;
}
async function analyze(items) {
  if (busy || !modelReady || !items.length) return;
  busy = true; setControls(); predictions = []; clearObjectUrls(); $('result').replaceChildren(); $('result-summary').hidden = true;
  if (items.length > MAX_FILES) { $('result').append(el('p', 'Hanya ' + MAX_FILES + ' foto pertama yang dianalisis (dipilih ' + items.length + ').', 'review')); items = items.slice(0, MAX_FILES); }
  if (matchMedia('(max-width: 980px)').matches) $('result').closest('.panel').scrollIntoView({behavior: 'smooth', block: 'start'});
  for (let i = 0; i < items.length; i++) {
    const {blob, name, truth} = items[i];
    const card = el('article', undefined, 'prediction-card');
    const head = el('div', undefined, 'result-head'), meta = el('div', undefined, 'meta');
    const thumb = el('img'); thumb.alt = 'Foto yang dianalisis: ' + name;
    thumb.onerror = () => { thumb.hidden = true; };
    if (ALLOWED.has(blob.type)) { const url = URL.createObjectURL(blob); objectUrls.push(url); thumb.src = url; } else thumb.hidden = true;
    meta.append(el('small', name)); head.append(thumb, meta); card.append(head);
    const message = el('p', 'Menganalisis foto ' + (i + 1) + ' / ' + items.length + '…', 'pending');
    const progress = el('div', undefined, 'bar-progress'), fill = el('span'); progress.append(fill);
    meta.append(message, progress);
    $('result').append(card);
    try {
      const image = await decodeImage(blob);
      const result = await engine.predict({rgba: image.rgba, width: image.width, height: image.height}, (done, total) => {
        message.textContent = 'Menganalisis foto ' + (i + 1) + ' / ' + items.length + ' · tampilan ' + done + '/' + total + '…';
        fill.style.width = Math.round(100 * done / total) + '%';
      });
      predictions.push({file: name, downscaled_in_browser: image.downscaled, ...result});
      renderPrediction(card, result, truth, image);
    } catch (error) {
      progress.remove();
      message.textContent = name + ': ' + error.message; message.className = 'error'; message.setAttribute('role', 'alert');
      predictions.push({file: name, error: error.message});
    }
  }
  busy = false; setControls(); renderSummary();
}
function setControls() {
  $('predict').disabled = busy || !modelReady || !selected.length;
  $('download-predictions').disabled = busy || !predictions.some(p => !p.error);
  $('upload').disabled = $('camera').disabled = busy;
  $('model-select').disabled = busy || !models || models.models.length < 2;
  document.querySelectorAll('.example-button').forEach(b => b.disabled = busy || !modelReady);
}
function selectFiles(list) {
  selected = Array.from(list); predictions = [];
  if (previewURL) { URL.revokeObjectURL(previewURL); previewURL = ''; }
  const first = selected.find(f => ALLOWED.has(f.type));
  $('selection').hidden = !selected.length;
  $('preview-frame').hidden = !first;
  if (first) { previewURL = URL.createObjectURL(first); $('preview').src = previewURL; }
  $('file-info').textContent = selected.length ? selected.length + ' foto dipilih' + (first ? ' · pratinjau: ' + first.name : '') +
    (selected.length > MAX_FILES ? ' · hanya ' + MAX_FILES + ' pertama yang akan dianalisis' : '') : 'Belum ada foto dipilih.';
  setControls();
}
$('upload').onchange = () => selectFiles($('upload').files);
$('camera').onchange = () => selectFiles($('camera').files);
$('predict').onclick = () => analyze(selected.map(file => ({blob: file, name: file.name})));
$('model-select').onchange = () => loadModel($('model-select').value);
$('download-predictions').onclick = () => download({generated_by: 'Tomato Vision static demo (inference in browser)',
  model: current ? {id: current.id, architecture: current.architecture, review_threshold: current.confidence_threshold} : undefined,
  predictions}, 'tomato-predictions.json');
const drop = $('drop');
['dragenter', 'dragover'].forEach(type => drop.addEventListener(type, event => { event.preventDefault(); drop.classList.add('dragging'); }));
['dragleave', 'drop'].forEach(type => drop.addEventListener(type, event => { event.preventDefault(); drop.classList.remove('dragging'); }));
drop.addEventListener('drop', event => {
  const files = Array.from(event.dataTransfer?.files || []);
  if (!files.length || busy) return;
  const transfer = new DataTransfer(); files.forEach(f => transfer.items.add(f)); $('upload').files = transfer.files; selectFiles(transfer.files);
});

// ---------------- static content from report.json ----------------
function renderHeroFacts() {
  const facts = [];
  const entry = models.models.find(m => m.id === models.default);
  if (report?.dataset) facts.push([String(report.dataset.images), ' foto berlabel · ', String(report.dataset.sessions), ' sesi']);
  if (entry) facts.push(['CNN ', idNumber(entry.parameters), ' parameter']);
  if (entry) facts.push(['Bobot ', mb(entry.weights_bytes)]);
  facts.push(['Tanpa unggah ke server']);
  $('hero-facts').replaceChildren(...facts.map(parts => { const li = el('li'); parts.forEach((p, i) => li.append(i % 2 === 1 ? p : el('b', p))); return li; }));
}
function renderExamples() {
  const examples = siteConfig.examples ? (report?.examples || []) : [];
  $('examples-block').hidden = $('gallery-panel').hidden = !examples.length;
  const box = $('examples'), gallery = $('gallery');
  box.replaceChildren(); gallery.replaceChildren();
  const fromValidation = examples.some(e => /^validation/.test(e.split));
  $('examples-note').textContent = fromValidation
    ? 'Foto contoh berasal dari split validation (dua sesi yang tidak dipakai untuk melatih bobot). Test terkunci tidak dipublikasikan.'
    : 'Foto contoh berasal dari test terkunci: tidak dipakai untuk training, pemilihan model, maupun kalibrasi ambang.';
  $('gallery-title').textContent = fromValidation ? 'Foto validation' : 'Foto test terkunci';
  examples.forEach((example, index) => {
    const button = el('button', undefined, 'example-button'); button.type = 'button';
    button.setAttribute('aria-label', 'Analisis foto contoh ' + (index + 1) + ' (label pemilik: ' + labels[example.label] + ')');
    const img = el('img'); img.src = example.file; img.alt = ''; img.width = 92; img.height = 92; img.loading = 'lazy';
    const tag = el('span', undefined, 'tag'); tag.append(swatch(example.label), el('span', labels[example.label]));
    button.append(img, tag);
    button.onclick = async () => {
      if (busy) return;
      $('result').replaceChildren(el('p', 'Memuat foto contoh…', 'pending')); $('result-summary').hidden = true;
      try { const response = await fetch(example.file); if (!response.ok) throw Error();
        const blob = new Blob([await response.blob()], {type: 'image/jpeg'});
        selectFiles([]); analyze([{blob, name: 'Contoh ' + (index + 1) + ' (' + example.source + ')', truth: example.label}]); }
      catch { $('result').replaceChildren(el('p', 'Foto contoh tidak dapat dimuat.', 'error')); }
    };
    box.append(button);
    const card = el('article', undefined, 'photo'), photo = el('img'), caption = el('p');
    photo.src = example.file; photo.loading = 'lazy'; photo.alt = 'Foto ' + example.split + ', label pemilik: ' + labels[example.label];
    caption.append(swatch(example.label), labels[example.label]);
    card.append(photo, el('small', example.source + ' · ' + example.split), caption);
    gallery.append(card);
  });
}
function renderClassBars(counts, note) {
  const total = sum(classKeys.map(k => counts[k] || 0));
  $('class-bars').replaceChildren(...classKeys.map(k => {
    const n = counts[k] || 0, box = el('div', undefined, 'class-bar'), row = el('div', undefined, 'row'), track = el('div', undefined, 'track'), fill = el('div', undefined, 'fill ' + k);
    row.append(el('span', labels[k]), el('span', n + ' foto · ' + pct(n / total)));
    fill.style.width = (100 * n / total) + '%'; track.append(fill); box.append(row, track); return box;
  }));
  $('class-note').textContent = note || '';
}
const LEGACY_AUDIT = 'Versi lama (own_v1) mengelompokkan foto per “set”. Audit menunjukkan setiap set berisi dua sesi berbeda, dan ' +
  '30 foto meja kayu (kemungkinan satu buah yang sama) tersebar di train, validasi, dan test, sehingga akurasi 100% versi lama tidak bisa dipercaya. ' +
  'Tidak ditemukan foto hampir kembar (jarak dHash/pHash minimum 11/12 bit). ';

function renderDatasetSingle() {
  const d = report.dataset, s = d.splits, total = key => sum(Object.values(s[key] || {}));
  const ss = d.split_sessions;
  stats('data-stats', [['Foto berlabel', d.images, 'Semua dikonfirmasi pemilik'], ['Sesi pemotretan', d.sessions, 'Satu sesi = satu grup'],
    ['Validation', total('val') + ' foto', ss.val.length + ' sesi · memilih model'], ['Test terkunci', total('test') + ' foto', ss.test.length + ' sesi · belum dinilai']]);
  $('data-state').textContent = 'Dataset kecil · ' + d.sessions + ' sesi';
  const segarRef = (s.train?.segar || 0);
  renderClassBars(d.class_counts, 'Kelas segar terwakili berlebihan: ' + segarRef + ' dari ' + total('train') +
    ' foto training adalah segar, termasuk 22 foto dari satu latar (meja kayu). Model berisiko mengaitkan latar dengan kelas.');
  $('audit-text').textContent = LEGACY_AUDIT + 'Versi ini memakai sesi pemotretan sebagai grup: semua sesi meja kayu selalu di training, ' +
    'validation (' + ss.val.join(', ') + ') dan test (' + ss.test.join(', ') + ') hanya dari sesi berlatar lain.';
  table('split-table', ['Split', ...classKeys.map(k => labels[k]), 'Foto', 'Sesi'],
    [['train', 'train'], ['validation', 'val'], ['test terkunci', 'test']].map(([name, key]) =>
      [name, ...classKeys.map(k => s[key]?.[k] || 0), total(key), ss[key].length + (key === 'train' ? ' sesi' : ' (' + ss[key].join(', ') + ')')]));
}
function renderTrainingSingle() {
  const sel = report.selection, rows = report.candidates;
  const items = ['Satu split tetap berbasis sesi pemotretan: ' + report.dataset.split_sessions.train.length + ' sesi training, ' +
      report.dataset.split_sessions.val.length + ' sesi validation, ' + report.dataset.split_sessions.test.length + ' sesi test terkunci.',
    'CNN dilatih dari bobot acak (tanpa pretrained) dengan augmentasi online; setiap epoch mengambil sampel seimbang per kelas.',
    'Aturan seleksi: macro-F1 validation tertinggi, lalu loss validation terendah.',
    'Validation juga dipakai untuk early stopping, sehingga skornya cenderung optimistis.',
    'Test terkunci ' + (sel.test_evaluated ? 'sudah dinilai.' : 'belum dibuka; angka test belum ada.')];
  $('protocol').replaceChildren(...items.map(t => el('li', t)));
  $('ablation-title').textContent = 'Kandidat model (' + report.study + ')';
  $('ablation-desc').textContent = 'Dua kandidat dilatih pada data yang sama. Akurasi train dihitung tanpa augmentasi.';
  table('ablation', ['Kandidat', 'Arsitektur', 'Param', 'Dropout', 'Epoch terbaik / dijalankan', 'Akurasi train', 'Akurasi validation', 'Macro-F1 val', 'Loss val'],
    rows.map(r => [r.name, r.architecture, idNumber(r.parameters), r.dropout ?? '—', r.best_epoch + ' / ' + r.epochs_run, pct(r.train.accuracy),
      pct(r.validation.accuracy) + ' (' + Math.round(r.validation.accuracy * r.validation.n_images) + '/' + r.validation.n_images + ')',
      num(r.validation.macro_f1), num(r.validation.loss)]), i => rows[i].name === sel.selected);
  $('selection-note').textContent = 'Terpilih: ' + sel.selected + '. ' + (sel.limitation || '');
  renderCurves('Akurasi dan loss per epoch (sumbu dimulai dari 0). Bobot yang dipakai berasal dari epoch terbaik pada validation (' +
    rows.map(r => r.name + ': epoch ' + r.best_epoch).join(', ') + '). Validation hanya ' + rows[0].validation.n_images + ' foto, sehingga kurvanya sangat berfluktuasi.');
}
function renderCurves(note) {
  const box = $('curves'); box.replaceChildren();
  models.models.forEach(m => {
    const figure = el('figure'), img = el('img'); img.src = 'img/learning_curve_' + m.id + '.png'; img.loading = 'lazy';
    img.alt = 'Kurva akurasi dan loss training model ' + m.id; img.className = 'plot';
    figure.append(img, el('figcaption', m.id + (m.role === 'selected' ? ' (terpilih)' : ' (pembanding)'))); box.append(figure);
  });
  $('curves-note').textContent = note;
}
function renderEvaluationSingle() {
  const sel = report.selection, rows = report.candidates, best = rows.find(r => r.name === sel.selected), v = best.validation;
  const correct = Math.round(v.accuracy * v.n_images);
  $('evaluation-notice').textContent = 'Validation hanya ' + v.n_images + ' foto dari ' + report.dataset.split_sessions.val.length +
    ' sesi dan juga dipakai untuk early stopping serta memilih model; satu foto salah mengubah akurasi ' + pct(1 / v.n_images) +
    '. Test terkunci belum dinilai. Angka ini indikasi awal, bukan bukti generalisasi.';
  stats('eval-stats', [['Akurasi validation', pct(v.accuracy), correct + ' dari ' + v.n_images + ' foto · ' + best.name],
    ['Macro-F1 validation', num(v.macro_f1), 'Rata-rata F1 tiga kelas'],
    ['Loss validation', num(v.loss), 'ECE ' + num(v.expected_calibration_error)],
    ['Test terkunci', 'Belum dinilai', report.dataset.split_sessions.test.length + ' sesi disimpan untuk penilaian sekali', 'muted-stat']]);
  const body = $('eval-body'); body.replaceChildren();
  table(panel('eval-body', 'Perbandingan kandidat (validation)', null, 'model-compare'), ['Model', 'Peran', 'Akurasi', 'Macro-F1', 'Loss', 'ECE', 'Akurasi train', 'Selisih train − val'],
    rows.map(r => [r.name, r.role === 'selected' ? 'terpilih' : 'pembanding', pct(r.validation.accuracy), num(r.validation.macro_f1), num(r.validation.loss),
      num(r.validation.expected_calibration_error), pct(r.train.accuracy), ((r.train.accuracy - r.validation.accuracy) * 100).toFixed(1).replace('.', ',') + ' poin']),
    i => rows[i].name === sel.selected);
  const grid = el('div', undefined, 'cm-grid');
  rows.forEach(r => { const box = el('div'), wrap = el('div', undefined, 'table-wrap'); box.append(el('h4', r.name + (r.role === 'selected' ? ' (terpilih)' : '')), wrap);
    confusionTable(wrap, r.validation.confusion_matrix); grid.append(box); });
  const cmPanel = panel('eval-body', 'Confusion matrix (validation)', 'Baris: label pemilik. Kolom: prediksi model. Hijau = benar, merah = salah.', 'confusion');
  cmPanel.className = ''; cmPanel.append(grid);
  table(panel('eval-body', 'Precision, recall, dan F1 per kelas (' + best.name + ', validation)', null, 'class-metrics'), ['Kelas', 'Precision', 'Recall', 'F1', 'Foto'],
    classKeys.map(k => { const m = v.report[k]; const name = el('span'); name.append(swatch(k), ' ' + labels[k]); return [name, num(m.precision, 2), num(m.recall, 2), num(m['f1-score'], 2), m.support]; }));
  const predRows = best.predictions.map((p, i) => {
    const cells = [p.source, p.session, labels[p.label]];
    rows.forEach(r => {
      const q = r.predictions[i], top = classKeys.reduce((a, b) => q.probabilities[a] >= q.probabilities[b] ? a : b);
      cells.push(el('span', labels[top] + ' · ' + pct(q.probabilities[top]), top === p.label ? 'ok-text' : 'miss-text'));
    });
    return cells;
  });
  table(panel('eval-body', 'Prediksi per foto validation', 'Prediksi asli dari TensorFlow saat training (bukan dari browser). Kolom per kandidat: kelas teratas dan probabilitasnya.', 'val-predictions'),
    ['Foto', 'Sesi', 'Label pemilik', ...rows.map(r => r.name)], predRows);
  renderLegacy([['own_v1', 'validation (1 split, bocor)', report.legacy_own_v1?.validation_images, report.legacy_own_v1 && pct(report.legacy_own_v1.validation_accuracy)],
    ['own_v1', 'test (1 split, bocor)', report.legacy_own_v1?.test_images, report.legacy_own_v1 && pct(report.legacy_own_v1.test_accuracy)],
    ...rows.map(r => [report.study + ' · ' + r.name, 'validation (split sesi)', r.validation.n_images, pct(r.validation.accuracy)])]);
}
function renderLegacy(rows) {
  const legacy = report.legacy_own_v1;
  if (!legacy) return;
  const target = panel('eval-body', 'Dibandingkan dengan versi lama (own_v1)', null, 'legacy');
  table(target, ['Versi', 'Evaluasi', 'Foto', 'Akurasi'], rows);
  target.after(el('p', legacy.note, 'footnote'));
}

// ---- own_v2 session cross-validation study (report.cv / report.test) ----
function renderDatasetCv() {
  const d = report.dataset;
  stats('data-stats', [['Foto berlabel', d.images, 'Dipakai training & evaluasi'], ['Belum berlabel', d.unlabeled, 'Tidak dipakai'],
    ['Sesi pemotretan', d.sessions, 'Satu sesi = satu grup'], ['Test terkunci', sum(Object.values(d.roles.test)) + ' foto', report.protocol.locked_test_sessions.length + ' sesi, dinilai sekali']]);
  $('data-state').textContent = 'Dataset kecil · ' + d.sessions + ' sesi';
  renderClassBars(d.class_counts);
  $('audit-text').textContent = LEGACY_AUDIT + 'Versi ini memakai sesi pemotretan sebagai grup: sesi meja kayu selalu di training, validasi dan test hanya dari sesi berlatar lain.';
  table('split-table', ['Peran', ...classKeys.map(k => labels[k]), 'Foto'], [['development (train + validasi silang)', 'dev'], ['test terkunci', 'test']].map(([name, key]) => {
    const c = d.roles[key]; return [name, ...classKeys.map(k => c[k] || 0), sum(classKeys.map(k => c[k] || 0))];
  }));
}
function renderTrainingCv() {
  const p = report.protocol, s = report.selection;
  const items = ['Grup: ' + p.grouping + '.', 'Test terkunci: sesi ' + p.locked_test_sessions.join(', ') + ' (dipilih acak dengan seed 42 sebelum training).',
    'Validasi silang: 5 fold atas sesi development lain; seed ' + p.seeds.join(', ') + '.', 'Aturan seleksi: ' + p.selection_rule,
    'Keterbatasan: ' + p.known_bias, ...p.amendments.map(a => 'Amandemen protokol (' + a.date.slice(0, 10) + '): ' + a.change + '. Alasan: ' + a.reason)];
  $('protocol').replaceChildren(...items.map(t => el('li', t)));
  $('ablation-title').textContent = 'Ablation: satu variabel per eksperimen';
  $('ablation-desc').textContent = 'Validasi silang 5 fold berbasis sesi × 5 seed (25 run per baris). Akurasi adalah prediksi out-of-fold, rata-rata ± simpangan baku antar seed.';
  const rows = report.cv.filter(r => r.complete);
  table('ablation', ['Konfigurasi', 'Param', 'Akurasi', 'Wilson 95%', 'Bootstrap sesi 95%', 'Macro-F1', 'Log loss', 'ECE', 'Run dgn val loss > 3', 'Best epoch (median)'],
    rows.map(r => [r.name, idNumber(r.parameters), meanStd(r.accuracy), interval(r.accuracy_wilson95_mean_seed), interval(r.accuracy_session_bootstrap95),
      meanStd(r.macro_f1, false), meanStd(r.loss, false), meanStd(r.ece, false), r.stability.runs_with_spike + '/' + r.stability.runs, r.stability.best_epoch_median]),
    i => rows[i].name === s.selected);
  $('selection-note').textContent = 'Terpilih: ' + s.selected + ' (dibekukan ' + s.frozen.replace('T', ' ') + ', sebelum test diakses). ' +
    (s.tied_within_one_std.length > 1 ? 'Seri dalam 1 simpangan baku: ' + s.tied_within_one_std.join(', ') + '.' : '');
  renderCurves('Model final dilatih ulang pada seluruh data development tanpa data validasi, dengan jumlah epoch dan jadwal learning rate dari median run validasi silang.');
}
function renderEvaluationCv() {
  const s = report.selection.selected, cv = report.cv.find(r => r.name === s), test = report.test.models[s];
  $('evaluation-notice').textContent = 'Test terkunci hanya ' + test.n_images + ' foto dari ' + report.test.sessions.length + ' sesi dan dinilai satu kali setelah model dipilih. ' +
    'Interval kepercayaan sangat lebar; angka ini indikasi awal, bukan bukti generalisasi.';
  stats('eval-stats', [['Akurasi validasi silang', pct(cv.accuracy.mean), '± ' + (cv.accuracy.std * 100).toFixed(1).replace('.', ',') + ' antar seed · 95%: ' + interval(cv.accuracy_wilson95_mean_seed)],
    ['Macro-F1 validasi silang', num(cv.macro_f1.mean), '± ' + num(cv.macro_f1.std)],
    ['Akurasi test terkunci', pct(test.accuracy.mean), '± ' + (test.accuracy.std * 100).toFixed(1).replace('.', ',') + ' · 95%: ' + interval(test.accuracy_wilson95_mean_seed)],
    ['Foto test', test.n_images, '5 seed model final']]);
  $('eval-body').replaceChildren();
  table(panel('eval-body', 'Validasi silang vs test terkunci', null, 'model-compare'), ['Model', 'Peran', 'Akurasi CV', 'Wilson 95% (CV)', 'Akurasi test', 'Wilson 95% (test)', 'Macro-F1 test'],
    Object.entries(report.test.models).map(([name, t]) => { const c = report.cv.find(r => r.name === name);
      return [name, t.role === 'selected' ? 'terpilih' : 'pembanding', meanStd(c.accuracy), interval(c.accuracy_wilson95_mean_seed), meanStd(t.accuracy), interval(t.accuracy_wilson95_mean_seed), meanStd(t.macro_f1, false)]; }));
  confusionTable(panel('eval-body', 'Confusion matrix (out-of-fold)', 'Baris: label sebenarnya. Kolom: prediksi. Dijumlahkan atas 5 seed.', 'confusion'), cv.pooled_over_seeds.confusion_matrix);
  confusionTable(panel('eval-body', 'Confusion matrix (test terkunci)', 'Dijumlahkan atas 5 seed model final.', 'confusion-test'), test.confusion_matrix_pooled);
  table(panel('eval-body', 'Precision, recall, dan F1 per kelas (out-of-fold)', null, 'class-metrics'), ['Kelas', 'Precision', 'Recall', 'F1', 'Prediksi'],
    classKeys.map(k => { const m = cv.pooled_over_seeds.report[k]; return [labels[k], num(m.precision), num(m.recall), num(m['f1-score']), m.support]; }));
  const scen = {darker: 'Lebih gelap (×0,8)', brighter: 'Lebih terang (×1,2)', blur: 'Gaussian blur (radius 1)'};
  table(panel('eval-body', 'Robustness awal (test terkunci)', 'Gambar test digelapkan, diterangkan, atau diburamkan. Bukan pengganti sesi pemotretan baru.', 'robustness'),
    ['Kondisi', 'Akurasi', 'Macro-F1', 'Loss'], [['Asli', meanStd(test.accuracy), meanStd(test.macro_f1, false), meanStd(test.loss, false)],
      ...Object.entries(test.robustness).map(([k, m]) => [scen[k] || k, meanStd(m.accuracy), meanStd(m.macro_f1, false), meanStd(m.loss, false)])]);
  const legacy = report.legacy_own_v1, ref = report.cv.find(r => r.name === 'ref');
  if (legacy) renderLegacy([['own_v1', 'validasi (1 split)', legacy.validation_images, pct(legacy.validation_accuracy)],
    ['own_v1', 'test (1 split)', legacy.test_images, pct(legacy.test_accuracy)],
    ...(ref ? [['konfigurasi own_v1 di protokol baru', 'validasi silang (ref)', ref.oof_images_per_seed, meanStd(ref.accuracy)]] : []),
    ['own_v2 (' + s + ')', 'validasi silang', cv.oof_images_per_seed, meanStd(cv.accuracy)],
    ['own_v2 (' + s + ')', 'test terkunci', test.n_images, meanStd(test.accuracy)]]);
}
$('download-report').onclick = () => report && download(report, 'tomato-vision-report.json');

async function init() {
  setPanel(location.hash.slice(1));
  engine = createEngine();
  try {
    [models, report, siteConfig, modelInfo] = await Promise.all([getJSON('data/models.json'), getJSON('data/report.json', true), getJSON('data/site-config.json', true),
      getJSON('model_info.json', true).catch(() => null)]);
    siteConfig = siteConfig || {examples: false};
  } catch (error) { setStatus('Data demo tidak dapat dimuat: ' + error.message, 'failed'); return; }
  const select = $('model-select');
  select.replaceChildren(...models.models.map(m => { const o = el('option', m.id + ' · ' + m.architecture + (m.role === 'selected' ? ' (terpilih)' : '')); o.value = m.id; return o; }));
  select.value = models.default;
  renderAbout(models.default);
  renderHeroFacts();
  if (report) {
    try {
      renderExamples();
      if (isSingleSplit()) { renderDatasetSingle(); renderTrainingSingle(); renderEvaluationSingle(); }
      else { renderDatasetCv(); renderTrainingCv(); renderEvaluationCv(); }
      const n = report.dataset?.images;
      if (n) $('honest-note').textContent = 'Model ini dilatih dari foto yang sangat terbatas (' + n + ' foto berlabel dari ' + report.dataset.sessions +
        ' sesi pemotretan), sehingga bisa salah pada pencahayaan, latar, kamera, atau varietas yang berbeda. Gunakan sebagai alat bantu, bukan penentu.';
    } catch (error) { console.error(error); $('evaluation-notice').textContent = 'Sebagian data evaluasi tidak dapat ditampilkan.'; }
  } else { $('evaluation-notice').textContent = 'Data evaluasi tidak tersedia.'; renderExamples(); }
  await loadModel(models.default);
}
init();
