'use strict';
// Tomato Vision: static demo for GitHub Pages. Inference runs in the browser (tomato-core.js / worker.js).
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

let report = null, models = null, modelInfo = null, current = null, engine = null, siteConfig = {examples: false};
let modelReady = false, busy = false, selected = [], predictions = [], objectUrls = [], previewURL = '';

function download(value, name) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], {type: 'application/json'}));
  const a = el('a'); a.href = url; a.download = name; document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function stats(target, items) {
  $(target).replaceChildren(...items.map(([name, value, note]) => {
    const card = el('div', undefined, 'stat'); card.append(el('span', name), el('strong', String(value)));
    if (note) card.append(el('small', note)); return card;
  }));
}
function table(target, headers, rows, highlight) {
  const t = el('table'), head = el('thead'), tr = el('tr'), body = el('tbody');
  headers.forEach(x => { const th = el('th', x); th.scope = 'col'; tr.append(th); }); head.append(tr);
  rows.forEach((values, i) => { const row = el('tr'); if (highlight?.(i)) row.className = 'highlight'; values.forEach(x => row.append(el('td', String(x)))); body.append(row); });
  t.append(head, body); $(target).replaceChildren(t);
}
function setPanel(id) {
  if (!$(id)?.classList.contains('tab-panel')) id = 'prediction';
  document.querySelectorAll('.tab-panel').forEach(n => n.hidden = n.id !== id);
  document.querySelectorAll('.tab').forEach(n => { const active = n.dataset.panel === id; n.classList.toggle('active', active); n.setAttribute('aria-pressed', String(active)); });
}
document.querySelectorAll('.tab').forEach(button => button.onclick = () => { location.hash = button.dataset.panel; setPanel(button.dataset.panel); });
window.addEventListener('hashchange', () => setPanel(location.hash.slice(1)));
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
    ['Ukuran bobot', (entry.weights_bytes / 1e6).toFixed(2).replace('.', ',') + ' MB'],
    ['Diekspor', exportDate(modelInfo.exported_at) + ' · TensorFlow ' + modelInfo.export_environment.tensorflow + ', Keras ' + modelInfo.export_environment.keras]];
  facts.replaceChildren(...rows.flatMap(([k, v]) => [el('dt', k), el('dd', v)]));
}

// ---------------- model selection ----------------
function renderModelCard(entry, loadMs) {
  const rows = [['Arsitektur', entry.architecture + ' CNN · ' + idNumber(entry.parameters) + ' parameter'],
    ['Ukuran bobot', (entry.weights_bytes / 1e6).toFixed(2).replace('.', ',') + ' MB' + (loadMs !== undefined ? ' · dimuat dalam ' + Math.round(loadMs) + ' ms' : '')]];
  if (entry.cv) rows.push(['Akurasi validasi silang', meanStd(entry.cv.accuracy) + ' (95%: ' + interval(entry.cv.accuracy_wilson95_mean_seed) + ', ' + entry.cv.oof_images_per_seed + ' foto/seed)']);
  if (entry.test) rows.push(['Akurasi test terkunci', meanStd(entry.test.accuracy) + ' (95%: ' + interval(entry.test.accuracy_wilson95_mean_seed) + ', hanya ' + entry.test.n_images + ' foto)']);
  const rt = entry.review_threshold;
  rows.push(['Ambang “perlu tinjauan”', pct(entry.confidence_threshold) + (rt ? ' · dikalibrasi dari prediksi out-of-fold (target akurasi ' + pct(rt.target_accuracy) + ', cakupan ' + pct(rt.coverage) + ')' : ' · bawaan, belum dikalibrasi')]);
  rows.push(['Status', entry.role === 'selected' ? 'Model terpilih menurut aturan seleksi validasi' : 'Model pembanding']);
  $('model-card').replaceChildren(...rows.flatMap(([k, v]) => [el('dt', k), el('dd', v)]));
}
async function loadModel(id) {
  const entry = models.models.find(m => m.id === id) || models.models[0];
  modelReady = false; setControls();
  $('status').textContent = 'Memuat model ' + entry.id + '…';
  const t0 = performance.now();
  try {
    await engine.load(entry);
    current = entry; modelReady = true;
    const loadMs = performance.now() - t0;
    renderModelCard(entry, loadMs); renderAbout(entry.id);
    $('status').textContent = 'Model ' + entry.id + ' (' + entry.architecture + ' CNN) dimuat dalam ' + Math.round(loadMs) + ' ms. Berjalan di browser Anda; foto tidak diunggah. Periksa hasil evaluasi sebelum menggunakan prediksi.';
    document.documentElement.dataset.modelLoadMs = String(Math.round(loadMs));
  } catch (error) {
    $('status').textContent = 'Model tidak dapat dimuat: ' + error.message + ' Muat ulang halaman atau coba browser lain.';
  }
  setControls();
}

// ---------------- prediction UI ----------------
function clearObjectUrls() { objectUrls.forEach(u => URL.revokeObjectURL(u)); objectUrls = []; }
function renderPrediction(card, result, truth, image) {
  card.querySelector('.pending')?.remove();
  const review = result.needs_review;
  const badge = el('span', review ? 'Perlu tinjauan' : 'Kandidat', review ? 'badge warn' : 'badge ok');
  const h3 = el('h3'); h3.append(document.createTextNode('Dugaan: ' + labels[result.label] + ' '), badge);
  card.append(h3, el('p', 'Probabilitas model: ' + pct(result.confidence) + ' (bukan kepastian)', 'muted'));
  for (const [key, value] of Object.entries(result.probabilities)) {
    const row = el('div', undefined, 'prob-label'); row.append(el('span', labels[key]), el('span', pct(value)));
    const bar = el('progress'); bar.max = 1; bar.value = value; bar.setAttribute('aria-label', 'Probabilitas ' + labels[key] + ' ' + pct(value));
    card.append(row, bar);
  }
  card.append(el('p', result.notice, 'review'));
  if (result.stability) card.append(el('p', 'Konsistensi pada variasi ringan: ' + pct(result.stability.agreement), 'muted'));
  if (truth) {
    const ok = truth === result.label;
    card.append(el('p', 'Label pemilik: ' + labels[truth] + (ok ? ' · sesuai dengan dugaan model.' : ' · berbeda dari dugaan model.'), ok ? 'truth ok' : 'truth miss'));
  }
  const reasons = [...(result.review_reasons || [])];
  if (image?.downscaled) reasons.push('Foto ' + image.original.join(' × ') + ' diperkecil di browser sebelum diproses; hasil bisa sedikit berbeda dari aplikasi Python.');
  if (reasons.length) { const list = el('ul', undefined, 'footnote'); reasons.forEach(r => list.append(el('li', r))); card.append(list); }
}
async function analyze(items) {
  if (busy || !modelReady || !items.length) return;
  busy = true; setControls(); predictions = []; clearObjectUrls(); $('result').replaceChildren();
  if (items.length > MAX_FILES) { $('result').append(el('p', 'Hanya ' + MAX_FILES + ' foto pertama yang dianalisis (dipilih ' + items.length + ').', 'review')); items = items.slice(0, MAX_FILES); }
  for (let i = 0; i < items.length; i++) {
    const {blob, name, truth} = items[i];
    const card = el('article', undefined, 'prediction-card');
    const head = el('div', undefined, 'result-head'), meta = el('div');
    const thumb = el('img'); thumb.alt = 'Foto yang dianalisis: ' + name;
    if (!isHeic(blob)) { const url = URL.createObjectURL(blob); objectUrls.push(url); thumb.src = url; } else thumb.hidden = true;
    meta.append(el('small', name)); head.append(thumb, meta); card.append(head);
    const message = el('p', 'Menganalisis foto ' + (i + 1) + ' / ' + items.length + '…', 'pending'); card.append(message);
    $('result').append(card);
    try {
      const image = await decodeImage(blob);
      const result = await engine.predict({rgba: image.rgba, width: image.width, height: image.height},
        (done, total) => { message.textContent = 'Menganalisis foto ' + (i + 1) + ' / ' + items.length + ' · tampilan ' + done + '/' + total + '…'; });
      predictions.push({file: name, downscaled_in_browser: image.downscaled, ...result});
      renderPrediction(card, result, truth, image);
    } catch (error) {
      message.textContent = name + ': ' + error.message; message.className = 'error'; message.setAttribute('role', 'alert');
      predictions.push({file: name, error: error.message});
    }
  }
  busy = false; setControls();
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
  const first = selected.find(f => !isHeic(f));
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
function renderExamples() {
  const examples = siteConfig.examples ? (report?.examples || []) : [];
  $('examples-block').hidden = $('gallery-panel').hidden = !examples.length;
  const box = $('examples'), gallery = $('gallery');
  box.replaceChildren(); gallery.replaceChildren();
  examples.forEach((example, index) => {
    const button = el('button', undefined, 'example-button secondary'); button.type = 'button';
    button.setAttribute('aria-label', 'Analisis foto contoh ' + (index + 1));
    const img = el('img'); img.src = example.file; img.alt = ''; img.width = 72; img.height = 72; img.loading = 'lazy';
    button.append(img, el('span', 'Contoh ' + (index + 1)));
    button.onclick = async () => {
      if (busy) return;
      try { const response = await fetch(example.file); if (!response.ok) throw Error();
        const blob = new Blob([await response.blob()], {type: 'image/jpeg'});
        selectFiles([]); analyze([{blob, name: 'Contoh ' + (index + 1) + ' (' + example.source + ')', truth: example.label}]); }
      catch { $('result').replaceChildren(el('p', 'Foto contoh tidak dapat dimuat.', 'error')); }
    };
    box.append(button);
    const card = el('article', undefined, 'photo'), photo = el('img');
    photo.src = example.file; photo.loading = 'lazy'; photo.alt = 'Foto test terkunci, label pemilik: ' + labels[example.label];
    card.append(photo, el('small', example.source), el('p', 'Label: ' + labels[example.label] + ' · ' + example.split, 'muted'));
    gallery.append(card);
  });
}
function renderDataset() {
  const d = report.dataset;
  stats('data-stats', [['FOTO BERLABEL', d.images, 'Dipakai training & evaluasi'], ['BELUM BERLABEL', d.unlabeled, 'Tidak dipakai'],
    ['SESI PEMOTRETAN', d.sessions, 'Satu sesi = satu grup'], ['TEST TERKUNCI', Object.values(d.roles.test).reduce((a, b) => a + b, 0) + ' foto', report.protocol.locked_test_sessions.length + ' sesi, dinilai sekali']]);
  $('data-state').textContent = 'Dataset kecil · ' + d.sessions + ' sesi';
  $('audit-text').textContent = 'Versi lama (own_v1) mengelompokkan foto per “set”. Audit menunjukkan setiap set berisi dua sesi berbeda, dan ' +
    '30 foto meja kayu (kemungkinan satu buah yang sama) tersebar di train, validasi, dan test, sehingga akurasi 100% versi lama tidak bisa dipercaya. ' +
    'Tidak ditemukan foto hampir kembar (jarak dHash/pHash minimum 11/12 bit). Versi ini memakai sesi pemotretan sebagai grup: sesi meja kayu selalu di training, ' +
    'validasi dan test hanya dari sesi berlatar lain.';
  table('split-table', ['Peran', ...classKeys.map(k => labels[k]), 'Foto'], [['development (train + validasi silang)', 'dev'], ['test terkunci', 'test']].map(([name, key]) => {
    const c = d.roles[key]; return [name, ...classKeys.map(k => c[k] || 0), classKeys.reduce((s, k) => s + (c[k] || 0), 0)];
  }));
}
function renderTraining() {
  const p = report.protocol, s = report.selection;
  const items = ['Grup: ' + p.grouping + '.', 'Test terkunci: sesi ' + p.locked_test_sessions.join(', ') + ' (dipilih acak dengan seed 42 sebelum training).',
    'Validasi silang: 5 fold atas sesi development lain; seed ' + p.seeds.join(', ') + '.', 'Aturan seleksi: ' + p.selection_rule,
    'Keterbatasan: ' + p.known_bias, ...p.amendments.map(a => 'Amandemen protokol (' + a.date.slice(0, 10) + '): ' + a.change + '. Alasan: ' + a.reason)];
  $('protocol').replaceChildren(...items.map(t => el('li', t)));
  const rows = report.cv.filter(r => r.complete);
  table('ablation', ['Konfigurasi', 'Param', 'Akurasi', 'Wilson 95%', 'Bootstrap sesi 95%', 'Macro-F1', 'Log loss', 'ECE', 'Run dgn val loss > 3', 'Best epoch (median)'],
    rows.map(r => [r.name, idNumber(r.parameters), meanStd(r.accuracy), interval(r.accuracy_wilson95_mean_seed), interval(r.accuracy_session_bootstrap95),
      meanStd(r.macro_f1, false), meanStd(r.loss, false), meanStd(r.ece, false), r.stability.runs_with_spike + '/' + r.stability.runs, r.stability.best_epoch_median]),
    i => rows[i].name === s.selected);
  $('selection-note').textContent = 'Terpilih: ' + s.selected + ' (dibekukan ' + s.frozen.replace('T', ' ') + ', sebelum test diakses). ' +
    (s.tied_within_one_std.length > 1 ? 'Seri dalam 1 simpangan baku: ' + s.tied_within_one_std.join(', ') + '.' : '');
  const box = $('curves'); box.replaceChildren();
  models.models.forEach(m => {
    const figure = el('figure'), img = el('img'); img.src = 'img/learning_curve_' + m.id + '.png'; img.loading = 'lazy';
    img.alt = 'Kurva akurasi dan loss training model final ' + m.id; img.className = 'plot'; figure.append(img, el('figcaption', m.id + ' (seed 42)')); box.append(figure);
  });
}
function renderEvaluation() {
  const s = report.selection.selected, cv = report.cv.find(r => r.name === s), test = report.test.models[s];
  $('evaluation-notice').textContent = 'Test terkunci hanya ' + test.n_images + ' foto dari ' + report.test.sessions.length + ' sesi dan dinilai satu kali setelah model dipilih. ' +
    'Interval kepercayaan sangat lebar; angka ini indikasi awal, bukan bukti generalisasi.';
  stats('eval-stats', [['AKURASI VALIDASI SILANG', pct(cv.accuracy.mean), '± ' + (cv.accuracy.std * 100).toFixed(1).replace('.', ',') + ' antar seed · 95%: ' + interval(cv.accuracy_wilson95_mean_seed)],
    ['MACRO-F1 VALIDASI SILANG', num(cv.macro_f1.mean), '± ' + num(cv.macro_f1.std)],
    ['AKURASI TEST TERKUNCI', pct(test.accuracy.mean), '± ' + (test.accuracy.std * 100).toFixed(1).replace('.', ',') + ' · 95%: ' + interval(test.accuracy_wilson95_mean_seed)],
    ['FOTO TEST', test.n_images, '5 seed model final']]);
  table('model-compare', ['Model', 'Peran', 'Akurasi CV', 'Wilson 95% (CV)', 'Akurasi test', 'Wilson 95% (test)', 'Macro-F1 test'],
    Object.entries(report.test.models).map(([name, t]) => { const c = report.cv.find(r => r.name === name);
      return [name, t.role === 'selected' ? 'terpilih' : 'pembanding', meanStd(c.accuracy), interval(c.accuracy_wilson95_mean_seed), meanStd(t.accuracy), interval(t.accuracy_wilson95_mean_seed), meanStd(t.macro_f1, false)]; }));
  const names = Object.values(labels);
  table('confusion', ['Label / Prediksi', ...names], cv.pooled_over_seeds.confusion_matrix.map((row, i) => [names[i], ...row]));
  table('confusion-test', ['Label / Prediksi', ...names], test.confusion_matrix_pooled.map((row, i) => [names[i], ...row]));
  table('class-metrics', ['Kelas', 'Precision', 'Recall', 'F1', 'Prediksi'], classKeys.map(k => { const m = cv.pooled_over_seeds.report[k]; return [labels[k], num(m.precision), num(m.recall), num(m['f1-score']), m.support]; }));
  const scen = {darker: 'Lebih gelap (×0,8)', brighter: 'Lebih terang (×1,2)', blur: 'Gaussian blur (radius 1)'};
  table('robustness', ['Kondisi', 'Akurasi', 'Macro-F1', 'Loss'], [['Asli', meanStd(test.accuracy), meanStd(test.macro_f1, false), meanStd(test.loss, false)],
    ...Object.entries(test.robustness).map(([k, m]) => [scen[k] || k, meanStd(m.accuracy), meanStd(m.macro_f1, false), meanStd(m.loss, false)])]);
  const legacy = report.legacy_own_v1, ref = report.cv.find(r => r.name === 'ref');
  if (legacy) {
    table('legacy', ['Versi', 'Evaluasi', 'Foto', 'Seed', 'Akurasi'], [
      ['own_v1', 'validasi (1 split)', legacy.validation_images, 1, pct(legacy.validation_accuracy)],
      ['own_v1', 'test (1 split)', legacy.test_images, 1, pct(legacy.test_accuracy)],
      ...(ref ? [['konfigurasi own_v1 di protokol baru', 'validasi silang (ref)', ref.oof_images_per_seed, 5, meanStd(ref.accuracy)]] : []),
      ['own_v2 (' + s + ')', 'validasi silang', cv.oof_images_per_seed, 5, meanStd(cv.accuracy)],
      ['own_v2 (' + s + ')', 'test terkunci', test.n_images, 5, meanStd(test.accuracy)]]);
    $('legacy-note').textContent = legacy.note;
  }
}
$('download-report').onclick = () => report && download(report, 'tomato-vision-report.json');

async function init() {
  setPanel(location.hash.slice(1));
  engine = createEngine();
  try {
    [models, report, siteConfig, modelInfo] = await Promise.all([getJSON('data/models.json'), getJSON('data/report.json', true), getJSON('data/site-config.json', true),
      getJSON('model_info.json', true).catch(() => null)]);
    siteConfig = siteConfig || {examples: false};
  } catch (error) { $('status').textContent = 'Data demo tidak dapat dimuat: ' + error.message; return; }
  const select = $('model-select');
  select.replaceChildren(...models.models.map(m => { const o = el('option', m.id + ' · ' + m.architecture + (m.role === 'selected' ? ' (terpilih)' : '')); o.value = m.id; return o; }));
  select.value = models.default;
  renderAbout(models.default);
  if (report) {
    try { renderExamples(); renderDataset(); renderTraining(); renderEvaluation(); }
    catch (error) { console.error(error); $('evaluation-notice').textContent = 'Sebagian data evaluasi tidak dapat ditampilkan.'; }
  } else { $('evaluation-notice').textContent = 'Data evaluasi tidak tersedia.'; renderExamples(); }
  await loadModel(models.default);
}
init();
