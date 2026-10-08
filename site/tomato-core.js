/* Tomato Vision: browser inference core.
 * Re-implements src/preprocessing.py (Pillow letterbox), src/model.py (baseline CNN, BatchNorm folded),
 * src/quality.py and the review logic of src/predict.py. Verified against Python by tests/js_parity.mjs. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.TomatoCore = factory();
})(typeof self !== 'undefined' ? self : globalThis, function () {
  'use strict';
  const PRECISION_BITS = 22;           // Pillow 8-bit resampling fixed-point precision
  const MAX_PIXELS = 25000000;         // same limit as load_image()

  // Python's round(): half to even.
  function pyRound(x) {
    const f = Math.floor(x), d = x - f;
    if (d < 0.5) return f;
    if (d > 0.5) return f + 1;
    return f % 2 === 0 ? f : f + 1;
  }

  // ---- Pillow-compatible bilinear resize (ImagingResample, 8 bits per channel) ----
  function coefficients(inSize, outSize) {
    const scale = inSize / outSize;
    const filterScale = Math.max(scale, 1);
    const support = filterScale;       // bilinear support = 1.0
    const ksize = Math.ceil(support) * 2 + 1;
    const bounds = new Int32Array(outSize * 2);
    const kk = new Int32Array(outSize * ksize);
    const w = new Float64Array(ksize);
    const inv = 1 / filterScale;
    for (let xx = 0; xx < outSize; xx++) {
      const center = (xx + 0.5) * scale;
      let xmin = Math.trunc(center - support + 0.5);
      if (xmin < 0) xmin = 0;
      let xmax = Math.trunc(center + support + 0.5);
      if (xmax > inSize) xmax = inSize;
      xmax -= xmin;
      let total = 0;
      for (let x = 0; x < xmax; x++) {
        const d = Math.abs((x + xmin - center + 0.5) * inv);
        w[x] = d < 1 ? 1 - d : 0;
        total += w[x];
      }
      for (let x = 0; x < xmax; x++) {
        const k = total !== 0 ? w[x] / total : w[x];
        kk[xx * ksize + x] = Math.trunc(k * (1 << PRECISION_BITS) + (k < 0 ? -0.5 : 0.5));
      }
      bounds[xx * 2] = xmin;
      bounds[xx * 2 + 1] = xmax;
    }
    return { ksize, bounds, kk };
  }

  const clip8 = v => (v < 0 ? 0 : v > 255 ? 255 : v);

  /** src: Uint8 array, stride `channels` per pixel (3 or 4); returns Uint8Array RGB (outW*outH*3). */
  function resizeBilinear(src, width, height, channels, outW, outH) {
    const half = 1 << (PRECISION_BITS - 1);
    let cur = src, curW = width, curStride = channels;
    if (outW !== width) {
      const { ksize, bounds, kk } = coefficients(width, outW);
      const out = new Uint8Array(outW * height * 3);
      for (let y = 0; y < height; y++) {
        const row = y * curW * curStride;
        for (let xx = 0; xx < outW; xx++) {
          const xmin = bounds[xx * 2], n = bounds[xx * 2 + 1], base = xx * ksize;
          let s0 = half, s1 = half, s2 = half;
          for (let x = 0; x < n; x++) {
            const k = kk[base + x], p = row + (xmin + x) * curStride;
            s0 += cur[p] * k; s1 += cur[p + 1] * k; s2 += cur[p + 2] * k;
          }
          const o = (y * outW + xx) * 3;
          out[o] = clip8(s0 >> PRECISION_BITS);
          out[o + 1] = clip8(s1 >> PRECISION_BITS);
          out[o + 2] = clip8(s2 >> PRECISION_BITS);
        }
      }
      cur = out; curW = outW; curStride = 3;
    }
    if (outH !== height) {
      const { ksize, bounds, kk } = coefficients(height, outH);
      const out = new Uint8Array(curW * outH * 3);
      for (let yy = 0; yy < outH; yy++) {
        const ymin = bounds[yy * 2], n = bounds[yy * 2 + 1], base = yy * ksize;
        for (let x = 0; x < curW; x++) {
          let s0 = half, s1 = half, s2 = half;
          for (let y = 0; y < n; y++) {
            const k = kk[base + y], p = ((ymin + y) * curW + x) * curStride;
            s0 += cur[p] * k; s1 += cur[p + 1] * k; s2 += cur[p + 2] * k;
          }
          const o = (yy * curW + x) * 3;
          out[o] = clip8(s0 >> PRECISION_BITS);
          out[o + 1] = clip8(s1 >> PRECISION_BITS);
          out[o + 2] = clip8(s2 >> PRECISION_BITS);
        }
      }
      cur = out; curStride = 3;
    }
    if (cur === src || curStride !== 3) {   // size unchanged: copy RGB
      const out = new Uint8Array(outW * outH * 3);
      for (let i = 0, n = outW * outH; i < n; i++) {
        out[i * 3] = cur[i * curStride]; out[i * 3 + 1] = cur[i * curStride + 1]; out[i * 3 + 2] = cur[i * curStride + 2];
      }
      return out;
    }
    return cur;
  }

  /** ImageOps.pad(image, (size,size), BILINEAR, color=(127,127,127), centering=(.5,.5)) -> uint8 RGB. */
  function letterbox(src, width, height, channels, size, pad) {
    if (!(width > 0 && height > 0)) throw new Error('Gambar kosong.');
    if (width * height > MAX_PIXELS) throw new Error('Gambar melebihi 25 megapiksel.');
    let newW = size, newH = size;
    const imRatio = width / height;
    if (imRatio !== 1) {
      if (imRatio > 1) { newH = pyRound(height / width * size); }
      else { newW = pyRound(width / height * size); }
    }
    newW = Math.max(1, newW); newH = Math.max(1, newH);
    const resized = resizeBilinear(src, width, height, channels, newW, newH);
    if (newW === size && newH === size) return resized;
    const out = new Uint8Array(size * size * 3).fill(pad === undefined ? 127 : pad);
    const ox = newW !== size ? pyRound((size - newW) * 0.5) : 0;
    const oy = newW === size ? pyRound((size - newH) * 0.5) : 0;
    for (let y = 0; y < newH; y++) {
      out.set(resized.subarray(y * newW * 3, (y + 1) * newW * 3), ((y + oy) * size + ox) * 3);
    }
    return out;
  }

  // ---- Network: executes the exported graph (src/export_web.py, format tomato-vision-graph/2) ----
  const SUPPORTED_OPS = new Set(['input', 'relu', 'softmax', 'conv', 'sepconv', 'affine', 'groupnorm', 'add', 'maxpool', 'gap', 'dense']);

  function createModel(spec, weightsBuffer) {
    if (spec.format !== 'tomato-vision-graph/2') throw new Error('Format model tidak dikenal.');
    const weights = new Float32Array(weightsBuffer);
    if (weights.length !== spec.weights_floats) throw new Error('Ukuran bobot tidak cocok dengan spesifikasi model.');
    const unknown = spec.nodes.filter(n => !SUPPORTED_OPS.has(n.op)).map(n => n.op);
    if (unknown.length) throw new Error('Operasi model tidak didukung: ' + [...new Set(unknown)].join(', '));
    return { spec, weights, size: spec.input_size, classes: spec.classes };
  }

  // TensorFlow padding rule: "same" puts the extra row/column at the bottom/right.
  function padding(n, k, s, mode) {
    if (mode === 'valid') return [0, Math.floor((n - k) / s) + 1];
    const out = Math.ceil(n / s), total = Math.max((out - 1) * s + k - n, 0);
    return [Math.floor(total / 2), out];
  }

  function conv(t, node, W) {
    const { k, stride: s, cin, cout } = node;
    const [pt, oh] = padding(t.h, k, s, node.padding), [pl, ow] = padding(t.w, k, s, node.padding);
    const out = new Float32Array(oh * ow * cout), acc = new Float64Array(cout), input = t.data;
    for (let oy = 0; oy < oh; oy++) for (let ox = 0; ox < ow; ox++) {
      if (node.b !== null) for (let o = 0; o < cout; o++) acc[o] = W[node.b + o]; else acc.fill(0);
      for (let ky = 0; ky < k; ky++) {
        const iy = oy * s + ky - pt;
        if (iy < 0 || iy >= t.h) continue;
        for (let kx = 0; kx < k; kx++) {
          const ix = ox * s + kx - pl;
          if (ix < 0 || ix >= t.w) continue;
          const ip = (iy * t.w + ix) * cin, wp = node.w + (ky * k + kx) * cin * cout;
          for (let c = 0; c < cin; c++) {
            const v = input[ip + c];
            if (v === 0) continue;
            const wo = wp + c * cout;
            for (let o = 0; o < cout; o++) acc[o] += v * W[wo + o];
          }
        }
      }
      const op = (oy * ow + ox) * cout;
      for (let o = 0; o < cout; o++) out[op + o] = node.act === 'relu' && acc[o] < 0 ? 0 : acc[o];
    }
    return { data: out, h: oh, w: ow, c: cout };
  }

  function sepconv(t, node, W) {
    const { k, stride: s, cin, cout } = node;
    const [pt, oh] = padding(t.h, k, s, node.padding), [pl, ow] = padding(t.w, k, s, node.padding);
    const out = new Float32Array(oh * ow * cout), depth = new Float64Array(cin), acc = new Float64Array(cout), input = t.data;
    for (let oy = 0; oy < oh; oy++) for (let ox = 0; ox < ow; ox++) {
      depth.fill(0);
      for (let ky = 0; ky < k; ky++) {
        const iy = oy * s + ky - pt;
        if (iy < 0 || iy >= t.h) continue;
        for (let kx = 0; kx < k; kx++) {
          const ix = ox * s + kx - pl;
          if (ix < 0 || ix >= t.w) continue;
          const ip = (iy * t.w + ix) * cin, dp = node.dw + (ky * k + kx) * cin;
          for (let c = 0; c < cin; c++) depth[c] += input[ip + c] * W[dp + c];
        }
      }
      if (node.b !== null) for (let o = 0; o < cout; o++) acc[o] = W[node.b + o]; else acc.fill(0);
      for (let c = 0; c < cin; c++) {
        const v = depth[c], wo = node.w + c * cout;
        for (let o = 0; o < cout; o++) acc[o] += v * W[wo + o];
      }
      const op = (oy * ow + ox) * cout;
      for (let o = 0; o < cout; o++) out[op + o] = node.act === 'relu' && acc[o] < 0 ? 0 : acc[o];
    }
    return { data: out, h: oh, w: ow, c: cout };
  }

  function groupNorm(t, node, W) {
    const { c } = t, g = node.groups, cg = c / g, n = t.h * t.w, out = new Float32Array(t.data.length);
    for (let group = 0; group < g; group++) {
      let sum = 0, sq = 0;
      for (let i = 0; i < n; i++) for (let j = 0; j < cg; j++) sum += t.data[i * c + group * cg + j];
      const mean = sum / (n * cg);
      for (let i = 0; i < n; i++) for (let j = 0; j < cg; j++) { const d = t.data[i * c + group * cg + j] - mean; sq += d * d; }
      const inv = 1 / Math.sqrt(sq / (n * cg) + node.epsilon);
      for (let i = 0; i < n; i++) for (let j = 0; j < cg; j++) {
        const ch = group * cg + j, idx = i * c + ch;
        out[idx] = (t.data[idx] - mean) * inv * W[node.gamma + ch] + W[node.beta + ch];
      }
    }
    return { ...t, data: out };
  }

  function maxPool(t, node) {
    const { k, stride: s } = node, oh = Math.floor((t.h - k) / s) + 1, ow = Math.floor((t.w - k) / s) + 1, c = t.c;
    const out = new Float32Array(oh * ow * c).fill(-Infinity);
    for (let y = 0; y < oh; y++) for (let x = 0; x < ow; x++) for (let ky = 0; ky < k; ky++) for (let kx = 0; kx < k; kx++) {
      const ip = ((y * s + ky) * t.w + (x * s + kx)) * c, op = (y * ow + x) * c;
      for (let ch = 0; ch < c; ch++) if (t.data[ip + ch] > out[op + ch]) out[op + ch] = t.data[ip + ch];
    }
    return { data: out, h: oh, w: ow, c };
  }

  function softmaxInPlace(v) {
    let max = -Infinity;
    for (const x of v) if (x > max) max = x;
    let sum = 0;
    for (let j = 0; j < v.length; j++) { v[j] = Math.exp(v[j] - max); sum += v[j]; }
    for (let j = 0; j < v.length; j++) v[j] /= sum;
    return v;
  }

  /** x: Float32Array (size*size*3) in [0,1]. Returns Float32Array of class probabilities (raw softmax). */
  function forward(model, x) {
    const { spec, weights: W } = model, values = new Map();
    for (const node of spec.nodes) {
      const a = node.inputs && node.inputs.length ? values.get(node.inputs[0]) : null;
      let y;
      switch (node.op) {
        case 'input': y = { data: x, h: spec.input_size, w: spec.input_size, c: 3 }; break;
        case 'relu': {
          const d = new Float32Array(a.data.length), max = node.max === null ? Infinity : node.max;
          for (let i = 0; i < d.length; i++) d[i] = Math.min(Math.max(a.data[i], 0), max);
          y = { ...a, data: d }; break;
        }
        case 'softmax': y = { ...a, data: softmaxInPlace(Float32Array.from(a.data)) }; break;
        case 'conv': y = conv(a, node, W); break;
        case 'sepconv': y = sepconv(a, node, W); break;
        case 'groupnorm': y = groupNorm(a, node, W); break;
        case 'affine': {
          const d = new Float32Array(a.data.length);
          for (let i = 0; i < d.length; i++) { const ch = i % a.c; d[i] = a.data[i] * W[node.scale + ch] + W[node.shift + ch]; }
          y = { ...a, data: d }; break;
        }
        case 'add': {
          const d = Float32Array.from(a.data);
          for (const name of node.inputs.slice(1)) { const b = values.get(name).data; for (let i = 0; i < d.length; i++) d[i] += b[i]; }
          y = { ...a, data: d }; break;
        }
        case 'maxpool': y = maxPool(a, node); break;
        case 'gap': {
          const d = new Float32Array(a.c), n = a.h * a.w;
          for (let i = 0; i < n; i++) for (let k = 0; k < a.c; k++) d[k] += a.data[i * a.c + k];
          for (let k = 0; k < a.c; k++) d[k] /= n;
          y = { data: d, h: 1, w: 1, c: a.c }; break;
        }
        case 'dense': {
          const d = new Float32Array(node.out);
          for (let j = 0; j < node.out; j++) d[j] = W[node.b + j];
          for (let i = 0; i < node.in; i++) {
            const v = a.data[i];
            if (v === 0) continue;
            const wo = node.w + i * node.out;
            for (let j = 0; j < node.out; j++) d[j] += v * W[wo + j];
          }
          if (node.act === 'relu') { for (let j = 0; j < d.length; j++) if (d[j] < 0) d[j] = 0; }
          else if (node.act === 'softmax') softmaxInPlace(d);
          y = { data: d, h: 1, w: 1, c: node.out }; break;
        }
        default: throw new Error('Operasi tidak didukung: ' + node.op);
      }
      values.set(node.name, y);
    }
    return values.get(spec.output).data;
  }

  // ---- Diagnostics (src/evaluate.py temperature_scale, src/quality.py) ----
  function temperatureScale(probs, temperature) {
    if (!(isFinite(temperature) && temperature > 0)) throw new Error('Temperature tidak valid.');
    const logits = Array.from(probs, p => Math.log(Math.min(Math.max(p, 1e-8), 1)) / temperature);
    const max = Math.max(...logits);
    const e = logits.map(v => Math.exp(v - max)), sum = e.reduce((a, b) => a + b, 0);
    return e.map(v => v / sum);
  }

  function assessQuality(x, size) {
    const n = size * size, gray = new Float64Array(n);
    let sum = 0;
    for (let i = 0; i < n; i++) {
      gray[i] = x[i * 3] * 0.2126 + x[i * 3 + 1] * 0.7152 + x[i * 3 + 2] * 0.0722;
      sum += gray[i];
    }
    const brightness = sum / n;
    let sq = 0;
    for (let i = 0; i < n; i++) sq += (gray[i] - brightness) ** 2;
    const contrast = Math.sqrt(sq / n);
    const sharpness = contentSharpness(x, gray, size);
    const warnings = [];
    if (brightness < 0.08) warnings.push('Gambar sangat gelap; ambil ulang dengan pencahayaan lebih baik.');
    if (brightness > 0.95) warnings.push('Gambar sangat terang; detail permukaan mungkin hilang.');
    if (contrast < 0.025) warnings.push('Kontras sangat rendah atau gambar hampir polos.');
    if (sharpness < SHARPNESS_MIN) warnings.push('Foto tampak buram atau detail tepi rendah; periksa fokus dan ukuran tomat dalam foto.');
    const tomatoFraction = tomatoColorFraction(x, size);
    if (tomatoFraction < TOMATO_COLOR_MIN) warnings.push('Warna merah/oranye khas tomat hampir tidak terdeteksi; foto mungkin bukan tomat, ' +
      'tomat hijau, atau buah terlalu kecil di foto.');
    return { brightness, contrast, edge_variance: sharpness, tomato_color_fraction: tomatoFraction, warnings,
             method: 'Heuristic on preprocessed image; not a non-tomato detector or validated blur classifier' };
  }

  // Same as src/quality.py content_sharpness: Laplacian variance inside the photo area (padding excluded).
  const SHARPNESS_MIN = 1e-3, PAD_VALUE = Math.fround(127 / 255);
  function contentSharpness(x, gray, size) {
    const isPad = i => x[i * 3] === PAD_VALUE && x[i * 3 + 1] === PAD_VALUE && x[i * 3 + 2] === PAD_VALUE;
    let top = -1, bottom = -1, left = -1, right = -1;
    for (let y = 0; y < size; y++) for (let xx = 0; xx < size; xx++) if (!isPad(y * size + xx)) {
      if (top < 0) top = y; bottom = y;
      if (left < 0 || xx < left) left = xx; if (xx > right) right = xx;
    }
    const h = bottom - top + 1, w = right - left + 1;
    if (top < 0 || h < 3 || w < 3) return 0;
    const n = (h - 2) * (w - 2), lap = new Float64Array(n);
    let k = 0, sum = 0;
    for (let y = top + 1; y < bottom; y++) for (let xx = left + 1; xx < right; xx++) {
      const i = y * size + xx;
      const v = gray[i - 1] + gray[i + 1] + gray[i - size] + gray[i + size] - 4 * gray[i];
      lap[k++] = v; sum += v;
    }
    const mean = sum / n;
    let sq = 0;
    for (let i = 0; i < n; i++) sq += (lap[i] - mean) ** 2;
    return sq / n;
  }

  // Same as src/quality.py tomato_color_fraction (threshold calibrated on own photos; see that file).
  const TOMATO_COLOR_MIN = 0.05;
  function tomatoColorFraction(x, size) {
    let hits = 0;
    const n = size * size;
    for (let i = 0; i < n; i++) {
      const r = x[i * 3], g = x[i * 3 + 1], b = x[i * 3 + 2];
      const high = Math.max(r, g, b), low = Math.min(r, g, b), delta = high - low;
      if (!(delta > 0) || high < 0.2 || delta / high < 0.35) continue;
      let hue;
      if (high === r) hue = ((((g - b) / delta) % 6) + 6) % 6;
      else if (high === g) hue = (b - r) / delta + 2;
      else hue = (r - g) / delta + 4;
      hue *= 60;
      if (hue <= 40 || hue >= 330) hits++;
    }
    return hits / n;
  }

  function reviewDecision(probs, rawProbs, viewPredictions, quality, threshold) {
    const ordered = Array.from(probs).sort((a, b) => a - b);
    let candidate = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[candidate]) candidate = i;
    const margin = ordered[ordered.length - 1] - ordered[ordered.length - 2];
    const agreement = viewPredictions.filter(v => v === candidate).length / viewPredictions.length;
    const reasons = [];
    if (Math.min(probs[candidate], rawProbs[candidate]) < threshold) reasons.push('Probabilitas prediksi belum melewati ambang tinjauan.');
    if (margin < 0.15) reasons.push('Dua kelas teratas memiliki probabilitas berdekatan.');
    if (agreement < 1) reasons.push('Prediksi berubah pada flip atau perubahan pencahayaan ringan.');
    reasons.push(...quality.warnings);
    return { decision: reasons.length ? 'review' : 'candidate', needs_review: reasons.length > 0,
             review_reasons: reasons, probability_margin: margin,
             stability: { agreement, views: viewPredictions.length,
                          method: 'Original, horizontal flip, brightness x0.9 and x1.1; diagnostic only' },
             notice: reasons.length ? 'Hasil masih berupa dugaan; perlu pemeriksaan manual.'
               : 'Dugaan kondisi visual; probabilitas bukan jaminan dan bukan penilaian keamanan pangan.' };
  }

  const argmax = a => { let k = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[k]) k = i; return k; };

  function inputViews(x, size) {
    const flip = new Float32Array(x.length), dark = new Float32Array(x.length), light = new Float32Array(x.length);
    for (let y = 0; y < size; y++) for (let xx = 0; xx < size; xx++) for (let c = 0; c < 3; c++) {
      flip[(y * size + xx) * 3 + c] = x[(y * size + (size - 1 - xx)) * 3 + c];
    }
    for (let i = 0; i < x.length; i++) {
      dark[i] = Math.min(Math.max(x[i] * 0.9, 0), 1);
      light[i] = Math.min(Math.max(x[i] * 1.1, 0), 1);
    }
    return [x, flip, dark, light];
  }

  /** Full prediction from a preprocessed uint8 tensor. onView(i, total) is called after each forward pass. */
  function predictTensor(model, tensor, onView) {
    const { spec } = model, size = spec.input_size;
    const x = new Float32Array(tensor.length);
    for (let i = 0; i < tensor.length; i++) x[i] = tensor[i] / 255;
    const views = inputViews(x, size), raw = [], scaled = [];
    views.forEach((view, i) => {
      const r = forward(model, view);
      raw.push(r); scaled.push(temperatureScale(r, spec.temperature));
      if (onView) onView(i + 1, views.length);
    });
    const probs = scaled[0], idx = argmax(probs);
    const quality = assessQuality(x, size);
    const decision = reviewDecision(probs, raw[0], scaled.map(argmax), quality, spec.confidence_threshold);
    return { label: spec.classes[idx], confidence: probs[idx],
             probabilities: Object.fromEntries(spec.classes.map((c, i) => [c, probs[i]])),
             raw_probabilities: Object.fromEntries(spec.classes.map((c, i) => [c, raw[0][i]])),
             quality, ...decision, calibration: spec.calibration, mode: spec.mode,
             production_ready: spec.production_ready,
             scope: 'Satu tomat terlihat jelas. Tidak mendeteksi objek non-tomat atau menjamin keamanan pangan.',
             _raw_views: raw };
  }

  /** rgba/rgb: decoded pixels at full resolution. */
  function predictPixels(model, pixels, width, height, channels, onView) {
    const tensor = letterbox(pixels, width, height, channels, model.size, model.spec.pad_color);
    return predictTensor(model, tensor, onView);
  }

  return { createModel, forward, letterbox, SUPPORTED_OPS, resizeBilinear, predictTensor, predictPixels,
           assessQuality, reviewDecision, temperatureScale, pyRound };
});
