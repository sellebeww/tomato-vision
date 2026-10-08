'use strict';
// Runs inference off the main thread so the page stays responsive. 'init' may be sent again to switch models.
importScripts('tomato-core.js');
let model = null;

async function sha256Hex(buffer) {
  if (!self.crypto || !self.crypto.subtle) return null;
  const digest = await self.crypto.subtle.digest('SHA-256', buffer);
  return Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('');
}

self.onmessage = async event => {
  const message = event.data;
  try {
    if (message.type === 'init') {
      model = null;
      const [specResponse, weightsResponse] = await Promise.all([fetch(message.specUrl), fetch(message.weightsUrl)]);
      if (!specResponse.ok || !weightsResponse.ok) throw new Error('Model tidak dapat diunduh.');
      const spec = await specResponse.json();
      const buffer = await weightsResponse.arrayBuffer();
      const hash = await sha256Hex(buffer);
      if (hash && hash !== spec.weights_sha256) throw new Error('Bobot model rusak (checksum tidak cocok).');
      model = TomatoCore.createModel(spec, buffer);
      self.postMessage({ type: 'ready', id: message.id, spec: { id: spec.id, architecture: spec.architecture } });
    } else if (message.type === 'predict') {
      if (!model) throw new Error('Model belum siap.');
      const result = TomatoCore.predictPixels(model, new Uint8ClampedArray(message.rgba), message.width, message.height, 4,
        (done, total) => self.postMessage({ type: 'progress', id: message.id, done, total }));
      delete result._raw_views;
      self.postMessage({ type: 'result', id: message.id, result });
    }
  } catch (error) {
    self.postMessage({ type: 'error', id: message.id, message: error.message || String(error) });
  }
};
