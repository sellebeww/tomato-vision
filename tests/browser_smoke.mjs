// Run against a separate headless Chrome instance with --remote-debugging-port=9234.
import assert from 'node:assert/strict';
import {writeFile,mkdir} from 'node:fs/promises';
import path from 'node:path';
const pages=await (await fetch('http://127.0.0.1:9234/json/list')).json();
const page=pages.find(p=>p.type==='page');
assert(page,'No browser page');
const initial=await (await fetch('http://127.0.0.1:7860/api/dashboard')).json();
const ws=new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject;});
let counter=0;const pending=new Map(),errors=[];
ws.onmessage=event=>{const message=JSON.parse(event.data);if(message.id){const item=pending.get(message.id);if(item){clearTimeout(item.timer);pending.delete(message.id);message.error?item.reject(Error(JSON.stringify(message.error))):item.resolve(message.result);}}else if(message.method==='Runtime.exceptionThrown')errors.push(message.params.exceptionDetails.text);};
function command(method,params={}){return new Promise((resolve,reject)=>{const id=++counter;const timer=setTimeout(()=>{pending.delete(id);reject(Error('CDP timeout: '+method));},20000);pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}));});}
async function evaluate(expression){const result=await command('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(result.exceptionDetails)throw Error(JSON.stringify(result.exceptionDetails));return result.result.value;}
async function until(expression){for(let i=0;i<100;i++){if(await evaluate(expression))return;await new Promise(r=>setTimeout(r,200));}throw Error('Timed out: '+expression);}
await command('Runtime.enable');await command('Page.enable');
await command('Page.navigate',{url:'http://127.0.0.1:7860'});
await until("document.querySelector('#data-state')?.textContent!=='Memuat…'");
assert.equal(await evaluate("document.querySelectorAll('.photo').length"),initial.dataset.own_images);
assert.equal(await evaluate("getComputedStyle(document.querySelector('#preview-frame')).display"),'none');
await mkdir('outputs/browser',{recursive:true});
async function screenshot(name){const result=await command('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});await writeFile('outputs/browser/'+name+'.png',Buffer.from(result.data,'base64'));}
await screenshot('desktop-prediction');
for(const id of ['dataset','training','evaluation']){
  await evaluate(`document.querySelector('[data-panel="${id}"]').click()`);
  await until(`!document.querySelector('#${id}').hidden`);
  await screenshot('desktop-'+id);
}
assert.equal(await evaluate("document.querySelectorAll('#class-metrics tbody tr').length"),3);
assert.equal(await evaluate("document.querySelector('#train').disabled"),!initial.dataset.can_train||['running','cancelling'].includes(initial.job.state));
await evaluate("document.querySelector('[data-panel=dataset]').click();document.querySelector('#photo-filter').value='approved';document.querySelector('#photo-filter').dispatchEvent(new Event('change'))");
assert.equal(await evaluate("document.querySelectorAll('.photo').length"),initial.dataset.approved);
await evaluate("document.querySelector('#photo-filter').value='all';document.querySelector('#photo-filter').dispatchEvent(new Event('change'));document.querySelector('#import').click()");
assert.match(await evaluate("document.querySelector('#import-message').textContent"),/Konfirmasi/);
await evaluate("document.querySelector('[data-panel=prediction]').click()");
const document=await command('DOM.getDocument');
const input=await command('DOM.querySelector',{nodeId:document.root.nodeId,selector:'#upload'});
await command('DOM.setFileInputFiles',{nodeId:input.nodeId,files:[path.resolve('data/raw/TOM001_segar.png'),path.resolve('data/raw/TOM042_tidak_segar.png')]});
await until("!document.querySelector('#predict').disabled");
await evaluate("document.querySelector('#predict').click()");
await until("!document.querySelector('#download-predictions').disabled");
assert.equal(await evaluate("document.querySelectorAll('.prediction-card').length"),2);
assert.equal(await evaluate("document.querySelectorAll('.prediction-card progress').length"),6);
assert.equal(await evaluate("document.querySelectorAll('.prediction-card .error').length"),0);
assert.match(await evaluate("document.querySelector('.prediction-card h3').textContent"),/Dugaan/);
assert.match(await evaluate("document.querySelector('.prediction-card').textContent"),/konsistensi/i);
await screenshot('desktop-results');
await command('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
for(const id of ['prediction','dataset','training','evaluation']){
  await evaluate(`document.querySelector('[data-panel="${id}"]').click();window.scrollTo(0,0)`);
  await until(`!document.querySelector('#${id}').hidden`);
  assert.equal(await evaluate('document.documentElement.scrollWidth<=window.innerWidth+1'),true,'Mobile overflow: '+id);
  await screenshot('mobile-'+id);
}
assert.deepEqual(errors,[],'Browser JavaScript exceptions');
console.log(JSON.stringify({browser:'passed',checks:['tabs','annotations','filters','import guard','batch predictions','probabilities','training guard','metrics','responsive layout','no JavaScript exceptions'],screenshots:'outputs/browser'},null,2));
ws.close();
