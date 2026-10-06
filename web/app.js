'use strict';
const $ = id => document.getElementById(id);
const labels = {segar:'Segar',tidak_segar:'Tidak segar',busuk:'Busuk'};
let csrf='', rows=[], revision='', dirty=false, dashboard=null, predictions=[], previewURL='';
let modelLoaded=false, jobTimer=null, importBusy=false;
const el=(tag,text,css)=>{const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(css)node.className=css;return node;};
const percent=value=>(Number(value)*100).toFixed(1)+'%';
async function request(path,body){
  const response=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-CSRF-Token':csrf},body:JSON.stringify(body)});
  const data=await response.json();
  if(!response.ok)throw Error(data.error||'Permintaan gagal.');
  return data;
}
function download(value,name){const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));const a=el('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function stats(target,items){$(target).replaceChildren(...items.map(([name,value,note])=>{const card=el('div',undefined,'stat');card.append(el('span',name),el('strong',String(value)));if(note)card.append(el('small',note));return card;}));}
function table(target,headers,values){const t=el('table'),head=el('thead'),tr=el('tr'),body=el('tbody');headers.forEach(x=>tr.append(el('th',x)));head.append(tr);values.forEach(values=>{const row=el('tr');values.forEach(x=>row.append(el('td',String(x))));body.append(row);});t.append(head,body);$(target).replaceChildren(t);}
function setPanel(id){if(!$(id)?.classList.contains('tab-panel'))id='prediction';document.querySelectorAll('.tab-panel').forEach(n=>n.hidden=n.id!==id);document.querySelectorAll('.tab').forEach(n=>{const active=n.dataset.panel===id;n.classList.toggle('active',active);n.setAttribute('aria-pressed',String(active));});}
document.querySelectorAll('.tab').forEach(button=>button.onclick=()=>{location.hash=button.dataset.panel;setPanel(button.dataset.panel);});
window.addEventListener('hashchange',()=>setPanel(location.hash.slice(1)));
window.addEventListener('beforeunload',event=>{if(dirty){event.preventDefault();event.returnValue='';}});

async function loadAnnotations(){const data=await request('/api/annotations');revision=data.revision;rows=data.rows;rows.forEach(row=>row.approved=['true','1'].includes(String(row.approved).toLowerCase()));dirty=false;renderPhotos();}
function changed(){dirty=true;$('saved').textContent='Ada perubahan belum disimpan.';}
function renderPhotos(){
  const query=$('photo-search').value.trim().toLowerCase(),filter=$('photo-filter').value;
  const visible=rows.filter(row=>(filter==='all'||(filter==='approved'?row.approved:!row.approved))&&(row.filepath+' '+row.group_id).toLowerCase().includes(query));
  $('photo-count').textContent=visible.length+' dari '+rows.length+' foto ditampilkan.';
  $('photos').replaceChildren();
  for(const row of visible){
    const card=el('article',undefined,'photo'),img=el('img');img.src='/photo/'+row.sha256;img.loading='lazy';img.alt='Foto sendiri: '+row.filepath.split('/').pop();card.append(img,el('small',row.filepath.split('/').pop()));
    for(const field of ['group_id','label','notes']){
      const label=el('label',{group_id:'ID buah fisik',label:'Kelas kesegaran',notes:'Catatan pengamatan'}[field]);
      const input=el(field==='label'?'select':'input');
      if(field==='label')for(const [value,name] of Object.entries({'':'Belum diberi label',...labels})){const option=el('option',name);option.value=value;input.append(option);}
      input.value=row[field];input.maxLength=500;
      input.oninput=()=>{row[field]=input.value;changed();};label.append(input);card.append(label);
    }
    const label=el('label',undefined,'check'),check=el('input');check.type='checkbox';check.checked=row.approved;check.onchange=()=>{row.approved=check.checked;changed();};label.append(check,document.createTextNode('Label dan ID buah terkonfirmasi'));card.append(label);$('photos').append(card);
  }
}
$('photo-search').oninput=renderPhotos;$('photo-filter').onchange=renderPhotos;
$('save').onclick=async()=>{if(importBusy)return;$('save').disabled=true;try{const result=await request('/api/labels',{rows,revision});revision=result.revision;dirty=false;$('saved').textContent=result.saved+' anotasi tersimpan; backup dibuat.';await refreshDashboard();}catch(error){$('saved').textContent=error.message;}finally{$('save').disabled=false;}};

function renderEvaluation(model){
  for(const target of ['eval-stats','confusion','class-metrics','robustness','model-details'])$(target).replaceChildren();
  $('learning-curve').hidden=true;$('learning-curve').removeAttribute('src');
  const result=model?.evaluation;
  $('evaluation-notice').textContent=model?.mode==='own'?'Hasil held-out foto sendiri. Periksa jumlah buah, distribusi kelas, dan batas generalisasi.':'Hasil evaluasi. Periksa jumlah data uji dan batas generalisasi.';
  if(!model){$('evaluation-notice').textContent='Belum ada model terpilih.';return;}
  const fields=[['Arsitektur',model.config.architecture+' CNN'],['Inisialisasi','Bobot acak; dari nol'],['Parameter',model.model_parameters.toLocaleString('id-ID')],['Epoch terbaik',model.best_epoch+' / '+model.epochs_run],['Ukuran input',model.config.image_size+' × '+model.config.image_size+' RGB'],['Validation accuracy',percent(model.validation.accuracy)],['Train bersih',percent(model.train_clean.accuracy)],['Direktori model',model.run],['Status','Belum tervalidasi untuk produksi']];
  $('model-details').replaceChildren(...fields.flatMap(([key,value])=>[el('dt',key),el('dd',value)]));
  if(!result){$('evaluation-notice').textContent+=' Evaluasi test belum tersedia.';return;}
  stats('eval-stats',[['TEST ACCURACY',percent(result.accuracy),result.n_images+' gambar test'],['MACRO F1',result.macro_f1.toFixed(3),'Rata-rata setara tiga kelas'],['TEST LOSS',result.loss.toFixed(3),'Cross-entropy setelah kalibrasi'],['KELOMPOK TEST',result.uncertainty.group_count,'Kelompok sedikit → estimasi tidak stabil']]);
  table('confusion',['Label / Prediksi',...Object.values(labels)],result.confusion_matrix.map((row,i)=>[Object.values(labels)[i],...row]));
  table('class-metrics',['Kelas','Precision','Recall','F1','Jumlah'],Object.entries(labels).map(([key,name])=>{const m=result.report[key];return [name,m.precision.toFixed(3),m.recall.toFixed(3),m['f1-score'].toFixed(3),m.support];}));
  table('robustness',['Kondisi','Accuracy','Macro F1','Loss'],Object.entries(result.robustness||{}).map(([name,m])=>[{darker:'Lebih gelap (×0,8)',brighter:'Lebih terang (×1,2)',blur:'Gaussian blur'}[name]||name,percent(m.accuracy),m.macro_f1.toFixed(3),m.loss.toFixed(3)]));
  $('learning-curve').src='/artifact/learning_curve.png?run='+encodeURIComponent(model.run);$('learning-curve').hidden=false;
}
function renderJob(job){
  const names={idle:'Belum berjalan',running:'Training berjalan',cancelling:'Menghentikan…',cancelled:'Dihentikan',succeeded:'Selesai',failed:'Gagal — periksa log'};
  $('job-state').textContent=names[job.state]||job.state;
  $('job-log').textContent=job.tail||'Belum ada training dari sesi aplikasi ini.';
  const active=['running','cancelling'].includes(job.state);
  $('train').disabled=!dashboard?.dataset.can_train||active;
  $('cancel-training').disabled=job.state!=='running';
  if(active&&!jobTimer)jobTimer=setInterval(pollJob,2500);
  if(!active&&jobTimer){clearInterval(jobTimer);jobTimer=null;if(job.state==='succeeded')$('train-message').textContent='Training selesai. Klik “Muat model terpilih terbaru” untuk memakainya.';}
}
async function pollJob(){try{renderJob(await request('/api/job'));}catch(error){$('train-message').textContent=error.message;}}
async function refreshDashboard(){
  dashboard=await request('/api/dashboard');const data=dashboard.dataset;
  stats('data-stats',[['FOTO SENDIRI',data.own_images,'Foto unik dalam inventaris'],['TERKONFIRMASI',data.approved,'Label & ID buah disetujui'],['PERLU REVIEW',data.unreviewed,'Belum masuk training nyata']]);
  $('data-state').textContent=data.can_train?'Lolos pemeriksaan teknis':'Perlu melengkapi data';
  $('readiness').replaceChildren();
  const title=el('p',data.can_train?'Dataset lolos pemeriksaan teknis. Tetap tinjau kualitas dan keragaman buah.':'Training belum dapat dimulai.','review');$('readiness').append(title);
  for(const messages of [data.blockers,data.warnings]){const list=el('ul');messages.forEach(message=>list.append(el('li',message)));$('readiness').append(list);}
  const counts=el('div');counts.id='readiness-counts';$('readiness').append(counts);
  table('readiness-counts',['Kelas','Foto disetujui','ID buah'],Object.entries(labels).map(([key,name])=>[name,data.class_counts[key],data.groups_per_class[key]]));
  renderEvaluation(dashboard.model);renderJob(dashboard.job);
  modelLoaded=!!dashboard.model;
  $('status').textContent=modelLoaded?(dashboard.model.mode==='own'?'Model foto sendiri dimuat. Periksa hasil evaluasi sebelum menggunakan prediksi.':'Model dimuat. Periksa hasil evaluasi sebelum menggunakan prediksi.'):'Belum ada model. Anda dapat mengimpor foto dan melengkapi label.';
  $('predict').disabled=!modelLoaded||!$('upload').files.length;
}
$('refresh').onclick=async()=>{try{await refreshDashboard();}catch(error){$('train-message').textContent=error.message;}};
$('download-report').onclick=async()=>{try{download(await request('/api/report'),'tomato-vision-report.json');}catch(error){$('evaluation-notice').textContent=error.message;}};
$('train').onclick=async()=>{
  if(dirty){$('train-message').textContent='Simpan perubahan label terlebih dahulu.';return;}
  if(!confirm('Mulai tiga eksperimen CNN dari nol? Proses memakai CPU dan menyimpan model baru.'))return;
  $('train').disabled=true;
  try{const job=await request('/api/train',{epochs:Number($('epochs').value)});$('train-message').textContent='Eksperimen dimulai. Pantau log di bawah; jangan tutup server terminal.';renderJob(job);}
  catch(error){$('train-message').textContent=error.message;$('train').disabled=!dashboard?.dataset.can_train;}
};
$('reload-model').onclick=async()=>{try{await request('/api/reload-model',{});await refreshDashboard();$('train-message').textContent='Model terpilih terbaru sudah dimuat.';}catch(error){$('train-message').textContent=error.message;}};
$('cancel-training').onclick=async()=>{if(!confirm('Hentikan training? File sementara dipertahankan, model terpilih tidak berubah.'))return;try{renderJob(await request('/api/cancel-training',{}));}catch(error){$('train-message').textContent=error.message;}};

async function fileData(file){if(file.size>10_000_000)throw Error('Ukuran maksimal 10 MB per foto.');return new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(String(reader.result).split(',')[1]);reader.onerror=()=>reject(Error('File tidak dapat dibaca.'));reader.readAsDataURL(file);});}
$('upload').onchange=()=>{const files=$('upload').files;predictions=[];$('download-predictions').disabled=true;$('predict').disabled=!files.length||!modelLoaded;if(previewURL)URL.revokeObjectURL(previewURL);$('preview-frame').hidden=!files.length;if(files.length){previewURL=URL.createObjectURL(files[0]);$('preview').src=previewURL;$('file-info').textContent=files.length+' foto dipilih · pratinjau: '+files[0].name;}else $('file-info').textContent='Belum ada foto dipilih.';};
$('predict').onclick=async()=>{
  const files=Array.from($('upload').files);if(!files.length)return;
  $('predict').disabled=true;$('upload').disabled=true;predictions=[];$('result').replaceChildren();
  for(let i=0;i<files.length;i++){
    const file=files[i],card=el('article',undefined,'prediction-card');card.append(el('small',file.name));const message=el('p','Menganalisis foto '+(i+1)+' / '+files.length+'…');card.append(message);$('result').append(card);
    try{const result=await request('/api/predict',{image:await fileData(file)});predictions.push({file:file.name,...result});message.remove();card.append(el('h3','Dugaan: '+labels[result.label]),el('p','Probabilitas model: '+percent(result.confidence)+' (bukan kepastian)','muted'));
      for(const [key,value]of Object.entries(result.probabilities)){const label=el('div',undefined,'prob-label');label.append(el('span',labels[key]),el('span',percent(value)));const bar=el('progress');bar.max=1;bar.value=value;bar.setAttribute('aria-label',labels[key]);card.append(label,bar);}
      card.append(el('p',result.notice,'review'));
      if(result.stability)card.append(el('p','Konsistensi pada variasi ringan: '+percent(result.stability.agreement),'muted'));
      if(result.review_reasons?.length){const list=el('ul',undefined,'footnote');result.review_reasons.forEach(reason=>list.append(el('li',reason)));card.append(list);}
    }catch(error){message.textContent=error.message;message.className='error';predictions.push({file:file.name,error:error.message});}
  }
  $('predict').disabled=false;$('upload').disabled=false;$('download-predictions').disabled=!predictions.length;
};
$('download-predictions').onclick=()=>download(predictions,'tomato-predictions.json');
$('import').onclick=async()=>{
  if(dirty){$('import-message').textContent='Simpan label yang sedang diedit sebelum impor.';return;}
  if(!$('own-attestation').checked){$('import-message').textContent='Konfirmasi bahwa foto merupakan hasil pemotretan Anda sendiri.';return;}
  const files=Array.from($('import-files').files);if(!files.length){$('import-message').textContent='Pilih foto terlebih dahulu.';return;}
  importBusy=true;$('import').disabled=true;$('save').disabled=true;let added=0,duplicates=0;const errors=[];
  try{for(let i=0;i<files.length;i++){const file=files[i];$('import-message').textContent='Mengimpor '+(i+1)+' / '+files.length+'…';try{const result=await request('/api/import',{name:file.name,image:await fileData(file),own_photo:true});if(result.duplicate)duplicates++;else added++;}catch(error){errors.push(file.name+': '+error.message);}}
    await loadAnnotations();await refreshDashboard();$('import-message').textContent=added+' foto ditambahkan, '+duplicates+' duplikat dilewati.'+(errors.length?' Gagal: '+errors.join('; '):'');
  }catch(error){$('import-message').textContent=error.message;}finally{importBusy=false;$('import').disabled=false;$('save').disabled=false;}
};
async function init(){setPanel(location.hash.slice(1));try{const status=await request('/api/status');csrf=status.csrf;await loadAnnotations();await refreshDashboard();}catch(error){$('status').textContent='Tidak dapat memuat aplikasi: '+error.message;}}
init();
