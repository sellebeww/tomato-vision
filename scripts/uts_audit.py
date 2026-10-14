"""Audit UTS dari berkas aktual; tidak melatih model atau mengubah data sumber.
Jalankan: venv/bin/python -m scripts.uts_audit
"""
from pathlib import Path
import os, json, hashlib, itertools

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'submission/uts'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/.matplotlib'))
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.preprocessing import load_image

CLASSES = ['segar', 'tidak_segar', 'busuk']
MANIFEST = ROOT / 'data/prepared/d634459bcd9edb74/manifest.csv'

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def run():
    OUT.mkdir(parents=True, exist_ok=True)
    figdir = OUT / 'figures'; figdir.mkdir(exist_ok=True)
    manifest = pd.read_csv(MANIFEST, keep_default_na=False)
    annotations = pd.read_csv(ROOT / 'data/annotations.csv', keep_default_na=False)
    assert set(manifest.sha256) == set(annotations.sha256), 'Manifest dan anotasi berbeda'
    assert (manifest.source == 'own').all()
    # Jalur aktif mengikuti anotasi (seluruh 229 foto ada di data/raw).
    manifest['historical_filepath'] = manifest.filepath
    manifest['filepath'] = manifest.sha256.map(annotations.set_index('sha256').filepath)
    manifest['approved'] = True
    confirmation=json.loads((OUT/'source_confirmation.json').read_text()) if (OUT/'source_confirmation.json').exists() else {}
    visual_review = bool(confirmation.get('jpeg_22_visual_review_done'))
    inventory = []; metrics = []; tensors = []; cleanrows = []
    for folder in ['data/raw', 'data/reference_import', 'submission/dataset/images', 'site/examples']:
        for p in sorted((ROOT / folder).glob('*')):
            if p.suffix.lower() not in {'.png','.jpg','.jpeg','.webp','.heic'}: continue
            row = {'filepath': str(p.relative_to(ROOT)), 'sha256': digest(p)}
            try:
                with Image.open(p) as im: im.verify()
                with Image.open(p) as im:
                    row.update(format=im.format, mode=im.mode, width=im.width, height=im.height,
                               exif_fields=len(im.getexif()), frames=getattr(im,'n_frames',1), status='ok')
                    rgb = ImageOps.exif_transpose(im).convert('RGB')
                    row['pixel_sha256'] = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            except Exception as exc: row.update(status='error', error=str(exc))
            inventory.append(row)
    inv = pd.DataFrame(inventory); inv.to_csv(OUT / 'inventory_all.csv', index=False)
    # Hitung inventaris akuisisi awal; JPEG di raw kini salinan operasional identik.
    originals = inv[inv.filepath.str.startswith('data/reference_import/') |
                    inv.filepath.str.startswith('data/raw/TOM') | inv.filepath.str.startswith('data/raw/own_')].copy()
    canonical=inv[inv.filepath.str.startswith('data/raw/')]
    if confirmation:
        assert len(canonical)==229 and canonical.sha256.nunique()==229
        assert set(canonical.sha256)==set(manifest.sha256)
    canonical.to_csv(OUT/'inventory_canonical.csv',index=False)
    originals.to_csv(OUT / 'inventory_originals.csv', index=False)
    originals[originals.duplicated('sha256',keep=False)].to_csv(OUT / 'duplicates_bytes.csv',index=False)
    originals[originals.duplicated('pixel_sha256',keep=False)].to_csv(OUT / 'duplicates_pixels.csv',index=False)
    for r in manifest.itertuples():
        p = ROOT / r.filepath
        assert digest(p) == r.sha256, f'Hash berubah: {p}'
        assert r.label in CLASSES
        x = load_image(p)
        assert x.shape == (128,128,3) and x.dtype == np.float32 and 0 <= x.min() <= x.max() <= 1
        tensors.append(x)
        cleanrows.append({'index': len(tensors)-1, 'filepath':r.filepath,'label':r.label,
                          'label_idx':CLASSES.index(r.label),'group_id':r.group_id,'split':r.split,'sha256':r.sha256,
                          'provenance_status':'owner_attestation' if confirmation else 'menunggu_konfirmasi_pemilik',
                          'label_review_status':('visual_review' if visual_review else 'pending')
                          if Path(r.filepath).name.startswith('tomat_ref_') else 'owner_annotation'})
        if r.split != 'train': continue
        with Image.open(p) as im:
            rgb = ImageOps.exif_transpose(im).convert('RGB')
            # Deskriptor memakai keseluruhan gambar tanpa padding, resolusi seragam.
            rgb.thumbnail((256,256), Image.Resampling.BILINEAR)
            a = np.asarray(rgb,dtype=float)/255
            gray = a @ np.array([.299,.587,.114])
            sat = np.asarray(rgb.convert('HSV'),dtype=float)[...,1]/255
            edge = (np.abs(np.diff(gray,axis=0)).mean()+np.abs(np.diff(gray,axis=1)).mean())/2
            g = np.asarray(rgb.convert('L').resize((9,8),Image.Resampling.BILINEAR))
            dh = int(''.join('1' if z else '0' for z in (g[:,1:]>g[:,:-1]).ravel()),2)
            metrics.append(dict(filepath=r.filepath,label=r.label,group_id=r.group_id,brightness=gray.mean(),
                                saturation=sat.mean(),edge=edge,red_mean=a[...,0].mean(),
                                green_mean=a[...,1].mean(),blue_mean=a[...,2].mean(),dhash=dh))
    df = pd.DataFrame(metrics); df.to_csv(OUT / 'training_metrics.csv',index=False)
    cols=['brightness','saturation','edge','red_mean','green_mean','blue_mean']
    df.groupby('label')[cols].agg(['count','mean','std','min','median','max']).to_csv(OUT/'descriptive_by_class.csv')
    df[cols].describe().to_csv(OUT/'descriptive_training.csv')
    clean = pd.DataFrame(cleanrows); clean.to_csv(OUT/'clean_manifest.csv',index=False)
    np.savez_compressed(OUT/'clean_dataset_128.npz', images=np.stack(tensors),
                        labels=clean.label_idx.to_numpy(dtype=np.int32), splits=clean.split.to_numpy(dtype=str),
                        label_origin=clean.label_review_status.to_numpy(dtype=str))
    manifest.to_csv(OUT/'frozen_manifest.csv',index=False)
    issues=[]
    for key in ['group_id','session_id','sha256','pixel_sha256']:
        bad=manifest.groupby(key).split.nunique(); issues.extend([f'{key}:{k}' for k in bad[bad>1].index])
    assert not issues, issues
    manifest_near=[]
    for a,b in itertools.combinations(manifest.itertuples(),2):
        distance=(int(a.dhash,16)^int(b.dhash,16)).bit_count()
        if distance<=4:
            manifest_near.append(dict(a=a.filepath,b=b.filepath,distance=distance,
                                      split_a=a.split,split_b=b.split,cross_split=a.split!=b.split))
    pd.DataFrame(manifest_near,columns=['a','b','distance','split_a','split_b','cross_split']).to_csv(
        OUT/'manifest_near_pairs.csv',index=False)
    pairs=[]
    for a,b in itertools.combinations(df.itertuples(),2):
        distance=(a.dhash ^ b.dhash).bit_count()
        pairs.append(dict(a=a.filepath,b=b.filepath,distance=distance,same_group=a.group_id==b.group_id,
                          same_label=a.label==b.label))
    pd.DataFrame(pairs).to_csv(OUT/'training_dhash_pairs.csv',index=False)
    plt.rcParams.update({'font.size':9,'figure.dpi':140})
    def save(i,title):
        plt.title(title); plt.tight_layout(); plt.savefig(figdir/f'{i:02d}.png'); plt.close()
    counts=manifest.label.value_counts().reindex(CLASSES)
    counts.plot.bar(color=['#399c58','#e3a633','#985244']); plt.ylabel('Foto unik'); plt.xticks(rotation=0)
    save(1,'Distribusi label (229 foto)')
    def bucket(g):
        if g.startswith('own:paket_lama_scene'): return 'scene01-scene49:\n49 grup x 3 foto'
        if g.startswith('own:b'): return 'grup b01-b10:\n10 grup x 3 foto'
        return {'own:a_meja_kayu':'a_meja_kayu\n(1 grup)','own:r_reference_meja_kayu':'r_reference\n(1 grup, 22 JPEG)'}[g]
    order_b=['a_meja_kayu\n(1 grup)','r_reference\n(1 grup, 22 JPEG)','grup b01-b10:\n10 grup x 3 foto','scene01-scene49:\n49 grup x 3 foto']
    pd.crosstab(manifest.group_id.map(bucket),manifest.label).reindex(index=order_b,columns=CLASSES).plot.bar(stacked=True,figsize=(9,4.5))
    plt.ylabel('Foto'); plt.xticks(rotation=0); save(2,'Label menurut kelompok grup (61 grup konservatif; grup bukan jumlah buah)')
    pd.crosstab(manifest.split,manifest.label).reindex(index=['train','val','test'],columns=CLASSES).plot.bar()
    plt.ylabel('Foto'); save(3,'Pembagian historis berbasis grup (bersifat sementara)')
    unique=originals.drop_duplicates('sha256')
    for fmt,part in unique.groupby('format'): plt.scatter(part.width,part.height,label=fmt,alpha=.6)
    plt.xlabel('Lebar (piksel)'); plt.ylabel('Tinggi (piksel)'); plt.legend(); save(4,'Dimensi gambar sumber unik')
    fig,axes=plt.subplots(3,4,figsize=(10,7))
    for i,label in enumerate(CLASSES):
        part=manifest[(manifest.label==label)&(manifest.split=='train')]
        heads=part.groupby('group_id').head(1)
        subset=heads.iloc[::max(1,len(heads)//4)].head(4)
        for ax in axes[i]: ax.axis('off')
        for ax,r in zip(axes[i],subset.itertuples()):
            with Image.open(ROOT/r.filepath) as im: ax.imshow(ImageOps.exif_transpose(im).convert('RGB'))
            ax.set_title(f'{label}\n{Path(r.filepath).name}',fontsize=7)
    fig.suptitle('Contoh training aktual, satu gambar per grup dan kelas'); fig.tight_layout(); fig.savefig(figdir/'05.png'); plt.close(fig)
    for i,col,title in [(6,'brightness','Luminans rata-rata'),(7,'saturation','Saturasi rata-rata'),(8,'edge','Rata-rata gradien absolut (bukan diagnosis blur)')]:
        plt.boxplot([df[df.label==c][col] for c in CLASSES],tick_labels=CLASSES)
        for j,c in enumerate(CLASSES,1):
            vals=df[df.label==c][col].to_numpy(); plt.scatter(j+np.linspace(-.12,.12,len(vals)),vals,s=9,alpha=.5)
        plt.ylabel(col); save(i,title+' — hanya training')
    corr=df[cols].corr(); plt.figure(figsize=(7,5)); plt.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm'); plt.colorbar()
    plt.xticks(range(len(cols)),cols,rotation=35,ha='right'); plt.yticks(range(len(cols)),cols)
    for i in range(len(cols)):
        for j in range(len(cols)): plt.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8)
    save(9,'Korelasi Pearson deskriptor gambar training')
    pd.Series([p['distance'] for p in pairs]).plot.hist(bins=range(0,66,2)); plt.axvline(4,color='red',ls='--',label='Ambang tinjauan ≤4')
    plt.xlabel('Jarak dHash 64 bit'); plt.ylabel('Pasangan training'); plt.legend(); save(10,'Kemiripan visual seluruh pasangan training')
    summary=dict(source_files=len(originals),unique_source_bytes=originals.sha256.nunique(),
                 duplicate_excess=len(originals)-originals.sha256.nunique(), corrupt=int((inv.status!='ok').sum()),
                 inventory_files=len(inv),label_counts=counts.to_dict(),split_counts=manifest.split.value_counts().to_dict(),
                 groups=int(manifest.group_id.nunique()),primary_verified=bool(confirmation),
                 provenance_basis='owner_attestation' if confirmation else 'unconfirmed',
                 canonical_unique_images=int(canonical.sha256.nunique()),pending_label_images=0 if visual_review else 22,
                 visually_reviewed_images=22 if visual_review else 0,physical_ground_truth_verified=False,
                 crossing_metadata_keys=issues,near_train_pairs=sum(p['distance']<=4 for p in pairs),
                 manifest_near_pairs=len(manifest_near),manifest_near_cross_split=sum(p['cross_split'] for p in manifest_near),
                 dimensions=unique.groupby(['format','width','height']).size().reset_index(name='n').to_dict('records'),
                 training_medians=df.groupby('label')[cols].median().to_dict(),
                 selected_correlation={'pair': ['brightness','red_mean'],'r':float(corr.loc['brightness','red_mean'])},
                 runtime={'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__},
                 manifest_sha256=digest(MANIFEST),test_policy='Hanya audit teknis; deskriptor dan contoh EDA menggunakan training; tidak menjalankan model.')
    (OUT/'audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary

if __name__ == '__main__': run()
