"""Local dataset readiness, safe photo intake, and versioned annotation saves."""
import hashlib
import io
from pathlib import Path
import secrets
import shutil
import pandas as pd
from PIL import Image
from src.config import CLASS_NAMES, DATA_DIR, RAW_DATA_DIR
from src.manifest import (LABEL_COLUMNS, init_labels, validate_annotation_rows, enrich,
                          group_near_duplicates, apply_session_groups, split_with_existing_holdout)
from src.dataset_split import assign_group_splits, validate_splits

def annotation_snapshot():
    path=DATA_DIR/'annotations.csv'
    raw=path.read_bytes()
    return {'revision':hashlib.sha256(raw).hexdigest(),
            'rows':pd.read_csv(io.BytesIO(raw),keep_default_na=False).to_dict('records')}

def save_annotations(rows, revision):
    current=annotation_snapshot()
    if not isinstance(revision,str) or revision!=current['revision']:
        raise ValueError('Label sudah berubah di tab lain. Muat ulang sebelum menyimpan.')
    if not isinstance(rows,list) or len(rows)!=len(current['rows']):
        raise ValueError('Inventaris berubah; muat ulang halaman.')
    edits={r['sha256']:r for r in rows}
    if len(edits)!=len(rows) or set(edits)!={r['sha256'] for r in current['rows']}:
        raise ValueError('Inventaris tidak cocok atau berisi duplikat.')
    for row in current['rows']:
        edit=edits[row['sha256']]
        for field in ('group_id','label','approved','notes'):
            row[field]=edit[field]
        if not isinstance(row['approved'],bool):
            raise ValueError('approved harus boolean.')
        if not all(isinstance(row[k],str) and len(row[k])<=500 for k in ('group_id','label','notes')):
            raise ValueError('Nilai anotasi tidak valid.')
    frame=validate_annotation_rows(current['rows'])
    path=DATA_DIR/'annotations.csv'
    backups=DATA_DIR/'annotation_backups'
    backups.mkdir(exist_ok=True)
    shutil.copy2(path,backups/(secrets.token_hex(8)+'.csv'))
    temporary=path.with_suffix('.pending.csv')
    frame.to_csv(temporary,index=False)
    temporary.replace(path)
    return annotation_snapshot()

def import_own_photo(raw, name):
    """Original bytes retained; caller must explicitly attest this is an own photo."""
    if not isinstance(name,str) or not name.strip() or len(name)>250:
        raise ValueError('Nama file tidak valid.')
    if not raw or len(raw)>10_000_000:
        raise ValueError('Batas file 10 MB.')
    with Image.open(io.BytesIO(raw)) as image:
        suffix={'JPEG':'.jpg','PNG':'.png','WEBP':'.webp'}.get(image.format)
        if not suffix or getattr(image,'n_frames',1)!=1:
            raise ValueError('Gunakan JPG, PNG, atau WebP statis.')
        if image.width*image.height>25_000_000 or min(image.size)<32:
            raise ValueError('Dimensi harus minimal 32 piksel dan maksimal 25 megapiksel.')
        image.load()
    digest=hashlib.sha256(raw).hexdigest()
    if digest in {r['sha256'] for r in annotation_snapshot()['rows']}:
        return {'duplicate':True,'sha256':digest}
    RAW_DATA_DIR.mkdir(parents=True,exist_ok=True)
    path=RAW_DATA_DIR/f'UPLOAD_{digest}{suffix}'
    if not path.exists():
        with path.open('xb') as output:
            output.write(raw)
    init_labels()
    return {'duplicate':False,'sha256':digest}

def dataset_readiness():
    snapshot=annotation_snapshot()
    frame=pd.DataFrame(snapshot['rows'],columns=LABEL_COLUMNS)
    frame['group_id']=frame.group_id.astype(str).str.strip().str.casefold()
    approved=frame[frame.approved.astype(str).str.lower().isin(['true','1'])]
    counts={name:int((approved.label==name).sum()) for name in CLASS_NAMES}
    groups={name:int(approved[approved.label==name].group_id.nunique()) for name in CLASS_NAMES}
    blockers=[]
    warnings=[]
    preview=None
    if len(approved)==0:
        blockers.append('Belum ada label foto sendiri yang dikonfirmasi.')
    for name in CLASS_NAMES:
        if groups[name]<3:
            blockers.append(f'{name.replace("_"," ")}: perlu minimal 3 kelompok buah berbeda; sekarang {groups[name]}.')
    try:
        if len(frame):
            validate_annotation_rows(snapshot['rows'])
        if not blockers:
            grouped,pairs=group_near_duplicates(enrich(apply_session_groups(approved)))
            split,_=split_with_existing_holdout(grouped)
            validate_splits(split)
            preview={s:{'images':int((split.split==s).sum()),
                        'groups':int(split[split.split==s].group_id.nunique())}
                     for s in ['train','val','test']}
            if pairs:
                warnings.append(f'{len(pairs)} pasangan mirip digabung sebelum split.')
    except (ValueError,OSError,KeyError) as error:
        blockers.append(str(error))
    if len(approved)<300:
        warnings.append('Foto terkonfirmasi masih sedikit. Jumlah minimum teknis bukan jaminan generalisasi.')
    warnings.append('ID buah harus sesuai buah fisik. Aplikasi tidak dapat memverifikasi identitas hanya dari nama file.')
    return {'own_images':len(frame),'approved':len(approved),'unreviewed':len(frame)-len(approved),
            'class_counts':counts,'groups_per_class':groups,'split_preview':preview,
            'can_train':not blockers,'blockers':blockers,'warnings':warnings}
