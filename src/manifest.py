"""Dataset inventory, manual labels, perceptual grouping, frozen split bundles."""
import argparse
import hashlib
import io
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from src.config import ROOT_DIR, DATA_DIR, CLASS_NAMES, FILENAME_PATTERN, RAW_DATA_DIR, resolve_path
from src.dataset_split import assign_group_splits, assign_locked_splits, validate_splits

LABEL_COLUMNS = ["filepath", "sha256", "group_id", "label", "approved", "source", "notes"]

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def image_fingerprint(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.load()
        pixels = hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
        gray = np.asarray(image.convert("L").resize((9,8), Image.Resampling.LANCZOS))
        bits = (gray[:,1:] > gray[:,:-1]).ravel()
        dhash = sum(int(bit) << i for i,bit in enumerate(bits))
        return pixels, f"{dhash:016x}", image.width, image.height

def relative(path):
    return str(Path(path).resolve().relative_to(ROOT_DIR))

def init_labels():
    """Inventory own photos without inventing labels or fruit identities."""
    target = DATA_DIR / "annotations.csv"
    previous = pd.read_csv(target, keep_default_na=False).to_dict("records") if target.exists() else []
    known = {r["sha256"]:r for r in previous}
    paths = [p for p in (DATA_DIR/"reference_import").rglob("*")
             if p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png",".webp"}]
    paths += [p for p in RAW_DATA_DIR.glob("*") if p.suffix.lower() in {".jpg",".jpeg",".png",".webp"}]
    duplicates, seen = [], set()
    for path in sorted(paths):
        digest = sha256(path)
        if digest in seen:
            duplicates.append({"filepath":relative(path), "sha256":digest})
            continue
        seen.add(digest)
        if digest not in known:
            known[digest] = dict(filepath=relative(path), sha256=digest, group_id="",
                                label="", approved=False, source="own", notes="")
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(known.values()), columns=LABEL_COLUMNS).to_csv(target, index=False)
    pd.DataFrame(duplicates, columns=["filepath","sha256"]).to_csv(DATA_DIR/"duplicates.csv", index=False)
    print(f"Own inventory: {len(known)} unique photos; {len(duplicates)} duplicate copies")
    return target

def validate_annotation_rows(rows):
    df = pd.DataFrame(rows)
    if not set(LABEL_COLUMNS).issubset(df.columns):
        raise ValueError(f"Required columns: {LABEL_COLUMNS}")
    if df.empty:
        raise ValueError("No photos inventoried")
    if df.sha256.duplicated().any():
        raise ValueError("Duplicate annotation hash")
    for row in df.to_dict("records"):
        if str(row['approved']).lower() not in {'true','false','0','1'}:
            raise ValueError('approved must be true/false or 0/1')
        path = resolve_path(row["filepath"]).resolve()
        if not (path.is_relative_to(DATA_DIR/"reference_import") or path.is_relative_to(RAW_DATA_DIR)):
            raise ValueError("Own labels may only reference imported photographs or data/raw")
        if row["source"] != "own" or sha256(path) != row["sha256"]:
            raise ValueError("Source/hash mismatch")
        if row["label"] and row["label"] not in CLASS_NAMES:
            raise ValueError("Unknown label")
        if str(row["approved"]).lower() in {"true","1"} and (not row["label"] or not str(row["group_id"]).strip()):
            raise ValueError("Approved photos require class and actual fruit group ID")
    return df[LABEL_COLUMNS]

def own_rows():
    path = DATA_DIR/"annotations.csv"
    if not path.exists():
        init_labels()
    df = validate_annotation_rows(pd.read_csv(path,keep_default_na=False).to_dict("records"))
    approved = df.approved.astype(str).str.lower().isin(["true","1"])
    rows = df[approved].copy()
    if rows.empty:
        raise ValueError("No approved own labels. Run python -m src.app, label photos and assign fruit IDs.")
    rows["group_id"] = "own:" + rows.group_id.astype(str).str.strip().str.casefold()
    return apply_session_groups(rows)


def apply_session_groups(rows):
    """Join audited sessions and conservatively link fruit IDs across sessions."""
    path = DATA_DIR / "sessions.csv"
    if not path.exists():
        return rows.copy()
    sessions = pd.read_csv(path, keep_default_na=False, dtype=str)
    if not {"file", "session_id"}.issubset(sessions.columns):
        raise ValueError("sessions.csv requires file and session_id")
    for key in ("file", "session_id"):
        sessions[key] = sessions[key].str.strip()
        if sessions[key].eq("").any():
            raise ValueError(f"Empty {key} in sessions.csv")
    if sessions.file.duplicated().any():
        raise ValueError("Duplicate file in sessions.csv")
    result = rows.copy()
    result["session_id"] = result.filepath.map(lambda p: Path(p).stem).map(sessions.set_index("file").session_id)
    if result.session_id.isna().any():
        raise ValueError("Every approved photo needs a session_id in data/sessions.csv")
    parent = {g: g for g in result.group_id}
    def root(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g
    for _, part in result.groupby("session_id"):
        groups = part.group_id.unique()
        for g in groups[1:]:
            a, b = root(groups[0]), root(g)
            parent[max(a, b)] = min(a, b)
    result["annotated_group_id"] = result.group_id
    result["group_id"] = result.group_id.map(root)
    return result


SYNTHETIC_LABELS = ROOT_DIR / "submission/dataset/labels.csv"

def synthetic_rows():
    """AI-generated images from the submission package; kept tagged source=synthetic_ai."""
    package = SYNTHETIC_LABELS.parent
    df = pd.read_csv(SYNTHETIC_LABELS, keep_default_na=False, dtype=str)
    df = df[df.source == "synthetic_ai"]
    if df.empty:
        raise ValueError("No synthetic_ai rows in " + relative(SYNTHETIC_LABELS))
    if not df.label.isin(CLASS_NAMES).all() or df.group_id.str.strip().eq("").any():
        raise ValueError("Synthetic rows need a known label and a scene group_id")
    return pd.DataFrame(dict(filepath=[relative(package / f) for f in df.file], sha256=df.sha256.values,
                             group_id=df.group_id.values, label=df.label.values, approved=True,
                             source="synthetic_ai", notes="included by owner request; labels not human-reviewed"))

def add_synthetic(own, synthetic):
    """Append synthetic images to the train split; they never touch validation or test."""
    synthetic = enrich(synthetic)
    synthetic = synthetic[~synthetic.pixel_sha256.isin(own.pixel_sha256)].drop_duplicates("pixel_sha256")
    synthetic = synthetic.assign(split="train", original_group_id=synthetic.group_id, annotated_group_id=synthetic.group_id)
    if "session_id" in own:
        synthetic["session_id"] = synthetic.group_id
    return pd.concat([own, synthetic], ignore_index=True)

def split_with_existing_holdout(base, seed=42):
    protocol = DATA_DIR.parent / "outputs/experiments/own_v2/protocol.json"
    if protocol.exists():
        metadata = json.loads(protocol.read_text())
        hashes = metadata["locked_test_sha256"]
        result = assign_locked_splits(base, hashes, seed)
        return result, {"locked_test_sha256": sorted(hashes), "holdout_protocol_sha256": sha256(protocol)}
    return assign_group_splits(base, seed), {}

def enrich(df):
    rows = []
    for row in df.to_dict("records"):
        path = resolve_path(row["filepath"])
        if sha256(path) != row["sha256"]:
            raise ValueError(f"Source changed: {path}")
        pixel, dhash, width, height = image_fingerprint(path)
        rows.append(dict(row, pixel_sha256=pixel, dhash=dhash, width=width, height=height,
                         label_idx=CLASS_NAMES.index(row["label"])))
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Dataset is empty")
    if (df.groupby("pixel_sha256").label.nunique()>1).any():
        raise ValueError("Conflicting labels on identical pixel data")
    # Keep duplicate records until their fruit IDs have been linked. Dropping first
    # could leave other views of the same fruit in different splits.
    return df.reset_index(drop=True)

def group_near_duplicates(df, distance=4):
    """Conservatively merge related groups by 64-bit dHash, report every merge."""
    parent = {g:g for g in df.group_id}
    def root(g):
        while parent[g]!=g:
            parent[g]=parent[parent[g]]
            g=parent[g]
        return g
    pairs = []
    hashes = [int(x,16) for x in df.dhash]
    groups=df.group_id.tolist()
    paths=df.filepath.tolist()
    for i in range(len(df)):
        for j in range(i):
            delta = (hashes[i]^hashes[j]).bit_count()
            if delta <= distance:
                a,b=root(groups[i]),root(groups[j])
                if a!=b:
                    parent[max(a,b)] = min(a,b)
                    pairs.append(dict(first=paths[j], second=paths[i], distance=delta))
    result = df.copy()
    result["original_group_id"] = result.group_id
    result["group_id"] = [root(g) for g in result.group_id]
    if 'pixel_sha256' in result:
        result=result.drop_duplicates('pixel_sha256').reset_index(drop=True)
    return result, pairs

def prepare(mode="own", seed=42):
    if mode not in ("own", "own_plus_synthetic"):
        raise ValueError("Unknown dataset mode")
    base = enrich(own_rows())
    base, pairs = group_near_duplicates(base)
    result, holdout = split_with_existing_holdout(base,seed)
    if mode == "own_plus_synthetic":
        result = add_synthetic(result, synthetic_rows())
    validate_splits(result)
    hashes=[int(x,16) for x in result.dhash]
    splits=result.split.tolist()
    for i in range(len(result)):
        for j in range(i):
            if splits[i] != splits[j] and (hashes[i]^hashes[j]).bit_count() <= 4:
                raise ValueError("Near duplicate across splits; review source photos before preparing")
    result=result.sort_values("filepath").reset_index(drop=True)
    settings={"mode":mode,"seed":seed,
              "classes":CLASS_NAMES,"dhash_distance":4,
              "label_policy":"manual approved own photos"+("; AI-generated images added to train only, tagged synthetic_ai, labels not human-reviewed" if mode!="own" else ""),
              "group_policy":"fruit ID plus audited sessions and conservative dHash connected components",
              **holdout,
              "limitation":"dHash cannot prove fruit independence; actual fruit IDs must be accurate"}
    folder=save_bundle(result,settings,pairs=pairs)
    print(json.dumps({"dataset":relative(folder),**json.loads((folder/'dataset.json').read_text())},indent=2))
    return folder

def save_bundle(result,settings,folder=None,pairs=None):
    """Write immutable bundles; used for preparation and development-only folds."""
    validate_splits(result)
    settings=dict(settings)
    if "locked_test_sha256" in settings and set(result[result.split == "test"].sha256) != set(settings["locked_test_sha256"]):
        raise ValueError("Manifest does not preserve the locked test")
    result=result.sort_values('filepath').reset_index(drop=True)
    payload=result.to_csv(index=False)
    fingerprint=hashlib.sha256((json.dumps(settings,sort_keys=True)+payload).encode()).hexdigest()
    folder=Path(folder) if folder is not None else DATA_DIR/"prepared"/fingerprint[:16]
    if folder.exists():
        _,existing=load_bundle(folder,verify=False)
        if (folder/'manifest.csv').read_text()!=payload or existing['fingerprint']!=fingerprint:
            raise ValueError('Refusing to overwrite a different frozen bundle')
        return folder
    folder.mkdir(parents=True,exist_ok=False)
    (folder/"manifest.csv").write_text(payload)
    settings["fingerprint"]=fingerprint
    settings["counts"]={s:result[result.split==s].label.value_counts().to_dict() for s in ("train","val","test")}
    settings["groups"]={s:int(result[result.split==s].group_id.nunique()) for s in ("train","val","test")}
    settings["unique_images"]=len(result)
    settings["near_duplicate_merges"]=pairs or []
    (folder/"dataset.json").write_text(json.dumps(settings,indent=2))
    return folder

def load_bundle(folder, verify=True):
    folder=resolve_path(folder)
    meta=json.loads((folder/"dataset.json").read_text())
    df=pd.read_csv(folder/"manifest.csv",keep_default_na=False)
    validate_splits(df)
    if "locked_test_sha256" in meta and set(df[df.split == "test"].sha256) != set(meta["locked_test_sha256"]):
        raise ValueError("Manifest does not preserve the locked test")
    settings={k:v for k,v in meta.items() if k not in
              {"fingerprint","counts","groups","unique_images","near_duplicate_merges"}}
    fingerprint=hashlib.sha256((json.dumps(settings,sort_keys=True)+(folder/"manifest.csv").read_text()).encode()).hexdigest()
    if fingerprint!=meta["fingerprint"]:
        raise ValueError("Prepared manifest/config was modified. Prepare a new version.")
    expected_counts={s:df[df.split==s].label.value_counts().to_dict() for s in ('train','val','test')}
    expected_groups={s:int(df[df.split==s].group_id.nunique()) for s in ('train','val','test')}
    if meta['counts']!=expected_counts or meta['groups']!=expected_groups or meta['unique_images']!=len(df):
        raise ValueError('Bundle summary does not match the actual manifest')
    if verify:
        for row in df.to_dict("records"):
            if sha256(resolve_path(row["filepath"]))!=row["sha256"]:
                raise ValueError(f"Image has changed: {row['filepath']}")
    return df,meta

def build_manifest(data_dir=RAW_DATA_DIR):
    """Legacy filename reader for photographed data; new training uses bundles."""
    rows=[]
    for path in sorted(Path(data_dir).glob("*.jpg")):
        match=re.match(FILENAME_PATTERN,path.name)
        if match:
            d=match.groupdict()
            rows.append(dict(filepath=str(path),filename=path.name,**d,label_idx=CLASS_NAMES.index(d["label"])))
    if not rows:
        raise ValueError("No correctly named photographs. Use the annotation workflow.")
    return pd.DataFrame(rows)

def summarize_manifest(df):
    print(df.label.value_counts().to_string())

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--init-labels",action="store_true")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--with-synthetic",action="store_true",help="add submission/dataset synthetic_ai images to train only")
    args=parser.parse_args()
    if args.init_labels:
        init_labels()
    else:
        prepare("own_plus_synthetic" if args.with_synthetic else "own",args.seed)

if __name__=="__main__":
    main()
