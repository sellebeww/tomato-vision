"""Export balanced TRAIN views, with parent provenance; never independent samples."""
import argparse
import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from src.config import CLASS_NAMES, resolve_path
from src.manifest import load_bundle, relative, sha256, image_fingerprint
from src.preprocessing import load_arrays, make_augmentation, balanced_indices

def export(dataset, output, per_class=120, size=224, seed=42):
    if not isinstance(per_class, int) or isinstance(per_class, bool) or per_class < 1 or not 32 <= size <= 512:
        raise ValueError("Positive integer count and size 32..512 required")
    folder = resolve_path(output).resolve()
    relative(folder)  # Fail before writing if the output is outside this project.
    if folder.exists():
        raise FileExistsError(f"Refusing to overwrite {folder}")
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    df, meta = load_bundle(dataset)
    train = df[df.split == "train"].reset_index(drop=True)
    images, labels = load_arrays(train, size)
    ids = balanced_indices(labels, per_class, np.random.default_rng(seed))
    aug = make_augmentation(seed)
    folder.parent.mkdir(parents=True, exist_ok=True)
    staging = tempfile.TemporaryDirectory(prefix=".augmentation-", dir=folder.parent)
    stage = Path(staging.name)
    records = []
    try:
        for start in range(0, len(ids), 32):
            batch_ids = ids[start:start+32]
            views = np.clip(aug(images[batch_ids], training=True).numpy(), 0, 1)
            for offset, (parent_id, view) in enumerate(zip(batch_ids, views)):
                parent = train.iloc[parent_id]
                name = f"AUG{start+offset+1:06d}.jpg"
                path = stage / parent.label / name
                path.parent.mkdir(exist_ok=True)
                Image.fromarray(np.rint(view*255).astype(np.uint8)).save(path, quality=95)
                pixel, _, _, _ = image_fingerprint(path)
                records.append(dict(filepath=relative(folder/parent.label/name), sha256=sha256(path),
                                    pixel_sha256=pixel,
                                    parent_filepath=parent.filepath, parent_sha256=parent.sha256,
                                    group_id=parent.group_id, session_id=parent.get("session_id", ""),
                                    source="augmented", parent_source=parent.source, label=parent.label,
                                    label_idx=int(parent.label_idx), split="train", is_augmented=True))
        generated = pd.DataFrame(records)
        if generated.pixel_sha256.duplicated().any() or generated.pixel_sha256.isin(df.pixel_sha256).any():
            raise ValueError("Duplicate generated pixels; choose another seed")
        generated.to_csv(stage/"manifest.csv", index=False)
        originals = df.assign(is_augmented=False, parent_filepath="", parent_sha256="", parent_source="")
        pd.concat([originals, generated], ignore_index=True).fillna("").to_csv(stage/"annotations.csv", index=False)
        (stage/"dataset.json").write_text(json.dumps({
        "parent_dataset":meta["fingerprint"], "mode":meta["mode"], "seed":seed,
        "count":len(records), "per_class":per_class, "size":size,
        "source_images":len(train), "source_groups":int(train.group_id.nunique()),
        "original_images":len(df), "combined_images":len(df)+len(records),
        "class_counts":generated.label.value_counts().to_dict(),
        "augmentation_config":aug.get_config(), "tensorflow_version":tf.__version__,
        "warning":"Augmented TRAIN views only; not new photographs, never validation/test. Online training already augments; do not ingest twice."
        }, indent=2))
        shutil.copytree(resolve_path(dataset), stage/"parent_dataset")
        (stage/"README.md").write_text(
            f"# Dataset training beraugmentasi\n\n{len(records)} gambar baru ({per_class}/kelas), "
            f"diturunkan dari {len(train)} foto training dalam {train.group_id.nunique()} grup.\n\n"
            "`manifest.csv`: variasi training dengan hash dan parent. `annotations.csv`: inventaris "
            "gabungan foto asli dan variasi. `parent_dataset/`: split asli yang dibekukan.\n\n"
            "Label variasi diwarisi dari parent; bukan anotasi independen. Validation/test tetap "
            "foto asli. Jangan split ulang inventaris gabungan atau menghitung variasi sebagai "
            "buah baru. Pipeline training memakai parent_dataset dengan augmentasi online; "
            "jangan masukkan ekspor ini lagi karena akan menggandakan augmentasi.\n",
            encoding="utf-8")
        stage.rename(folder)
    finally:
        staging.cleanup()
    print(f"Exported {len(records)} training views from {len(train)} source images to {folder}")
    return folder

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--per-class",type=int,default=120)
    p.add_argument("--size",type=int,default=224)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    export(a.dataset,a.output,a.per_class,a.size,a.seed)

if __name__=="__main__":
    main()
