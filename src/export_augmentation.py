"""Export balanced TRAIN views, with parent provenance; never independent samples."""
import argparse
import json
import numpy as np
import pandas as pd
from PIL import Image
from src.config import CLASS_NAMES, resolve_path
from src.manifest import load_bundle, relative, sha256
from src.preprocessing import load_arrays, make_augmentation, balanced_indices

def export(dataset, output, per_class=1000, size=224, seed=42):
    if per_class < 1 or size < 32:
        raise ValueError("Positive count and size >=32 required")
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    df, meta = load_bundle(dataset)
    train = df[df.split == "train"].reset_index(drop=True)
    images, labels = load_arrays(train, size)
    ids = balanced_indices(labels, per_class, np.random.default_rng(seed))
    folder = resolve_path(output)
    folder.mkdir(parents=True, exist_ok=False)
    aug = make_augmentation(seed)
    records = []
    for start in range(0, len(ids), 32):
        batch_ids = ids[start:start+32]
        views = np.clip(aug(images[batch_ids], training=True).numpy(), 0, 1)
        for offset, (parent_id, view) in enumerate(zip(batch_ids, views)):
            parent = train.iloc[parent_id]
            path = folder / parent.label / f"AUG{start+offset+1:06d}.jpg"
            path.parent.mkdir(exist_ok=True)
            Image.fromarray(np.rint(view*255).astype(np.uint8)).save(path, quality=95)
            records.append(dict(filepath=relative(path), sha256=sha256(path),
                                parent_filepath=parent.filepath, parent_sha256=parent.sha256,
                                group_id=parent.group_id, source=parent.source, label=parent.label,
                                split="train", is_augmented=True))
    pd.DataFrame(records).to_csv(folder/"manifest.csv", index=False)
    (folder/"dataset.json").write_text(json.dumps({
        "parent_dataset":meta["fingerprint"], "mode":meta["mode"], "seed":seed,
        "count":len(records), "per_class":per_class, "size":size,
        "independent_source_images":len(train),
        "warning":"Augmented TRAIN views only; not new photographs, never validation/test. Online training already augments; do not ingest twice."
    }, indent=2))
    print(f"Exported {len(records)} training views from {len(train)} source images to {folder}")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--per-class",type=int,default=1000)
    p.add_argument("--size",type=int,default=224)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    export(a.dataset,a.output,a.per_class,a.size,a.seed)

if __name__=="__main__":
    main()
