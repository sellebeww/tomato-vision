"""Group-disjoint splitting with class coverage and fail-closed leakage checks."""
import numpy as np
import pandas as pd
from src.config import CLASS_NAMES, RANDOM_SEED

def validate_splits(df):
    required={"split","group_id","sha256","pixel_sha256","label","label_idx","source"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing manifest columns: {sorted(required-set(df.columns))}")
    expected=df.label.map({name:i for i,name in enumerate(CLASS_NAMES)})
    if expected.isna().any() or not expected.eq(df.label_idx).all():
        raise ValueError("Class labels and integer indexes disagree")
    if not df.source.isin(["own"]).all():
        raise ValueError("Unsupported source; public/dummy data is not accepted")
    if set(df.split) != {"train", "val", "test"}:
        raise ValueError("Need nonempty train, val and test splits")
    for key in ("group_id", "sha256", "pixel_sha256"):
        if key not in df or df[key].isna().any() or (df[key].astype(str).str.len() == 0).any():
            raise ValueError(f"Missing {key}")
        if (df.groupby(key).split.nunique() > 1).any():
            raise ValueError(f"Leakage: {key} occurs in multiple splits")
    for split in ("train", "val", "test"):
        subset = df[df.split == split]
        if set(subset.label) != set(CLASS_NAMES):
            raise ValueError(f"All three classes required in {split}")
    if df.groupby("pixel_sha256").label.nunique().max() > 1:
        raise ValueError("Identical image has contradictory labels")

def assign_group_splits(df, seed=RANDOM_SEED, ratios=(0.7, 0.15, 0.15)):
    """Allocate groups using label counts only, never model performance."""
    if len(ratios) != 3 or min(ratios) <= 0 or not np.isclose(sum(ratios), 1):
        raise ValueError("Three positive ratios summing to one required")
    groups = np.array(sorted(df.group_id.unique()))
    if len(groups) < 3:
        raise ValueError("Need at least three independent fruit groups; never split one fruit by photo")
    counts = pd.crosstab(df.group_id, df.label).reindex(index=groups, columns=CLASS_NAMES, fill_value=0).to_numpy()
    if np.any((counts > 0).sum(axis=0) < 3):
        raise ValueError("Each class needs at least three independent groups for train/val/test")
    n_val, n_test = max(1, round(len(groups)*ratios[1])), max(1, round(len(groups)*ratios[2]))
    n_train = len(groups) - n_val - n_test
    if n_train < 1:
        raise ValueError("Not enough training groups")
    rng = np.random.default_rng(seed)
    best, best_score = None, float("inf")
    for _ in range(2048):
        order = rng.permutation(len(groups))
        parts = np.split(order, [n_train, n_train+n_val])
        values = np.array([counts[idx].sum(axis=0) for idx in parts])
        if (values == 0).any():
            continue
        score = np.abs(values/counts.sum(axis=0)-np.array(ratios)[:,None]).sum()
        if score < best_score:
            best, best_score = parts, score
    if best is None:
        raise ValueError("No group-disjoint split covering every class; collect more fruits")
    mapping = {groups[i]: s for s, part in zip(("train","val","test"), best) for i in part}
    result = df.copy()
    result["split"] = result.group_id.map(mapping)
    return result

def split_by_tomato_id(df, train_ratio=.7, val_ratio=.15, test_ratio=.15, seed=RANDOM_SEED):
    df = df.copy()
    df["group_id"] = df["tomato_id"]
    result = assign_group_splits(df, seed, (train_ratio,val_ratio,test_ratio))
    return tuple(result[result.split==s].reset_index(drop=True) for s in ("train","val","test"))

def split_two_tomatoes(*args, **kwargs):
    raise ValueError("Two fruits cannot form independent train/validation/test. Collect more fruit IDs.")
