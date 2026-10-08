"""Phase-1 data audit: split distribution, near-duplicates across splits, and session structure.

Reproduce: python -m src.audit_leakage --run outputs/experiments/own_v1/baseline --output outputs/audit/own_v1
Hashes cannot prove fruit identity; contact sheets are written for visual review by the data owner.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.fft import dctn
from src.config import CLASS_NAMES, resolve_path

NEAR_DUPLICATE_BITS = 8   # dHash/pHash distance at or below this is treated as near-duplicate


def hashes(path):
    gray = Image.open(path).convert("L")
    a = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=float)
    b = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    coefficients = dctn(b, norm="ortho")[:8, :8].ravel()
    return (a[:, 1:] > a[:, :-1]).ravel(), coefficients > np.median(coefficients[1:])


def contact_sheet(frame, out, by="group_id", size=150):
    groups = list(frame.groupby(by))
    width = max(len(g) for _, g in groups)
    sheet = Image.new("RGB", (110 + width * size, len(groups) * (size + 16)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, (name, part) in enumerate(groups):
        y = row * (size + 16)
        draw.text((4, y + size // 2), f"{name}\n{'/'.join(sorted(set(part.split)))}", fill="black")
        for col, item in enumerate(part.sort_values("filepath").itertuples()):
            with Image.open(resolve_path(item.filepath)) as image:
                sheet.paste(image.convert("RGB").resize((size, size)), (110 + col * size, y))
            draw.text((110 + col * size + 2, y + size + 2), Path(item.filepath).stem[:16], fill="black")
    sheet.save(out, quality=88)


def audit(run, output, sessions=None):
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(run / "dataset" / "manifest.csv")
    frame["file"] = frame.filepath.map(lambda p: Path(p).stem)
    counts = pd.crosstab(frame.split, frame.label).reindex(index=["train", "val", "test"], columns=CLASS_NAMES)
    counts.to_csv(output / "split_counts.csv")
    codes = {r.file: hashes(resolve_path(r.filepath)) for r in frame.itertuples()}
    rows = []
    for x, y in itertools.combinations(frame.itertuples(), 2):
        (dx, px), (dy, py) = codes[x.file], codes[y.file]
        rows.append({"a": x.file, "b": y.file, "split_a": x.split, "split_b": y.split,
                     "group_a": x.group_id, "group_b": y.group_id,
                     "dhash": int((dx != dy).sum()), "phash": int((px != py).sum())})
    pairs = pd.DataFrame(rows)
    pairs.to_csv(output / "pairwise_hash.csv", index=False)
    cross = pairs[pairs.split_a != pairs.split_b]
    near = cross[(cross.dhash <= NEAR_DUPLICATE_BITS) | (cross.phash <= NEAR_DUPLICATE_BITS)]
    cross.nsmallest(15, ["phash", "dhash"]).to_csv(output / "closest_cross_split.csv", index=False)
    contact_sheet(frame, output / "groups.jpg")
    summary = {
        "run": str(run), "images": int(len(frame)), "groups": int(frame.group_id.nunique()),
        "split_counts": counts.to_dict(orient="index"),
        "groups_per_split": frame.groupby("split").group_id.nunique().to_dict(),
        "near_duplicate_threshold_bits": NEAR_DUPLICATE_BITS,
        "cross_split_min_dhash": int(cross.dhash.min()), "cross_split_min_phash": int(cross.phash.min()),
        "cross_split_near_duplicates": int(len(near)),
    }
    if sessions is not None:
        mapping = pd.read_csv(sessions)
        merged = frame.merge(mapping, on="file", how="left")
        if merged.session_id.isna().any():
            raise ValueError("Session file does not cover every image")
        spread = merged.groupby("session_id").split.agg(lambda s: sorted(set(s)))
        leaking = {k: v for k, v in spread.items() if len(v) > 1}
        summary["sessions"] = int(merged.session_id.nunique())
        summary["sessions_spanning_splits"] = leaking
        summary["images_in_leaking_sessions_by_split"] = (
            merged[merged.session_id.isin(leaking)].groupby("split").size().to_dict())
        contact_sheet(merged, output / "sessions.jpg", by="session_id")
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="outputs/experiments/own_v1/baseline")
    parser.add_argument("--output", default="outputs/audit/own_v1")
    parser.add_argument("--sessions", help="CSV with columns file,session_id (data owner's session/fruit map)")
    args = parser.parse_args()
    print(json.dumps(audit(resolve_path(args.run), resolve_path(args.output),
                          resolve_path(args.sessions) if args.sessions else None), indent=2))


if __name__ == "__main__":
    main()
