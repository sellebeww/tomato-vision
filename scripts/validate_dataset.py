"""Periksa paket submission/dataset: penamaan, keunikan, hash, split, dan berkas yatim."""
import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {"segar", "tidak_segar", "busuk"}
NAME = {
    "own": re.compile(r"^images/(?P<id>(?:own|ref)_\d{3})_(?P<label>segar|tidak_segar|busuk)\.(?:png|jpeg)$"),
    "synthetic_ai": re.compile(r"^images/(?P<id>syn_\d{3})_(?P<label>segar|tidak_segar|busuk)\.png$"),
    "unlabeled": re.compile(r"^images/(?P<id>ref_\d{3})\.jpeg$"),
}


def check(folder):
    """Kembalikan daftar pelanggaran; kosong berarti paket konsisten."""
    folder = Path(folder)
    with (folder / "labels.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    problems = []
    for key in ("id", "file", "sha256"):
        values = [r[key] for r in rows]
        if len(set(values)) != len(values):
            problems.append(f"kolom {key} tidak unik")

    groups = {}
    for row in rows:
        kind = "unlabeled" if not row["label"] else row["source"]
        match = NAME.get(kind) and NAME[kind].match(row["file"])
        if not match:
            problems.append(f"nama/folder tidak sesuai pola: {row['file']}")
            continue
        if match["id"] != row["id"] or ("label" in match.groupdict() and match["label"] != row["label"]):
            problems.append(f"id/label {row['id']} tidak cocok dengan nama {row['file']}")
        if row["label"] and row["label"] not in CLASSES:
            problems.append(f"label tidak dikenal: {row['id']}")
        path = folder / row["file"]
        if not path.is_file():
            problems.append(f"berkas hilang: {row['file']}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            problems.append(f"hash berbeda: {row['file']}")
        if row["source"] == "synthetic_ai" and (row["split"] in ("val", "test") or row["approved"] == "True"):
            problems.append(f"sintetis tidak boleh di val/test atau disetujui: {row['id']}")
        if row["source"] == "own" and row["label"] and row["split"] not in ("train", "val", "test"):
            problems.append(f"foto berlabel tanpa split: {row['id']}")
        if not row["label"] and (row["split"] != "unassigned" or row["approved"] == "True"):
            problems.append(f"foto tanpa label harus unassigned dan belum disetujui: {row['id']}")
        if row["label"]:
            groups.setdefault(row["group_id"], set()).add(row["split"])
    problems += [f"grup terpecah antar split: {g} -> {sorted(s)}" for g, s in groups.items() if len(s) > 1]

    listed = {r["file"] for r in rows}
    on_disk = {p.relative_to(folder).as_posix() for p in (folder / "images").rglob("*")
               if p.is_file() and p.name != ".DS_Store"}
    problems += [f"subfolder tidak diharapkan: {d.relative_to(folder).as_posix()}"
                 for d in (folder / "images").rglob("*") if d.is_dir()]
    problems += [f"berkas tidak tercatat di labels.csv: {f}" for f in sorted(on_disk - listed)]
    return problems, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", default="submission/dataset")
    args = parser.parse_args()
    problems, rows = check(ROOT / args.folder)
    print(json.dumps({"images": len(rows), "problems": problems}, indent=2, ensure_ascii=False))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
