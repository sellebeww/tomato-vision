"""Gabungkan paket dataset lama menjadi satu paket dengan penamaan konsisten.

Tata letak baru (semua nama huruf kecil, nomor 3 digit, label di akhir nama):

    labels.csv                      satu tabel induk, satu baris per gambar
    prompts.jsonl                   rencana prompt gambar sintetis (kunci: id)
    report.json                     ringkasan jumlah dan pemeriksaan integritas
    images/own_001_segar.png        foto asli
    images/syn_001_segar.png        gambar sintetis (AI)
    images/ref_001.jpeg             foto impor, label belum dikonfirmasi pemilik
    (semua foto di satu folder `images/`, tanpa subfolder)

Nomor sama dengan nama lama (TOM001 -> own_001, SYN001 -> syn_001,
REF001 -> ref_001); nama lama dicatat di kolom `original_name`.

Skrip ini migrasi satu kali: membaca paket tata letak lama, menulis paket baru ke
folder staging, memverifikasi setiap hash, lalu (dengan --swap) menukar folder.
Paket lama dipindahkan ke `submission/dataset_lama`, tidak dihapus.
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("segar", "tidak_segar", "busuk")
FACTORS = ("background", "lighting", "subject_scale", "viewpoint", "morphology",
           "surface_moisture", "focus_condition")
COLUMNS = ["id", "file", "label", "source", "split", "group_id", "approved", "review_status",
           "width", "height", "sha256", "original_name", *FACTORS,
           "suggested_label", "model_prediction", "model_confidence", "needs_review", "notes"]
OLD_COLUMNS = ["file", "label", "group_id", "split", "width", "height", "sha256", "source",
               "approved", "is_augmented", "prompt_id", "review_status"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_old_labels(path):
    """labels.csv lama punya 147 baris sintetis tanpa kolom `source`; pulihkan kolom itu."""
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        assert header == OLD_COLUMNS, header
        for number, row in enumerate(reader, start=2):
            if len(row) == 11 and "/synthetic/" in row[0]:
                row.insert(7, "synthetic_ai")
            if len(row) != 12:
                raise ValueError(f"{path}:{number} punya {len(row)} kolom")
            rows.append(dict(zip(header, row)))
    return rows


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def build(legacy, stage):
    old = read_old_labels(legacy / "labels.csv")
    variation = {r["id"]: r for r in read_csv(legacy / "variation_manifest.csv")}
    references = {r["id"]: r for r in read_csv(legacy / "reference_review.csv")}
    out, counters = [], Counter()

    for row in old:
        name = Path(row["file"]).name
        stem, _, extra = name.rpartition(".")
        if row["source"] == "own" and name.startswith("TOM"):
            kind, new_id = "own", "own_" + stem[3:6]
            new_name = f"{new_id}_{row['label']}.{extra}"
            folder = Path("images")
            status, approved = "approved", True
            extra_fields = {}
        elif row["source"] == "synthetic_ai":
            kind, new_id = "synthetic", "syn_" + stem[3:6]
            new_name = f"{new_id}_{row['label']}.{extra}"
            folder = Path("images")
            status, approved = "pending_review", False
            factors = variation[stem[:6]]
            extra_fields = {f: factors[f] for f in FACTORS}
        elif row["source"] == "own" and name.startswith("REF"):
            kind, new_id = "unlabeled", "ref_" + stem[3:6]
            new_name = f"{new_id}.{extra}"
            folder = Path("images")
            status, approved = "unlabeled", False
            ref = references[stem[:6]]
            extra_fields = {"original_name": Path(ref["filepath"]).name,
                            "suggested_label": ref["suggested_visual_label"],
                            "model_prediction": ref["model_prediction"],
                            "model_confidence": f"{float(ref['model_confidence']):.4f}",
                            "needs_review": ref["needs_review"],
                            "notes": ref["quality_notes"]}
        else:
            raise ValueError(f"Baris tidak dikenali: {row['file']}")

        source = legacy / row["file"]
        digest = sha256(source)
        if digest != row["sha256"]:
            raise ValueError(f"Hash berbeda: {row['file']}")
        target = stage / folder / new_name
        link(source, target)
        counters[kind] += 1
        record = dict.fromkeys(COLUMNS, "")
        record.update(id=new_id, file=target.relative_to(stage).as_posix(), label=row["label"],
                      source="synthetic_ai" if kind == "synthetic" else "own", split=row["split"],
                      group_id=row["group_id"], approved=str(approved), review_status=status,
                      width=row["width"], height=row["height"], sha256=digest, original_name=name)
        record.update(extra_fields)
        out.append(record)

    out.sort(key=lambda r: ({"own": 0, "syn": 1, "ref": 2}[r["id"][:3]], r["id"]))
    with open(stage / "labels.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(out)
    return out, counters


def write_prompts(legacy, stage, records):
    saved = {r["id"]: r["file"] for r in records if r["id"].startswith("syn_")}
    plan = [json.loads(line) for line in (legacy / "generation_prompts_300.jsonl").read_text().splitlines()]
    with open(stage / "prompts.jsonl", "w", encoding="utf-8") as handle:
        for job in plan:
            new_id = "syn_" + job["id"][3:]
            entry = {"id": new_id, "label": job["label"], "group_id": job["group_id"],
                     "saved": new_id in saved, "file": saved.get(new_id, ""), "prompt": job["prompt"]}
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return plan


def write_report(stage, records, plan):
    def count(rows, key):
        return dict(Counter(r[key] for r in rows))

    synthetic = [r for r in records if r["id"].startswith("syn_")]
    labeled = [r for r in records if r["label"]]
    saved = {r["id"] for r in synthetic}
    planned = sorted("syn_" + job["id"][3:] for job in plan)
    report = {
        "total_images": len(records),
        "by_source": count(records, "source"),
        "by_label": {k: dict(Counter(r["label"] for r in labeled if r["source"] == k))
                     for k in ("own", "synthetic_ai")},
        "unlabeled_images": len(records) - len(labeled),
        "by_split": count(labeled, "split"),
        "synthetic": {"planned": len(planned), "saved": len(saved),
                      "missing_ids": [i for i in planned if i not in saved]},
        "intended_factor_counts_by_label": {
            f: {v: dict(Counter(r["label"] for r in synthetic if r[f] == v))
                for v in sorted({r[f] for r in synthetic})} for f in FACTORS},
        "all_file_hashes_verified": all(sha256(stage / r["file"]) == r["sha256"] for r in records),
        "synthetic_in_validation_or_test": sum(r["split"] in ("val", "test") for r in synthetic),
        "factor_values_are_prompt_instructions_not_measured_ground_truth": True,
        "human_review_required_for_synthetic": True,
        "accuracy_improvement_measured": False,
    }
    (stage / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def write_zip(stage, destination):
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(stage.rglob("*")):
            if path.is_file() and path.suffix != ".zip" and path.name != ".DS_Store":
                archive.write(path, path.relative_to(stage).as_posix())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--legacy", default="submission/dataset", help="paket tata letak lama")
    parser.add_argument("--stage", default="submission/dataset_baru", help="folder paket baru")
    parser.add_argument("--readme", default=None, help="README yang disalin ke paket baru")
    parser.add_argument("--swap", action="store_true",
                        help="pindahkan paket lama ke submission/dataset_lama, jadikan paket baru submission/dataset")
    args = parser.parse_args()
    legacy, stage = ROOT / args.legacy, ROOT / args.stage
    if stage.exists():
        sys.exit(f"{stage} sudah ada; hapus dulu bila ingin membangun ulang")

    records, counters = build(legacy, stage)
    plan = write_prompts(legacy, stage, records)
    report = write_report(stage, records, plan)
    if args.readme:
        shutil.copy2(ROOT / args.readme, stage / "README.md")
    write_zip(stage, stage / "TomatoVision_Dataset.zip")
    print(json.dumps({"dibuat": dict(counters), "verifikasi_hash": report["all_file_hashes_verified"]}, indent=2))

    if args.swap:
        archive = legacy.parent / "dataset_lama"
        if archive.exists():
            sys.exit(f"{archive} sudah ada")
        legacy.rename(archive)
        stage.rename(legacy)
        print(f"Paket lama -> {archive.relative_to(ROOT)}; paket baru -> {legacy.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
