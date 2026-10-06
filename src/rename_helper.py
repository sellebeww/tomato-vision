"""
rename_helper.py
=================
Helper interaktif untuk memberi nama file foto tomat SESUAI KONVENSI
sejak hari pertama pemotretan, supaya tidak ada inkonsistensi penamaan
manual (typo, urutan kebalik, dsb).

Konvensi nama file (lihat src/config.py):
    [ID_TOMAT]_[HARI]_[SUDUT]_[LABEL].jpg
    contoh: T1_D03_atas_segar.jpg

CARA PAKAI (skenario harian):
    1. Foto 10 gambar hari ini (2 tomat x 5 sudut), taruh semua di SATU
       folder sumber sembarang, misal folder "DCIM" hasil transfer HP,
       dengan nama file asli apa pun (IMG_001.jpg, dst).
    2. Jalankan:

           python -m src.rename_helper --source "/path/ke/folder/hp" --day 3

       lalu ikuti instruksi interaktif: untuk tiap file, pilih ID tomat,
       sudut, dan label -> script akan COPY (bukan pindah asli) file
       tsb ke data/raw/ dengan nama baru sesuai konvensi.

    3. Bisa juga langsung non-interaktif kalau urutan file foto sudah
       konsisten tiap hari (lihat fungsi `batch_rename_by_order` di bawah).

Jalankan dengan --help untuk lihat semua opsi:
    python -m src.rename_helper --help
"""

import argparse
import re
from pathlib import Path
from PIL import Image, ImageOps

from src.config import ANGLE_LABELS, CLASS_NAMES, RAW_DATA_DIR, TOMATO_IDS

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def _copy_photo(source, destination):
    """Actually encode JPEG and never silently replace an existing photo."""
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        with Path(destination).open("xb") as output:
            image.save(output, format="JPEG", quality=95)


def _prompt_choice(prompt: str, options: list) -> str:
    """Tampilkan daftar pilihan bernomor, minta user pilih salah satu."""
    print(f"\n{prompt}")
    for i, opt in enumerate(options, start=1):
        print(f"  {i}. {opt}")
    while True:
        raw = input("Pilih nomor: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print("Input tidak valid, coba lagi.")


def rename_interactive(source_dir: Path, day: int, dest_dir: Path = RAW_DATA_DIR):
    """
    Mode interaktif: untuk setiap file gambar di source_dir, tanya ke user
    ID tomat / sudut / label lewat terminal, lalu copy dengan nama baru
    ke dest_dir sesuai konvensi penamaan.
    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    day_str = f"D{day:02d}"
    image_files = sorted(
        f for f in source_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        print(f"Tidak ada file gambar ditemukan di {source_dir}")
        return

    print(f"Ditemukan {len(image_files)} file gambar di {source_dir}. Hari: {day_str}")

    for f in image_files:
        print(f"\n=== File: {f.name} ===")
        tomato_id = _prompt_choice("Pilih ID tomat:", TOMATO_IDS)
        angle = _prompt_choice("Pilih sudut foto:", ANGLE_LABELS)
        label = _prompt_choice("Pilih label kesegaran:", CLASS_NAMES)

        new_name = f"{tomato_id}_{day_str}_{angle}_{label}.jpg"
        dest_path = dest_dir / new_name

        if dest_path.exists():
            print(f"Dilewati: {new_name} sudah ada; foto lama tidak ditimpa.")
            continue

        _copy_photo(f, dest_path)
        print(f"-> Disimpan sebagai: {dest_path}")

    print(f"\nSelesai. Semua file baru ada di: {dest_dir}")


def batch_rename_by_order(source_dir: Path, day: int, tomato_order: list, angle_order: list = None,
                           label: str = None, dest_dir: Path = RAW_DATA_DIR):
    """
    Mode cepat NON-interaktif: dipakai kalau urutan foto tiap hari SELALU
    konsisten (misal selalu difoto: T1 atas, T1 serong_atas, ..., T2 atas,
    T2 serong_atas, ...). Urutan file di source_dir (diurutkan alfabetis/
    berdasarkan waktu modifikasi) dipetakan langsung ke kombinasi
    (tomato_id, angle) berikut urutan `tomato_order` x `angle_order`.

    Args:
        source_dir: folder berisi foto asli hasil transfer dari kamera/HP
        day: hari ke berapa (integer, misal 3 -> "D03")
        tomato_order: urutan ID tomat sesuai urutan pemotretan, misal ["T1", "T2"]
        angle_order: urutan sudut sesuai urutan pemotretan (default: ANGLE_LABELS)
        label: label kesegaran untuk SEMUA foto hari ini (biasanya semua
               tomat di hari yang sama punya label yang mirip, tapi kalau
               beda per tomat, gunakan mode interaktif saja)
        dest_dir: folder tujuan (default: RAW_DATA_DIR)
    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    angle_order = angle_order or ANGLE_LABELS

    if label is None:
        raise ValueError("Parameter 'label' wajib diisi untuk mode batch (segar/tidak_segar/busuk).")
    if label not in CLASS_NAMES:
        raise ValueError(f"Label '{label}' tidak dikenal. Pilihan: {CLASS_NAMES}")

    day_str = f"D{day:02d}"
    image_files = sorted(
        f for f in source_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
    )

    expected_count = len(tomato_order) * len(angle_order)
    if len(image_files) != expected_count:
        raise ValueError(
            f"Jumlah file ({len(image_files)}) tidak sama dengan yang diharapkan "
            f"({expected_count} = {len(tomato_order)} tomat x {len(angle_order)} sudut). "
            "Gunakan mode interaktif untuk kontrol manual per file."
        )

    idx = 0
    targets = [dest_dir / f"{t}_{day_str}_{a}_{label}.jpg" for t in tomato_order for a in angle_order]
    if any(not re.fullmatch(r"[A-Za-z0-9_]+", str(v)) for v in list(tomato_order)+list(angle_order)):
        raise ValueError("IDs and angles must contain only letters, digits, underscores")
    if len(set(targets)) != len(targets) or any(p.exists() for p in targets):
        raise FileExistsError("Duplicate/existing destination; no photos copied")
    for tomato_id in tomato_order:
        for angle in angle_order:
            src_file = image_files[idx]
            new_name = f"{tomato_id}_{day_str}_{angle}_{label}.jpg"
            dest_path = dest_dir / new_name
            _copy_photo(src_file, dest_path)
            print(f"{src_file.name} -> {new_name}")
            idx += 1

    print(f"\nSelesai. {expected_count} file disalin ke: {dest_dir}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rename foto tomat harian sesuai konvensi penamaan.")
    parser.add_argument("--source", required=True, help="Folder sumber berisi foto hasil pemotretan hari ini")
    parser.add_argument("--day", required=True, type=int, help="Hari ke berapa (contoh: 3 untuk D03)")
    parser.add_argument("--mode", choices=["interactive", "batch"], default="interactive",
                         help="interactive: tanya per file lewat terminal. batch: pakai urutan tetap (lebih cepat)")
    parser.add_argument("--label", choices=CLASS_NAMES, default=None,
                         help="Wajib diisi untuk --mode batch: label kesegaran semua foto hari ini")
    parser.add_argument("--dest", default=str(RAW_DATA_DIR), help="Folder tujuan (default: data/raw)")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    if args.mode == "interactive":
        rename_interactive(source_dir=Path(args.source), day=args.day, dest_dir=Path(args.dest))
    else:
        batch_rename_by_order(
            source_dir=Path(args.source),
            day=args.day,
            tomato_order=TOMATO_IDS,
            label=args.label,
            dest_dir=Path(args.dest),
        )
