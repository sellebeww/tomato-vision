"""
generate_dummy_data.py
=======================
Membuat dataset gambar dummy (bukan foto asli) dengan struktur dan
konvensi penamaan file yang PERSIS SAMA seperti dataset asli nantinya.

Tujuan: supaya seluruh pipeline (preprocessing, split, training, evaluasi)
bisa dites SEKARANG, sebelum dataset asli (yang masih difoto tiap hari)
selesai dikumpulkan.

Cara kerja gambar dummy:
- Setiap gambar adalah kanvas dengan blob/lingkaran bulat (meniru bentuk
  tomat) yang warnanya bergantung pada LABEL:
    - "segar"       -> merah cerah, sedikit noise, permukaan mulus
    - "tidak_segar" -> merah pudar / oranye kecoklatan, noise sedang,
                       mulai ada bercak
    - "busuk"       -> coklat gelap/kehitaman, banyak noise & bercak
                       gelap tidak beraturan
- Supaya lebih realistis, ditambahkan progresi bertahap: semakin besar
  nomor hari (day index) untuk 1 ID tomat, warnanya makin bergeser dari
  segar -> tidak_segar -> busuk (meniru proses pembusukan asli).

PENTING: gambar ini BUKAN untuk melatih model yang benar-benar akurat,
hanya untuk memastikan pipeline (loading, resize, augmentasi, split,
training loop, evaluasi) berjalan tanpa error sebelum dataset asli siap.

Jalankan:
    python -m src.generate_dummy_data
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from src.config import (
    ANGLE_LABELS,
    CLASS_NAMES,
    DUMMY_DATA_DIR,
    DUMMY_NUM_DAYS,
    DUMMY_TOMATO_IDS,
    IMAGE_SIZE,
    RANDOM_SEED,
)

# Kanvas dibuat lebih besar dari IMAGE_SIZE supaya mirip foto asli
# (nanti di-resize lagi saat preprocessing), bukan langsung pas ukuran model.
CANVAS_SIZE = (400, 400)


def _label_for_day(day_index: int, total_days: int) -> str:
    """
    Menentukan label berdasarkan posisi hari relatif terhadap total hari
    pengamatan tomat tsb, supaya progresi kesegaran terlihat masuk akal:
    - 1/3 hari pertama  -> segar
    - 1/3 hari tengah   -> tidak_segar
    - 1/3 hari terakhir -> busuk
    """
    fraction = day_index / max(total_days - 1, 1)
    if fraction < 1 / 3:
        return "segar"
    elif fraction < 2 / 3:
        return "tidak_segar"
    else:
        return "busuk"


def _base_color_for_label(label: str) -> tuple:
    """Warna dasar (RGB) sesuai tingkat kesegaran."""
    if label == "segar":
        return (200 + random.randint(-15, 15), 30 + random.randint(-10, 10), 25 + random.randint(-10, 10))
    elif label == "tidak_segar":
        return (150 + random.randint(-15, 15), 70 + random.randint(-15, 15), 40 + random.randint(-10, 10))
    else:  # busuk
        return (60 + random.randint(-15, 15), 35 + random.randint(-10, 10), 25 + random.randint(-10, 10))


def _generate_single_image(label: str, angle: str) -> Image.Image:
    """
    Membuat satu gambar dummy berbentuk blob menyerupai tomat,
    dengan warna & tekstur (noise, bercak) sesuai label.
    Sudut foto (angle) hanya mempengaruhi posisi/ukuran blob secara acak
    kecil, karena ini cuma dummy -- bukan render 3D sungguhan.
    """
    img = Image.new("RGB", CANVAS_SIZE, (245, 245, 240))  # background meja/putih
    draw = ImageDraw.Draw(img)

    base_color = _base_color_for_label(label)

    # Variasi posisi & ukuran blob per sudut, supaya tiap sudut tidak
    # identik persis (meniru perspektif foto berbeda).
    cx, cy = CANVAS_SIZE[0] // 2, CANVAS_SIZE[1] // 2
    jitter = 20
    cx += random.randint(-jitter, jitter)
    cy += random.randint(-jitter, jitter)
    radius = random.randint(120, 150)
    if angle in ("atas", "bawah"):
        radius = int(radius * 0.9)  # tampak sedikit lebih kecil dari atas/bawah

    bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
    draw.ellipse(bbox, fill=base_color)

    # Tambahkan bercak gelap acak untuk label tidak_segar & busuk
    if label in ("tidak_segar", "busuk"):
        n_spots = random.randint(3, 6) if label == "tidak_segar" else random.randint(8, 15)
        for _ in range(n_spots):
            sx = cx + random.randint(-radius, radius)
            sy = cy + random.randint(-radius, radius)
            # hanya gambar bercak jika masih di dalam area blob (approx)
            if (sx - cx) ** 2 + (sy - cy) ** 2 <= radius ** 2:
                spot_r = random.randint(5, 20)
                dark = tuple(max(c - random.randint(40, 90), 0) for c in base_color)
                draw.ellipse([sx - spot_r, sy - spot_r, sx + spot_r, sy + spot_r], fill=dark)

    # Highlight/kilau kecil untuk kesan mulus di tomat segar
    if label == "segar":
        hx, hy = cx - radius // 3, cy - radius // 3
        hr = radius // 6
        highlight = tuple(min(c + 60, 255) for c in base_color)
        draw.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=highlight)

    # Noise acak (mensimulasikan tekstur permukaan & noise sensor kamera)
    arr = np.array(img).astype(np.int16)
    noise_level = {"segar": 5, "tidak_segar": 10, "busuk": 18}[label]
    noise = np.random.randint(-noise_level, noise_level + 1, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # Sedikit blur supaya tidak terlalu "vektor"/tajam sempurna
    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))

    return img


def generate_dummy_dataset(output_dir=DUMMY_DATA_DIR, tomato_ids=None, num_days=DUMMY_NUM_DAYS, seed=RANDOM_SEED, clean=False):
    """
    Generate seluruh dataset dummy: untuk tiap ID tomat x tiap hari x tiap
    sudut, buat satu gambar dan simpan dengan nama sesuai konvensi:
        [ID_TOMAT]_[HARI]_[SUDUT]_[LABEL].jpg

    Args:
        output_dir: folder tujuan (default: DUMMY_DATA_DIR dari config)
        tomato_ids: list ID tomat dummy (default: DUMMY_TOMATO_IDS)
        num_days: jumlah hari simulasi per tomat (default: DUMMY_NUM_DAYS)
        seed: random seed supaya hasil reproducible
        clean: legacy argument; destructive regeneration is no longer supported
    """
    tomato_ids = tomato_ids or DUMMY_TOMATO_IDS
    random.seed(seed)
    np.random.seed(seed)

    if clean:
        raise ValueError("Destructive regeneration disabled; choose a new empty directory")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Use a new empty dummy directory; existing data is preserved")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_images = 0
    for tomato_id in tomato_ids:
        for day_index in range(num_days):
            day_str = f"D{day_index + 1:02d}"
            label = _label_for_day(day_index, num_days)
            for angle in ANGLE_LABELS:
                img = _generate_single_image(label, angle)
                filename = f"{tomato_id}_{day_str}_{angle}_{label}.jpg"
                img.save(output_dir / filename, quality=90)
                total_images += 1

    print(f"[generate_dummy_data] Selesai. {total_images} gambar dummy dibuat di: {output_dir}")
    print(f"[generate_dummy_data] ID tomat dummy: {tomato_ids}")
    print(f"[generate_dummy_data] Jumlah hari per tomat: {num_days}")
    print(f"[generate_dummy_data] Kelas: {CLASS_NAMES}")
    return total_images


if __name__ == "__main__":
    generate_dummy_dataset()
