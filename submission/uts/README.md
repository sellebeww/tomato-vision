# UTS IS794 — dataset primer dan EDA

Berkas yang dikumpulkan: `IS794_UTS_00000138850_Vassel_Goleyu_TomatoVision.pdf`
(satu-satunya format laporan; teksnya dibuat oleh `scripts/uts_pdf.py` dari file audit di folder ini).
Kode: `../../scripts/uts_audit.py`, `../../scripts/uts_pdf.py`, `../../scripts/uts_execute_notebook.py`.
Notebook: `../../notebooks/IS794_UTS_Dataset_EDA.ipynb` (hanya kode dan keluaran; interpretasi ada di PDF).

```bash
venv/bin/python -m scripts.uts_audit              # inventaris, dataset bersih, statistik, 10 grafik
venv/bin/python -m scripts.uts_execute_notebook   # menjalankan sel notebook berurutan
venv/bin/python -m scripts.uts_pdf                # membangun PDF
```

Semua angka dan grafik berasal dari file aktual. Foto sumber, model, dan laporan proyek akhir
tidak diubah. Statistik piksel dan contoh memakai training; test hanya dibaca untuk audit
struktur/checksum, tidak dievaluasi model. Sel notebook dijalankan berurutan lewat executor di
atas, bukan lewat kernel Jupyter (peluncuran kernel dibatasi kebijakan port localhost di
lingkungan ini). Dependensi: `requirements-uts.txt`.

## Dataset (229 foto milik sendiri)

229 foto unik, seluruhnya berlabel: 60 PNG `TOM001`–`TOM060`, 22 JPEG `tomat_ref_*` (semuanya segar), dan 147 PNG
(`own_061`–`own_207`). Asal semua foto adalah pernyataan pemilik (foto sendiri; 14 Okt 2026); tidak ada EXIF.
Perangkat iPhone 13 dicatat untuk 82 foto pertama; perangkat, tanggal, dan ID buah foto lain tidak tercatat.
Proses pelabelan: label 22 JPEG dari inspeksi visual (`visual_review_22.csv`); label
`own_061`–`own_207` dari nama berkas lama dan belum diperiksa per foto.
Split per grup (test terkunci dari eksperimen sebelumnya): train 193, val 30, test 6.
Kondisi fisik dan ID buah belum terverifikasi.

## Berkas

- `clean_dataset_128.npz`: tensor 229 × 128 × 128 × 3 (float32, [0,1]), `labels`, `splits`, `label_origin`
  (asal anotasi). Urutan mengikuti `clean_manifest.csv`. Label 0=segar, 1=tidak_segar, 2=busuk.
- `clean_manifest.csv`, `frozen_manifest.csv`: indeks, sumber, label, grup, split, checksum.
- `audit_summary.json`, `descriptive_*.csv`, `training_metrics.csv`, `training_dhash_pairs.csv`,
  `manifest_near_pairs.csv`: hasil audit dan statistik. `manifest_near_pairs.csv` memakai hash dari pipeline;
  `training_dhash_pairs.csv` memakai thumbnail pada kode audit, sehingga jumlah pasangan dekat dapat berbeda.
- `inventory_*.csv`, `duplicates_*.csv`: inventaris berkas dan salinan identik (4 salinan JPEG).
- `source_confirmation.json`, `consolidation_map.csv`, `visual_review_22.csv`, `review_22_JPEG.*`:
  catatan asal dan tinjauan 22 JPEG.
- `figures/01.png`–`10.png`: grafik EDA.

```python
import numpy as np
data = np.load('submission/uts/clean_dataset_128.npz', allow_pickle=False)
train = (data['splits'] == 'train')   # 193 foto
x_train, y_train = data['images'][train], data['labels'][train]
```

Pembagian ini tidak membuktikan independensi buah. Label belum diverifikasi fisik, jadi bukan ground truth.

## Masih diperlukan

- Tinjauan penulis terhadap label (terutama label dari nama berkas dan 22 JPEG) dan pemeriksaan fisik bila masih mungkin.
- ID buah antarfile bila masih diketahui; jika tidak, keterbatasan tetap dinyatakan.
- Pembacaan ulang dan penyesuaian PDF dengan kata-kata sendiri; kelas belum diberikan.
