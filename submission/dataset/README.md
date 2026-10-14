# Tomato Vision — Dataset

Satu paket, satu tabel induk. Seluruh 229 gambar dicatat di `labels.csv`.

Seluruh 229 gambar adalah **foto milik pemilik, diambil sendiri** (pernyataan pemilik,
14 Okt 2026) dan semuanya berlabel (`source=own`, `label_status=approved`): segar 91,
tidak_segar 69, busuk 69.

| Awalan nama | Jumlah |
|---|---|
| `own_001`–`own_207` | 207 PNG |
| `ref_001`–`ref_022` | 22 JPEG |

Semua foto ada di satu folder `images/`, tanpa subfolder. Sumber dan label dibaca dari nama berkas atau `labels.csv`.

Kelas: `segar`, `tidak_segar`, `busuk`.

## Penamaan

Satu pola untuk semua berkas: huruf kecil, nomor tiga digit, label di akhir.

```
own_001_segar.png          foto asli berformat PNG
ref_001_segar.jpeg         foto asli berformat JPEG
```

Kolom `id` (`own_001`, `own_061`, `ref_001`) sama dengan awal nama berkas dan unik.

## Isi paket

```
labels.csv     tabel induk, satu baris per gambar
report.json    jumlah per sumber/kelas/split dan pemeriksaan integritas
images/        229 foto, semua di satu folder
TomatoVision_Dataset.zip   seluruh isi di atas dalam satu berkas
```

### Kolom `labels.csv`

| Kolom | Arti |
|---|---|
| `id`, `file` | pengenal unik dan path relatif gambar |
| `label` | `segar` / `tidak_segar` / `busuk`; kosong bila belum dikonfirmasi |
| `source` | `own` (foto asli) |
| `label_status` | `approved` (semua 229) |
| `source_note` | catatan asal foto dan asal label |
| `split` | `train`, `val`, `test`, atau `unassigned` |
| `group_id` | sesi/adegan; foto satu grup tidak boleh terpecah ke split berbeda |
| `approved` | `True` bila label sudah disetujui pemilik |
| `review_status` | `approved`, `pending_review`, atau `unlabeled` |
| `width`, `height`, `sha256` | ukuran piksel dan hash berkas |
| `suggested_label`, `model_prediction`, `model_confidence`, `needs_review`, `notes` | tinjauan foto impor (hanya `unlabeled`) |

## Aturan pemakaian

- Pipeline memuat seluruh 229 foto (`python -m src.manifest`) dari `data/raw/` dan `data/annotations.csv`;
  tidak ada opsi khusus.
- Split dibuat per grup: train 193, val 30, test 6 (test terkunci dari eksperimen sebelumnya).
  Kolom `split` dan `group_id` di `labels.csv` disalin dari bundle itu.
- Foto tidak memiliki EXIF, sehingga asal pemotretan tidak dapat dibuktikan dari metadata; perangkat,
  tanggal, dan ID buah tidak tercatat per berkas.
- Proses pelabelan: label `own_061`–`own_207` berasal dari nama berkas dan belum
  diperiksa per foto; label 22 JPEG berasal dari inspeksi visual. Tidak
  ada klaim verifikasi fisik atau penilai kedua.
- Hasil model yang dilaporkan sebelumnya (`own_v1`–`own_v4_ref22`) memakai 82 foto; model belum dilatih
  ulang dengan 229 foto.

## Meninjau label

- **Segar:** kulit utuh dan tegang; tetes air, pantulan, dan warna oranye tidak dengan sendirinya menunjukkan kerusakan.
- **Tidak segar:** keriput atau penyusutan terlihat, tanpa ciri busuk yang jelas.
- **Busuk:** bercak kerusakan membusuk, jaringan rusak, atau jamur tampak.
- Bila blur, jarak, bayangan, atau air menutupi ciri pembeda, tandai untuk ditinjau. Jangan
  menyetujui label hanya karena nama berkas atau prediksi model.

Untuk menguji manfaat 147 foto tambahan, bandingkan model yang dilatih pada 82 foto pertama dengan
model yang dilatih pada 229 foto, memakai validation dan anggaran training yang sama.

## Yang tidak ada di paket ini

- `data/raw/` memuat foto asli yang sama (nama kerja `TOM###_label.png`) untuk pipeline training.
- `data/augmented/` adalah ekspor augmentasi turunan, bisa dibuat ulang.
- `data/external/` adalah dataset publik FGrade, terpisah dari dataset ini.
