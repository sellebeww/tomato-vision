# Tomato Vision — Dataset

Satu paket, satu tabel induk. Seluruh 229 gambar dicatat di `labels.csv`.

| Sumber | Jumlah | Awalan nama | Label | Pemakaian |
|---|---|---|---|---|
| `own` — foto tomat asli (`own_`) | 60 (20 per kelas) | `own_` | disetujui pemilik | train / val / test |
| `synthetic_ai` — hasil AI | 147 (49 per kelas) | `syn_` | menurut prompt, belum ditinjau manusia | hanya train, setelah ditinjau |
| `own` — foto primer impor (`ref_`) | 22 | `ref_` | `segar` (diberikan pemilik 9 Okt 2026) | train, satu grup |

Semua foto ada di satu folder `images/`, tanpa subfolder. Sumber dan label dibaca dari nama berkas atau `labels.csv`.

Kelas: `segar`, `tidak_segar`, `busuk`.

## Penamaan

Satu pola untuk semua berkas: huruf kecil, nomor tiga digit, label di akhir.

```
own_001_segar.png          foto asli
syn_001_segar.png          gambar sintetis (AI)
ref_001_segar.jpeg        foto primer impor (JPEG), label dari pemilik
```

Kolom `id` (`own_001`, `syn_001`, `ref_001`) sama dengan awal nama berkas dan unik.
Nama lama tetap tercatat di kolom `original_name` (`TOM001_segar.png`,
`SYN001_segar.png`, `tomat_ref_01.jpeg`; berkas sekarang `ref_001_segar.jpeg`), dan nomornya tidak berubah.

## Isi paket

```
labels.csv     tabel induk, satu baris per gambar
prompts.jsonl  rencana prompt gambar sintetis (kunci: id)
report.json    jumlah per sumber/kelas/split dan pemeriksaan integritas
images/        229 foto, semua di satu folder
TomatoVision_Dataset.zip   seluruh isi di atas dalam satu berkas
```

### Kolom `labels.csv`

| Kolom | Arti |
|---|---|
| `id`, `file` | pengenal unik dan path relatif gambar |
| `label` | `segar` / `tidak_segar` / `busuk`; kosong bila belum dikonfirmasi |
| `source` | `own` atau `synthetic_ai` |
| `split` | `train`, `val`, `test`, atau `unassigned` |
| `group_id` | sesi/adegan; foto satu grup tidak boleh terpecah ke split berbeda |
| `approved` | `True` bila label sudah disetujui pemilik |
| `review_status` | `approved`, `pending_review`, atau `unlabeled` |
| `width`, `height`, `sha256` | ukuran piksel dan hash berkas |
| `original_name` | nama berkas sebelum penamaan ulang |
| `background` … `focus_condition` | kondisi yang diminta pada prompt (hanya `synthetic_ai`) |
| `suggested_label`, `model_prediction`, `model_confidence`, `needs_review`, `notes` | tinjauan foto impor (hanya `unlabeled`) |

## Aturan pemakaian

- Validation dan test hanya berisi foto `own` (masing-masing 6 gambar). Gambar
  `synthetic_ai` tidak boleh masuk validation/test dan tidak boleh diubah menjadi `source=own`.
- `split=train` pada gambar sintetis adalah alokasi calon pemakaian setelah tinjauan,
  bukan izin ingest otomatis. Pipeline hanya memakainya dengan `--with-synthetic`.
- Gambar sintetis dibuat dengan built-in imagegen, bukan bukti pemotretan buah fisik.
  Labelnya mengikuti prompt dan pemeriksaan visual asisten (`approved=False`).
- Rencana prompt berisi 218 gambar sintetis, baru 147 tersimpan (`syn_001`–`syn_147`).
  `syn_148`–`syn_218` belum dibuat; jangan menyebut jumlah rencana sebagai jumlah aktual.
  Di `prompts.jsonl`, kolom `saved` menandai yang sudah ada.
- Nilai `background`…`focus_condition` adalah instruksi prompt, bukan hasil pengukuran.
  `not_recorded_in_earlier_batch` berarti atribut belum dicatat, bukan jaminan kondisinya tidak ada.
- Dataset ini belum dilatih dan belum membuktikan peningkatan akurasi.

## Meninjau label

- **Segar:** kulit utuh dan tegang; tetes air, pantulan, dan warna oranye tidak dengan sendirinya menunjukkan kerusakan.
- **Tidak segar:** keriput atau penyusutan terlihat, tanpa ciri busuk yang jelas.
- **Busuk:** bercak kerusakan membusuk, jaringan rusak, atau jamur tampak.
- Bila blur, jarak, bayangan, atau air menutupi ciri pembeda, tandai untuk ditinjau. Jangan
  menyetujui label hanya karena nama berkas atau prediksi model.

Untuk menguji manfaat gambar sintetis, bandingkan model yang dilatih pada foto `own` saja
dengan model yang ditambah gambar sintetis yang sudah ditinjau, memakai split sumber dan
anggaran training yang sama.

## Yang tidak ada di paket ini

- `data/raw/` memuat foto asli yang sama (nama kerja `TOM###_label.png`) untuk pipeline training.
- `data/dummy/` adalah gambar placeholder hasil `src/generate_dummy_data.py`, bukan data sungguhan.
- `data/augmented/` adalah ekspor augmentasi turunan, bisa dibuat ulang.
- `data/external/` adalah dataset publik FGrade, terpisah dari dataset ini.
