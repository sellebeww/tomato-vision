# Dataset card dan protokol pengumpulan

## Sumber dan tujuan

Target adalah kondisi visual satu buah tomat, bukan varietas, tingkat kematangan,
atau keamanan konsumsi. Tomat hijau tidak otomatis tidak segar/busuk.
Data publik dan dummy lama tidak digunakan.

- Folder `data/reference_import/`: 26 file JPEG, 22 unik; empat salinan identik
  dikecualikan tanpa dihapus. Label dan ID buahnya belum dikonfirmasi.
  Nama berkas: `tomat_ref_01`–`tomat_ref_22`; salinan identik berakhiran `_salinan1`.
- Folder `data/raw/`: 60 foto PNG berlabel, 20 per kelas, dalam 10 kelompok
  (`set01`–`set10`).

Foto yang berasal dari buah sama atau skenario sama berbagi satu `group_id`.
Jumlah foto bukan jumlah buah independen, dan augmentasi tidak menambah jumlah
buah independen.

## Kriteria label

| Kelas | Kriteria pengamatan | Hal yang tidak cukup untuk menentukan label |
|---|---|---|
| segar | Kulit relatif kencang, bentuk utuh, tanpa pembusukan tampak | Warna merah saja |
| tidak_segar | Keriput, layu, kehilangan kekencangan tanpa pembusukan nyata | Usia/hari saja |
| busuk | Pembusukan/lesi jaringan atau jamur yang jelas | Warna gelap akibat bayangan |

Foto ambigu: jangan setujui. Catat pengamatan nyata dan diskusikan kriteria
dengan pihak yang berwenang. Foto saja tidak selalu dapat membedakan tekstur/kerusakan
internal. Jangan memaksa semua tahap berdasarkan urutan hari. Sebaiknya dua orang
menilai sebagian foto secara terpisah dan menyelesaikan perbedaan sebelum split dibekukan.

## ID buah dan independensi

Satu buah fisik memakai satu `group_id` sepanjang pengamatan, termasuk ketika
berubah kelas. Semua sudut/hari buah yang sama harus berada pada split yang sama.
Jika identitas tidak dapat dipastikan, kelompokkan secara konservatif; jangan
mengarang ID berbeda. Kemiripan dHash hanya alat bantu, bukan bukti identitas.

Pipeline memeriksa SHA-256, hash piksel, dan dHash 64-bit (jarak <=4), lalu
memisahkan kelompok secara deterministik. Pembagian menargetkan 70/15/15;
jumlah foto dapat berbeda karena ukuran kelompok. Setiap split wajib memuat
semua kelas. Hanya 1–2 buah tidak cukup untuk tiga split independen.

## Pengumpulan berikutnya

Prioritaskan lebih banyak **buah dan sesi pemotretan independen**, bukan banyak
foto hampir sama. Sebagai target perencanaan awal, kumpulkan 30–50 buah yang
beragam dan ratusan foto seimbang, lalu nilai kurva belajar dan error.
Ini bukan ambang yang menjamin akurasi.

Gunakan latar, pencahayaan, kamera, jarak, sudut, bentuk, ukuran, dan varietas
yang bervariasi pada SEMUA kelas. Jangan memotret semua tomat busuk di satu
latar dan semua segar di latar lain. Pertahankan buah utuh terlihat jelas.
Sediakan sesi baru yang tidak dipakai merancang model sebagai uji eksternal.

Model tidak punya kelas non-tomat; gambar benda lain bisa tetap diberi
probabilitas tinggi. Jangan memakai demo sebagai detektor umum.

## Kapan layak dilaporkan sebagai hasil?

Label dikonfirmasi; kelompok independen; seluruh kelas terwakili;
hasil test beserta confusion matrix, macro-F1, jumlah kelompok, dan
contoh kesalahan dilaporkan; perbedaan train/validation diperiksa. Uji brightness
dan blur hanya pemeriksaan awal robustness, bukan sertifikasi.
Kalibrasi hanya menggunakan validation. Run baru mempertahankan softmax mentah
jika validation kurang dari 20 gambar per kelas atau lima kelompok.

Gunakan aplikasi lokal untuk melengkapi `data/annotations.csv`, kemudian jalankan
pipeline. Jangan memasukkan augmentasi ke validation/test.

## Ekspansi dataset 7 Oktober 2026

Dataset lokal diperluas dengan **360 gambar augmentasi (120 per kelas)** di
`data/augmented/own_v3_360/`. Inventaris gabungannya ada di
`data/augmented/own_v3_360/annotations.csv`: **420 gambar**, terdiri dari 60 foto
asli berlabel dan 360 variasi training. Sebanyak 22 foto yang belum disetujui
tetap berada di inventaris asli dan tidak digunakan. Augmentasi tidak menambah
jumlah buah atau sesi independen dan tidak membuktikan peningkatan akurasi.

Kesalahan grup `set01`–`set10` pada 60 anotasi asli diperbaiki mengikuti audit
`data/sessions.csv` (11 sesi); anotasi sebelumnya dicadangkan di
`data/annotation_backups/`. Pipeline umum dan pratinjau aplikasi sekarang
menggabungkan ID buah, sesi pemotretan, dan kemiripan gambar sebelum split.
Jika tabel sesi tersedia, setiap foto yang disetujui wajib memiliki sesi.
Enam foto test terkunci di protokol `own_v2` tetap dipertahankan; gambar yang
berbagi grup dengan test menyebabkan pipeline berhenti, bukan memindahkannya.

| Bagian | Foto asli | Variasi hasil generate | Grup sesi |
|---|---:|---:|---:|
| Training | 48 | 360 | 7 |
| Validation | 6 | 0 | 2 |
| Test terkunci | 6 | 0 | 2 |

Variasi menggunakan flip horizontal, rotasi ringan, translasi, zoom, kontras,
dan pencahayaan dari fungsi augmentasi training yang sama. Warna kelas tidak
diganti dan kerusakan buah tidak direkayasa. Label diwarisi dari foto asal.
Manifest menyimpan SHA-256 gambar, hash piksel, parent, label, grup, sesi,
dan `source=augmented`. Ekspor memeriksa keunikan piksel, menulis ke direktori
sementara, dan menolak menimpa output yang sudah ada.

Untuk membuat ekspor baru yang dapat diulang (gunakan folder output baru):

```bash
python -m src.manifest
python -m src.export_augmentation \
  --dataset data/prepared/7d208f30a6b6e8d7 \
  --output data/augmented/ekspansi_baru --per-class 120 --size 224 --seed 42
```

`manifest.csv` pada folder ekspor berisi 360 variasi; `annotations.csv` berisi
inventaris gabungan. `parent_dataset/` menyimpan split asli yang sudah dibekukan.
Jangan split ulang CSV gabungan. Training bawaan membaca `parent_dataset/`
dan menghasilkan variasi baru secara online setiap epoch; tidak perlu
memasukkan file hasil ekspor lagi karena itu akan menggandakan augmentasi.

```bash
python -m src.train --dataset data/augmented/own_v3_360/parent_dataset \
  --run outputs/experiments/latihan_baru --architecture regularized \
  --epochs 20 --samples-per-class 120
```

Run ini menilai validation untuk memilih bobot, tanpa mengevaluasi test atau
mengganti model demo secara otomatis. Hasil validation hanya berasal dari
enam foto dan belum cukup untuk klaim akurasi di kondisi baru.

### Hasil training lokal

Dua kandidat telah dilatih dengan augmentasi online dan split sesi yang sama.
Model `regularized` (20 epoch, bobot terbaik epoch 15) memperoleh **66,7%
akurasi validasi (4/6)**, macro-F1 **0,656**, dan loss **0,538**. Pembanding
`baseline_augmented` berhenti pada epoch 19 (bobot epoch 9), dengan akurasi dan
macro-F1 yang sama, tetapi loss **0,904**. Konfigurasi pembanding disimpan di
`configs/augmented_baseline.json` dan dapat dipakai dengan `src.train --config`.

Kandidat `regularized` dipilih berdasarkan macro-F1 lalu loss validasi; artefak
ada di `outputs/experiments/own_v3_augmented/regularized/`, dan perbandingan
lengkap di `outputs/experiments/own_v3_augmented/summary.md`. Ini eksperimen
eksploratif, bukan bukti peningkatan akurasi independen: validasi hanya dua sesi
dan juga dipakai untuk early stopping. Test terkunci tidak dievaluasi dan model
demo tidak diganti. Penambahan foto dari buah/sesi baru masih diperlukan.

### Paket dataset tunggal (`submission/dataset/`)

Seluruh 229 gambar kini ada dalam satu paket dengan satu tabel induk,
`labels.csv`, dan satu pola nama (huruf kecil, nomor tiga digit, label di
akhir). Rincian kolom, aturan pemakaian, dan panduan meninjau label ada di
`submission/dataset/README.md`.

| Sumber | Jumlah | Nama berkas (semua di `images/`) | Status |
|---|---|---|---|
| foto asli (`source=own`) | 60 | `own_001_segar.png` | disetujui pemilik; train 48, val 6, test 6 |
| gambar AI (`source=synthetic_ai`) | 147 | `syn_001_segar.png` | `approved=False`; hanya calon train setelah tinjauan |
| foto impor | 22 | `ref_001.jpeg` | label kosong, `split=unassigned`; diagnostik saja |

Nomor tidak berubah dari nama lama (`TOM001` → `own_001`, `SYN001` →
`syn_001`, `tomat_ref_01` → `ref_001`); nama lama tercatat di kolom
`original_name`. Foto asli yang sama tetap ada di `data/raw/` dengan nama kerja
`TOM###_label.png` karena pipeline training, `data/annotations.csv`, dan
`data/sessions.csv` memakai nama itu.

Berkas pendukung dalam paket: `prompts.jsonl` (rencana 218 prompt, 147 sudah
tersimpan; kolom `saved` menandainya), `report.json` (jumlah per sumber,
kelas, split, dan pemeriksaan hash), serta `TomatoVision_Dataset.zip`. Atribut
variasi (`background`, `lighting`, dan seterusnya) adalah instruksi prompt, bukan
hasil pengukuran. `syn_148`–`syn_218` belum dibuat, jadi jumlah rencana tidak
boleh disebut sebagai jumlah aktual. Belum ada pengukuran peningkatan akurasi dari
gambar AI.

Pipeline own-only tidak memakai gambar AI secara otomatis; `python -m
src.manifest --with-synthetic` menambahkannya ke train saja, tidak pernah ke
validation/test. Pemeriksaan konsistensi paket:

```bash
python -m scripts.validate_dataset          # nama, hash, split, grup, berkas yatim
python -m unittest tests.test_dataset_package
```

Paket ini dibuat dari tata letak lama oleh `scripts/consolidate_dataset.py`
(migrasi satu kali). Tata letak lama, tiga ZIP lama, dan CSV lama
(`labels_v2_127.csv`, `variation_manifest.csv`, `reference_review.csv`, dan
lainnya) dipindahkan ke `submission/dataset_lama/` dan tidak dihapus.

Folder `data/reference_import/` tetap utuh: 26 file berisi 22 foto unik dan
4 salinan identik. Semua foto unik sudah dipakai untuk diagnostik model
`own_v3_augmented/regularized`; hasilnya 18 prediksi busuk dan 4 tidak segar,
dengan 14 ditandai perlu review. Hasil ini bukan akurasi karena label kondisi
buah belum dikonfirmasi pemilik. Prediksi tidak dijadikan label training.
Inventaris review ada di `data/reference_review.csv` dan paket
(`labels.csv`, kolom `suggested_label`, `model_prediction`, `notes`), sedangkan
contact sheet, prediksi, dan daftar duplikat ada di `outputs/reference_review/`.
Semua foto impor disimpan dalam satu grup sesi konservatif karena identitas
buah antar tanggal belum pasti.

## Penggabungan foto primer 9 Oktober 2026

Atas arahan pemilik, 22 foto unik di `data/reference_import/` digabung ke dataset
berlabel sebagai **segar** (sesuai saran visual; kondisi tiap foto tidak diperiksa
satu per satu oleh pemilik). Semua foto berada di satu grup/sesi
`R_reference_meja_kayu` karena identitas buah antar tanggal tidak pasti, sehingga
berpindah split bersama-sama. Anotasi sebelumnya dicadangkan di
`data/annotation_backups/e15e301bf6d4a65f.csv`.

Manifest baru: `data/prepared/ff822adef8db3ae5` (82 foto asli). Enam foto test
terkunci tidak berubah dan validation tetap 6 foto; seluruh 22 foto baru masuk
training (segar 38, tidak_segar 16, busuk 16; 8 grup). Karena satu latar dan satu
grup, tambahan ini memperberat kelas segar di training tetapi tidak menambah
jumlah buah/sesi independen. Hasil lama (60 sumber, 48/6/6) belum dilatih ulang
dengan data ini.

