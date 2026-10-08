# Klasifikasi Tingkat Kesegaran Tomat Menggunakan Convolutional Neural Network (CNN) Berbasis TensorFlow

## Ringkasan

Proyek ini mengimplementasikan klasifikasi kondisi visual satu buah tomat menjadi
segar, tidak segar, dan busuk menggunakan CNN yang dilatih dari nol dengan
TensorFlow/Keras. Sistem mencakup inventarisasi sumber proyek, anotasi, pemisahan
data berdasarkan identitas buah, augmentasi training, pembandingan model,
evaluasi, dan aplikasi prediksi lokal.

Implementasi perangkat lunak telah dijalankan. Dataset berisi 82 foto unik yang
seluruhnya berlabel: 60 foto lama (20 per kelas) dan 22 foto primer impor yang pada
9 Oktober 2026 digabung dengan label `segar` atas arahan pemilik. Angka evaluasi
harus dibaca bersama jumlah kelompok validation/test yang kecil.

## Tujuan dan batas masalah

Masukan berupa foto satu tomat yang terlihat jelas. Keluaran berupa kelas,
probabilitas ketiga kelas, dan penanda perlunya tinjauan manual. Kesegaran tidak
disamakan dengan kematangan: warna hijau/merah saja bukan dasar penentuan kelas.
Sistem tidak menilai keamanan pangan, kerusakan internal, objek non-tomat, atau
banyak buah sekaligus. Klaim generalisasi memerlukan test foto nyata independen.

## Dataset

Folder `data/reference_import/` memuat 26 JPEG foto primer, 22 di antaranya unik berdasarkan
SHA-256. Salinan duplikat dipertahankan namun tidak menjadi sampel independen.
Ke-22 foto unik digabung ke dataset berlabel sebagai `segar` (label dari pemilik,
tidak diperiksa satu per satu); semuanya satu grup `R_reference_meja_kayu` sehingga
berpindah split bersama dan masuk training.
Berkas diberi nama seragam `tomat_ref_01.jpeg` sampai `tomat_ref_22.jpeg`; salinan identik
memakai akhiran `_salinan1` (misalnya `tomat_ref_06_salinan1.jpeg`).
Folder `data/raw/` memuat 60 sumber berlabel, 20 per kelas, dengan 11 kelompok sesi pada split yang telah diaudit. Asal pemotretan sumber lama perlu bukti terpisah.
Dataset publik dan dummy berbentuk lingkaran tidak dipakai oleh pipeline.

Anotasi lama ditandai disetujui pada inventaris; 22 foto impor disetujui pada 9 Oktober 2026. `group_id` menggabungkan ID buah dan sesi secara konservatif: seluruh hari dan sudut satu buah berada di split yang sama.
Pemeriksaan memakai hash file, hash piksel, dan pengelompokan konservatif dHash.
dHash tidak dapat membuktikan identitas buah; kualitas metadata tetap penting.
Proporsi aktual adalah 70 training / 6 validation / 6 test terkunci (manifest `data/prepared/ff822adef8db3ae5`); test terkunci dan validation tidak berubah dari sebelum penggabungan, dan grup sesi membuat proporsi berbeda dari target 70/15/15.
Training berisi segar 38, tidak_segar 16, busuk 16 foto asli dari 8 grup. Kelas segar berlebih dan 22 fotonya satu latar, sehingga ada risiko model mengaitkan latar dengan kelas.
Setiap split wajib mencakup ketiga kelas. Satu atau dua buah tidak dipisah secara
paksa menjadi train/validation/test.

Validation dan test tidak diaugmentasi.

## Preprocessing dan augmentasi

Loader yang sama digunakan ketika training, evaluasi, dan inferensi: koreksi
orientasi EXIF, konversi RGB, resize dengan mempertahankan aspek rasio melalui
letterbox, lalu normalisasi piksel ke rentang [0,1]. Ukuran eksperimen 128×128.
Gambar animasi, terlalu besar, atau format tidak didukung ditolak.

CNN regularized memakai flip horizontal, rotasi kecil, translasi, zoom, variasi
kontras, dan brightness hanya saat training. Warna tidak diubah ekstrem agar
kondisi permukaan tidak terdistorsi. Augmentasi dinonaktifkan pada evaluasi dan
inferensi. Sampling per epoch seimbang antar kelas, tetapi pengulangan sampel
tetap dihitung sebagai pengulangan, bukan data baru.

## CNN dari nol

Semua bobot diinisialisasi acak. Tidak ada backbone pretrained atau transfer
learning. Konvolusi belajar pola lokal dari data training; pooling/striding
mengurangi resolusi fitur, dan softmax mengubah output menjadi tiga probabilitas.

Baseline mempertahankan topologi empat blok Conv2D (32/64/128/256 filter),
Batch Normalization, ReLU, Max Pooling, Global Average Pooling, Dense 128,
dropout, dan output tiga kelas. Parameter total: 423.619.

CNN regularized menggunakan:

1. Augmentasi training-only, kemudian Conv2D 24 filter dengan stride 2.
2. Group Normalization dan ReLU.
3. Tiga blok residual separable convolution dengan 48, 96, dan 160 filter;
   setiap blok menurunkan resolusi dengan stride 2, memiliki shortcut, normalisasi,
   dan spatial dropout.
4. Global Average Pooling, Dense 64, dropout, dan Dense softmax tiga kelas.

Parameter total regularized: 96.019. Separable convolution tetap merupakan
konvolusi yang bobotnya dilatih dari nol. Group Normalization tidak memakai
statistik rata-rata batch berjalan seperti Batch Normalization. Pemilihan ini
diuji pada data kecil; tidak dianggap pasti terbaik untuk semua data.
Implementasi ada di [model.py](../src/model.py).

## Pelatihan dan pemilihan model

Optimizer Adam dan sparse categorical cross-entropy digunakan. Seed 42 dan
operasi deterministik diaktifkan. Konfigurasi tersimpan pada setiap run.
Early stopping memantau validation loss dengan patience 10; learning rate
diturunkan setelah tiga epoch tanpa perbaikan. Model terbaik dipulihkan.

Eksperimen akhir membandingkan baseline, regularized LR 0,001/dropout 0,35,
dan regularized LR 0,0003/dropout 0,50. Pemilihan kandidat memakai validation
macro-F1 dengan tie-break validation loss. Baseline dan pemenang kemudian
dievaluasi pada test sama. Hyperparameter tidak diubah berdasarkan hasil test.
Eksperimen historis memasang temperature scaling menggunakan validation saja;
ukuran kecil tidak cukup untuk menyatakan probabilitas terkalibrasi dengan baik.
Run baru mensyaratkan sedikitnya 20 gambar validation per kelas dan lima kelompok
untuk fitting temperature; jika tidak, softmax mentah dipertahankan. Ini guard
konservatif, bukan jaminan statistik.

Setiap run menyimpan model Keras, konfigurasi, snapshot source, versi runtime,
manifest beku, fingerprint, history, dan grafik. Seed tidak menjamin kesamaan
bit-per-bit lintas perangkat/versi. File sumber dataset dan hash harus dipertahankan.

## Hasil

Hasil eksperimen disimpan pada `outputs/experiments/<run>/` dan ditampilkan pada
menu Evaluasi. Laporkan accuracy, macro-F1, confusion matrix, jumlah kelompok
test, dan contoh kesalahan. Jangan menjalankan eksperimen berulang hanya untuk
menaikkan angka test; jangan menilai model dari training accuracy saja.

Pelatihan ulang setelah penggabungan 22 foto primer (9 Oktober 2026; ringkasan di
`outputs/experiments/own_v4_ref22/summary.md`, test tidak dievaluasi):

| Kandidat | Accuracy train | Accuracy validation | Macro-F1 validation | Loss validation |
|---|---:|---:|---:|---:|
| regularized | 75,7% | 66,7% (4/6) | 0,556 | 0,505 |
| baseline_augmented | 98,6% | 100% (6/6) | 1,000 | 0,100 |

Menurut aturan pemilihan (macro-F1 validation, lalu loss) kandidat terpilih adalah
baseline_augmented. Angka ini berasal dari enam foto validation (dua sesi) yang juga
memandu early stopping; selisih 4/6 menjadi 6/6 hanya dua foto dan **bukan bukti
peningkatan**. Regularized memprediksi kedua foto segar validation sebagai tidak_segar.
Hasil own_v3 sebelumnya (kedua kandidat 4/6, macro-F1 0,656) tidak diganti oleh angka ini
sebagai klaim akurasi; model demo tidak diganti dan belum siap produksi.

## Aplikasi dan verifikasi

Empat menu aplikasi adalah Prediksi, Dataset & label, Training, dan Evaluasi.
Prediksi batch dapat diunduh. Impor foto membutuhkan konfirmasi sumber sendiri.
Penyimpanan anotasi memiliki revisi untuk menolak perubahan tab usang dan backup
untuk pemulihan. Training hanya berjalan jika pemeriksaan data lolos; proses
mendukung log dan penghentian tanpa menghapus hasil sebelumnya.

Server bersifat lokal, memeriksa Host dan token sesi untuk perubahan data,
membatasi ukuran upload, serta menerapkan Content Security Policy. Ini bukan
layanan internet publik; deployment publik memerlukan autentikasi, pengelolaan
akses, dan peninjauan keamanan tersendiri.

Tes otomatis meliputi leakage, format gambar, label/index, sampling, temperatur,
serialisasi model, impor, revisi anotasi, kesiapan data, dan pengelolaan training.
Tes HTTP dan browser memeriksa alur pengguna, termasuk desktop dan ponsel.

## Kesimpulan dan langkah lanjut

Perangkat lunak dapat dijalankan dan didemokan sebagai prototipe end-to-end.
Untuk memperkuat hasil, perlu verifikasi kondisi tiap foto dari 22 foto yang kini
berlabel `segar`, penambahan buah dan latar independen terutama untuk kelas
tidak_segar dan busuk, lalu training dan evaluasi ulang. Tahap itu tidak dapat
digantikan dengan label/ID tebakan atau jumlah augmentasi yang besar.


## Laporan pengumpulan terbaru

Dokumen Word lengkap berada di [submission/IS794_Laporan_TomatoVision.docx](../submission/IS794_Laporan_TomatoVision.docx). Laporan tersebut menggantikan hasil historis own_v1 yang terpengaruh kebocoran sesi. Hasil own_v3 yang dipakai adalah validation accuracy 4/6 (66,7%) dan macro-F1 0,656. Notebook proyek ada di `notebooks/IS794_TomatoVision_Project.ipynb`.

Evaluasi prediksi selektif (`src/selective_evaluation.py`) membandingkan coverage dan proporsi kesalahan pada prediksi yang diterima. Pada enam validation, aturan review menerima empat gambar (3/4 benar), menahan satu kesalahan, tetapi satu kesalahan tetap lolos. Ini bukan peningkatan accuracy keseluruhan atau hasil test independen. Rancangan pembeda dan latihan presentasi ada di `submission/audit/PEMBEDA_DAN_PRESENTASI.md`.
