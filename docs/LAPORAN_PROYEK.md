# Klasifikasi Tingkat Kesegaran Tomat Menggunakan Convolutional Neural Network (CNN) Berbasis TensorFlow

## Ringkasan

Proyek ini mengimplementasikan klasifikasi kondisi visual satu buah tomat menjadi
segar, tidak segar, dan busuk menggunakan CNN yang dilatih dari nol dengan
TensorFlow/Keras. Sistem mencakup inventarisasi foto sendiri, anotasi, pemisahan
data berdasarkan identitas buah, augmentasi training, pembandingan model,
evaluasi, dan aplikasi prediksi lokal.

Implementasi perangkat lunak telah dijalankan. Dataset berisi 82 foto unik; 60
foto sudah berlabel dan dikelompokkan, sedangkan 22 foto lain belum dikonfirmasi.
Angka evaluasi harus dibaca bersama jumlah kelompok test yang kecil.

## Tujuan dan batas masalah

Masukan berupa foto satu tomat yang terlihat jelas. Keluaran berupa kelas,
probabilitas ketiga kelas, dan penanda perlunya tinjauan manual. Kesegaran tidak
disamakan dengan kematangan: warna hijau/merah saja bukan dasar penentuan kelas.
Sistem tidak menilai keamanan pangan, kerusakan internal, objek non-tomat, atau
banyak buah sekaligus. Klaim generalisasi memerlukan test foto nyata independen.

## Dataset

Folder `data/reference_import/` memuat 26 JPEG, 22 di antaranya unik berdasarkan
SHA-256. Salinan duplikat dipertahankan namun tidak menjadi sampel independen.
Folder `data/raw/` memuat 60 foto berlabel, 20 per kelas, dalam 10 kelompok.
Dataset publik dan dummy berbentuk lingkaran tidak dipakai oleh pipeline.

Label foto sendiri disetujui berdasarkan pengamatan pemilik. `group_id` adalah
identitas buah fisik: seluruh hari dan sudut satu buah berada di split yang sama.
Pemeriksaan memakai hash file, hash piksel, dan pengelompokan konservatif dHash.
dHash tidak dapat membuktikan identitas buah; kualitas metadata tetap penting.
Target proporsi split 70/15/15, tetapi kelompok membuat proporsi aktual berbeda.
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
Untuk memperkuat hasil, perlu konfirmasi label/ID buah pada 22 foto yang belum
dilabeli, penambahan buah independen, lalu training dan evaluasi ulang. Tahap itu
tidak dapat digantikan dengan label/ID tebakan atau jumlah augmentasi yang besar.
