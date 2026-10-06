# Menjalankan dan mendemokan Tomato Vision

Judul: **Klasifikasi Tingkat Kesegaran Tomat Menggunakan Convolutional Neural Network (CNN) Berbasis TensorFlow**.

## Jalankan

Dari folder proyek, gunakan Python environment yang sudah dipasang:

```bash
venv/bin/python -m src.check --deep
venv/bin/python -m src.app
```

Jika memakai environment baru, ganti `venv/bin/python` dengan interpreter yang
aktif sesuai README. Buka http://127.0.0.1:7860. Server hanya menerima koneksi
localhost. Port yang terpakai dapat diganti dengan `--port 7861`.

Pemeriksaan `check` membedakan `demo_ready` dan kesiapan data nyata. Demo yang
berjalan bukan bukti generalisasi model. `--require-real` menghasilkan exit code 2
jika data foto sendiri belum memenuhi pemeriksaan teknis.

## Urutan presentasi

1. Jelaskan tiga kelas **kesegaran**, bukan sekadar warna/kematangan. CNN memakai
   bobot acak, bukan pretrained.
2. Di **Prediksi**, pilih satu atau beberapa foto. Klik Analisis foto. Bahas
   probabilitas dan peringatan review, termasuk contoh prediksi salah/ragu.
   Hasil dapat diunduh sebagai JSON. Foto prediksi tidak otomatis masuk dataset.
3. Di **Dataset & label**, tunjukkan jumlah foto, yang sudah terkonfirmasi, dan yang belum ditinjau.
   Foto baru hanya diimpor setelah pemilik mengonfirmasi hasil pemotretan sendiri.
   File asli tidak diubah; salinan identik dilewati.
4. Isi label dan satu ID tetap per buah fisik. Semua hari/sudut satu buah memiliki
   ID sama. Simpan label; backup dibuat. Edit dari tab lain yang usang akan ditolak
   agar tidak menimpa perubahan baru. Gunakan filter/search untuk meninjau data.
5. Di **Training**, lihat alasan dataset belum siap. Jika sudah lolos, pilih jumlah
   epoch. Klik Mulai training lalu konfirmasi. Tiga
   eksperimen dijalankan, log dapat dipantau. Jangan menutup terminal server.
6. Setelah sukses, klik **Muat model terpilih terbaru**. Model lama tidak ditimpa;
   run disimpan pada direktori baru. Jika dihentikan, file parsial dipertahankan
   dan model terpilih tetap seperti sebelumnya. Menutup server membatalkan job
   training yang dimulai oleh sesi server tersebut.
7. Di **Evaluasi**, bahas accuracy, macro-F1, loss, jumlah kelompok test, confusion
   matrix, metrik per kelas, kurva pembelajaran, dan robustness. Unduh laporan JSON.

## Pernyataan hasil yang tepat

Sebutkan jumlah foto test, jumlah kelompok test, accuracy, dan macro-F1 apa
adanya, lalu nyatakan bahwa dataset masih kecil dan hasil belum membuktikan
kemampuan generalisasi pada foto lapangan. Tampilkan kesalahan model, bukan
hanya contoh yang benar.

## Masalah umum

- **Training nonaktif:** lihat blokir di halaman Training; label/ID buah harus
  dikonfirmasi dan tiap kelas membutuhkan kelompok independen. Jangan menciptakan
  ID baru dari sudut/hari untuk melewati pemeriksaan.
- **Label berubah di tab lain:** salin catatan yang belum disimpan, muat ulang,
  lalu terapkan ulang. Aplikasi sengaja tidak menimpa versi lebih baru.
- **Impor ditolak:** gunakan JPG/PNG/WebP statis, maksimal 10 MB/25 megapiksel,
  minimal sisi 32 piksel; centang konfirmasi foto sendiri.
- **Model tidak ada:** pelabelan tetap tersedia dengan `--labels-only`. Jalankan
  pipeline lalu pilih model melalui `--run` atau penunjuk model terpilih.
- **Port sedang dipakai:** buka tab browser aplikasi yang sudah berjalan atau
  gunakan port lain. Jangan mematikan proses lain secara sembarang.
- **Training gagal:** baca log; hasil parsial tetap tersimpan. Perbaiki sebabnya
  lalu mulai run baru. Jangan memakai checkpoint parsial sebagai model final.
- **Prediksi benda bukan tomat:** sistem ini classifier tiga kelas, bukan detektor
  non-tomat. Masukkan satu tomat terlihat jelas dan periksa hasil secara manual.

## Pengujian pengembang

```bash
venv/bin/python -m unittest discover -s tests -v
venv/bin/python tests/http_smoke.py
```

Tes HTTP memerlukan server pada port 7860 dan tidak menyimpan label. Tes unit
impor/penyimpanan memakai direktori sementara, bukan foto pengguna. Pengujian
browser tambahan memakai Node 22+ dan Chrome headless dengan remote debugging
port 9234: jalankan `node tests/browser_smoke.mjs`. Profil browser harus terpisah
dari profil pribadi. Tes browser membaca jumlah inventaris saat berjalan dan
menggunakan dua foto tersimpan di `data/raw/` untuk menguji upload, bukan akurasi.
