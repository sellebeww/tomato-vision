# Kontribusi yang dapat dipertanggungjawabkan

Usulan fokus judul: **Klasifikasi Kesegaran Visual Tomat dengan CNN Ringan,
Pemisahan Sesi, dan Evaluasi Prediksi Selektif**. Ini penajaman ruang lingkup,
bukan klaim algoritma pertama atau lebih unggul dari seluruh penelitian terdahulu.

## Pembeda terhadap karya yang ditinjau

1. Tiga kondisi satu tomat: segar, tidak segar, busuk. Tidak menyamakan kesegaran
   dengan kematangan. Artikel YOLO kualitas tomat [3] memiliki tugas deteksi;
   artikel [1], [2], [4], dan [5] memakai data/tugas lain.
2. Audit unit pemisahan: setiap sesi/kelompok tetap dalam satu split. Audit ini
   telah membatalkan angka historis 100% yang terpengaruh kebocoran sesi.
3. Prediksi selektif: laporan menilai berapa banyak foto tetap diterima (coverage)
   dan berapa proporsi prediksi diterima yang salah (selective risk). Accuracy
   pada subset tidak dilaporkan tanpa penyebut dan coverage.
4. Rancangan variasi terstruktur: latar, pencahayaan, ukuran objek, sudut, dan
   bentuk tomat dicatat dalam prompt manifest. Faktor adegan digunakan pada
   ketiga kelas. Ini rancangan stress-test, bukan pasangan foto buah fisik sama
   atau pengukuran kausal pengaruh latar.

## Pertanyaan penelitian

- RQ1: Bagaimana kinerja dua CNN dari nol pada split berdasarkan sesi?
  Bukti saat ini: validation macro-F1 0,656 untuk kedua kandidat, regularized
  memiliki loss lebih rendah. Hanya enam validation, jadi simpulan terbatas.
- RQ2: Bagaimana trade-off coverage dan kesalahan saat prediksi ragu ditinjau?
  Bukti: `outputs/assignment_audit/selective_validation.json`; ambang 0,7
  sudah ada dalam konfigurasi, bukan dipilih untuk mempercantik hasil.
- RQ3: Bagaimana perilaku model pada variasi adegan dan ukuran objek?
  Rancangan faktor tersedia dalam `generation_prompts_300.jsonl`.
  Atribut itu adalah instruksi generasi; keberhasilannya perlu diperiksa visual.
  Pengujian ini tidak menggantikan evaluasi pada data primer independen.

## Hasil prediksi selektif saat ini

| Kebijakan | Diterima | Coverage | Benar dari yang diterima | Salah diterima |
|---|---:|---:|---:|---:|
| Selalu klasifikasi | 6/6 | 100% | 4/6 (66,7%) | 2 |
| Confidence ≥0,7 | 4/6 | 66,7% | 3/4 (75%) | 1 |
| Confidence + kualitas + konsistensi | 4/6 | 66,7% | 3/4 (75%) | 1 |

Dua kebijakan selektif memberikan hasil sama pada sampel ini. Maka belum ada
bukti manfaat tambahan quality/consistency guard pada validation tersebut.
Satu dari dua kesalahan dirujuk untuk review; satu tetap lolos. Jangan menyebut
75% sebagai peningkatan akurasi keseluruhan atau jaminan keamanan.

## Rencana eksperimen berikutnya

Bekukan protokol sebelum training: data primer terverifikasi, label lintas
penilai, kelompok buah/sesi, split test independen, model pembanding, metrik,
dan anggaran training. Bandingkan baseline, regularized, dan satu transfer
learning dengan split/seed sama. Untuk menguji latar secara kausal, ambil foto
buah yang sama dalam beberapa latar pada waktu berdekatan, lalu simpan seluruh
pasangannya dalam satu split. Evaluasi berkelompok dan laporkan ketidakpastian.

## Alur bicara 10–12 menit

Masalah (1 menit), akuisisi dan EDA (2), metode (2), hasil dan kesalahan (3),
demo (1), batas dan tindak lanjut (1–2). Bagikan bagian sesuai kontribusi nyata;
identitas anggota dan pembagian kontribusi digabungkan dari dokumen tim.

## Latihan pertanyaan

- Mengapa bukan 100%? Angka lama terkena kebocoran sesi; setelah perbaikan,
  hasil yang dipakai adalah 4/6 validation, bukan test independen.
- Mengapa model kecil? Regularized memiliki 96.019 parameter; efisiensi bukan
  bukti otomatis bahwa akurasinya lebih tinggi.
- Apa beda validation dan test? Validation memilih epoch/model; test disimpan
  untuk pemeriksaan independen setelah pemilihan selesai.
- Apakah 300 gambar pasti memenuhi tugas? Jumlah file saja tidak membuktikan
  asal primer, label benar, atau buah independen.
- Apakah perlu review menaikkan accuracy? Accuracy subset naik dengan mengurangi
  coverage. Selalu laporkan keduanya dan jumlah kesalahan yang masih diterima.
- Apa novelty-nya? Kombinasi audit sesi, diagnosis variasi, dan prediksi selektif
  pada studi kasus ini; bukan klaim menemukan CNN atau abstention baru.
