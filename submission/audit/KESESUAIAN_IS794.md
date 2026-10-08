# Audit ketentuan IS794

Dasar: `IS794-Deep Learning-Final Project-Gsl2026-2027.pdf`, halaman 1–3,
dibaca 7 Oktober 2026. Dokumen ini adalah audit, bukan laporan utama.
Laporan utama berada di `submission/IS794_Laporan_TomatoVision.docx`.

| Ketentuan | Bukti / status |
|---|---|
| Studi kasus dengan neural network | Tiga kelas kondisi tomat; baseline dan residual separable CNN di `src/model.py` |
| Data primer dari akuisisi | Belum terverifikasi lengkap. Metadata `own` bukan bukti kamera; gambar sintetis tidak diubah menjadi `own` |
| Minimal 100 gambar | Jumlah paket melampaui 100, tetapi jumlah primer terverifikasi belum memenuhi bukti persyaratan |
| Evaluasi relevan | Accuracy, macro precision/recall/F1, confusion matrix, ECE, coverage dan selective risk tersedia untuk validation |
| Deployment, nilai tambah | Aplikasi TensorFlow lokal dan CLI tersedia; model web historis tidak dianggap hasil eksperimen terbaru |
| Week 7: PPT, laporan PDF, dataset | PPT Week 7 dan dataset tersedia; laporan dibuat Word sesuai permintaan pengguna, ekspor PDF diperlukan saat pengumpulan resmi |
| Final: PPT, IPYNB, laporan PDF, dataset | PPT final dan IPYNB proyek tersedia; pengecualian format Word sama seperti di atas |
| Judul unik antar kelompok | Tim perlu memeriksa daftar judul kelas |
| Format nama ZIP | `IS794-Deep Learning-ClassXX-GroupXX-Week7.zip` dan `IS794-Deep Learning-ClassXX-GroupXX-Presentation Final.zip`; ganti XX dari identitas terpisah |
| Referensi 5–10 | 8 rujukan dengan URL penerbit, termasuk 5 jurnal terkait dan fondasi metode |
| Pemahaman dan kontribusi seluruh anggota | Speaker notes dan panduan tanya jawab tersedia; kontribusi aktual berasal dari dokumen tim |

PDF ketentuan tidak menyertakan template laporan terpisah. Struktur Word mengikuti
heading template yang sudah ada di proyek. Tidak ada identitas, kontribusi tim,
atau angka eksperimen yang direkayasa. Syarat administratif yang belum lengkap
tidak boleh diartikan sebagai kegagalan teknis pipeline.

Perbaikan utama: hasil historis own_v1 100% tidak digunakan lagi sebagai hasil
utama karena kebocoran sesi; hasil aktual own_v3 adalah validation 4/6 benar.
Test terkunci tidak dievaluasi ulang untuk menyusun laporan.
