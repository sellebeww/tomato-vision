# Berkas pengumpulan Tomato Vision

Laporan utama: **IS794_Laporan_TomatoVision.docx**. Salinan di `laporan/` dijaga
identik agar tidak ada laporan lama dengan hasil 100% yang terkirim tanpa sengaja.
Sesuai permintaan, laporan dibuat Word saja. Ekspor ke PDF dari Word saat perlu
memenuhi format resmi dosen; identitas kelompok digabungkan dari file sendiri.

- `presentasi/IS794_Presentasi_Final.pptx`: 20 slide, termasuk speaker notes.
- `presentasi/IS794_Presentasi_Week7.pptx`: 10 slide akuisisi dan EDA.
- `../notebooks/IS794_TomatoVision_Project.ipynb`: kode proyek dan output eksekusi lokal.
- `dataset/`: satu paket dataset (`labels.csv` induk, gambar, `prompts.jsonl`, `report.json`, ZIP). Lihat `dataset/README.md`.
- `dataset_lama/`: tata letak, ZIP, dan CSV dataset sebelum disatukan; aman dihapus bila tidak diperlukan.
- `audit/KESESUAIAN_IS794.md`: pemetaan syarat PDF tugas dan bukti.
- `audit/PEMBEDA_DAN_PRESENTASI.md`: kontribusi, batas klaim, dan latihan tanya jawab.
- `audit/deliverable_validation.json`: pemeriksaan struktur dan konsistensi berkas.

Target perluasan terbaru adalah **300 gambar total**. Selama proses generasi,
`labels.csv` dan ZIP sebelumnya tetap berisi versi terakhir yang sudah lolos audit.
Jumlah terbaru yang benar-benar selesai dapat dilihat di `variation_report.json`
setelah `python -m scripts.audit_generated_variations` dijalankan.

Penamaan pengumpulan resmi:
`IS794-Deep Learning-ClassXX-GroupXX-Week7.zip` dan
`IS794-Deep Learning-ClassXX-GroupXX-Presentation Final.zip`.
Ganti ClassXX/GroupXX dengan identitas yang benar. Judul unik dan kontribusi
anggota perlu dicocokkan dengan kelas/dokumen tim.

Hasil model yang dibahas adalah validation own_v3_augmented, bukan hasil situs
historis own_v1. Accuracy 66,7% berasal dari empat jawaban benar pada enam gambar.
Jumlah file dataset tidak membuktikan jumlah data primer terverifikasi.
