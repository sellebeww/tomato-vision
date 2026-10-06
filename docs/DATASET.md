# Dataset card dan protokol pengumpulan

## Sumber dan tujuan

Target adalah kondisi visual satu buah tomat, bukan varietas, tingkat kematangan,
atau keamanan konsumsi. Tomat hijau tidak otomatis tidak segar/busuk.
Data publik dan dummy lama tidak digunakan.

- Folder `data/reference_import/`: 26 file JPEG, 22 unik; empat salinan identik
  dikecualikan tanpa dihapus. Label dan ID buahnya belum dikonfirmasi.
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
