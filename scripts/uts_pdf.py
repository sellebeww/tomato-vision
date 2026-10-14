"""Bangun PDF final UTS IS794 dari artefak audit di submission/uts."""
from pathlib import Path
import json
import re
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
UTS = ROOT / 'submission/uts'
OUT = UTS / 'IS794_UTS_00000138850_Vassel_Goleyu_TomatoVision.pdf'
FD = '/System/Library/Fonts/Supplemental/'
pdfmetrics.registerFont(TTFont('A', FD + 'Arial.ttf'))
pdfmetrics.registerFont(TTFont('A-B', FD + 'Arial Bold.ttf'))
pdfmetrics.registerFont(TTFont('A-I', FD + 'Arial Italic.ttf'))
pdfmetrics.registerFontFamily('A', normal='A', bold='A-B', italic='A-I', boldItalic='A-B')

BODY = ParagraphStyle('b', fontName='A', fontSize=10, leading=14.2, alignment=TA_JUSTIFY, spaceAfter=6)
H1 = ParagraphStyle('h1', fontName='A-B', fontSize=14, leading=18, spaceBefore=12, spaceAfter=6, textColor=colors.HexColor('#7a1f1f'))
H2 = ParagraphStyle('h2', fontName='A-B', fontSize=11, leading=15, spaceBefore=8, spaceAfter=4)
CAP = ParagraphStyle('c', fontName='A-I', fontSize=8.5, leading=11, spaceAfter=6, alignment=1)
CELL = ParagraphStyle('t', fontName='A', fontSize=8.5, leading=11)
CELLB = ParagraphStyle('tb', parent=CELL, fontName='A-B')
REF = ParagraphStyle('r', parent=BODY, alignment=0, leftIndent=14, firstLineIndent=-14, fontSize=9, leading=12.5)
TITLE = ParagraphStyle('ti', fontName='A-B', fontSize=14, leading=19, alignment=1, spaceAfter=4)
SUB = ParagraphStyle('su', fontName='A', fontSize=10.5, leading=15, alignment=1)

story = []
P = lambda t: story.append(Paragraph(re.sub(r'(?<=\d)\.(?=\d)', ',', t), BODY))
H = lambda t: story.append(Paragraph(t, H1))
h = lambda t: story.append(Paragraph(t, H2))


def table(rows, widths, head=True):
    data = [[Paragraph(str(c), CELLB if (head and i == 0) else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if head else 0)
    st = [('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#999999')),
          ('VALIGN', (0, 0), (-1, -1), 'TOP'),
          ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]
    if head:
        st.append(('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f0e3e3')))
    t.setStyle(TableStyle(st))
    story.extend([t, Spacer(1, 8)])


def fig(n, caption, width=13.5):
    path = UTS / f'figures/{n:02d}.png'
    w, hh = PILImage.open(path).size
    img = Image(str(path), width=width * cm, height=width * cm * hh / w)
    story.append(KeepTogether([img, Paragraph(caption, CAP)]))


def f4(x):
    return f'{x:.4f}'.replace('.', ',')


# ---------- data ----------
S = json.loads((UTS / 'audit_summary.json').read_text())
fm = pd.read_csv(UTS / 'frozen_manifest.csv', keep_default_na=False)
m = pd.read_csv(UTS / 'training_metrics.csv')
pairs = pd.read_csv(UTS / 'training_dhash_pairs.csv')
order = ['segar', 'tidak_segar', 'busuk']
cols = ['brightness', 'saturation', 'edge', 'red_mean', 'green_mean', 'blue_mean']
corr = m[cols].corr()
N = len(fm)
TR = fm[fm.split == 'train']
lab = fm.label.value_counts()
assert N == S['canonical_unique_images'] == 229 and len(m) == len(TR)
med = m.groupby('label')[['brightness', 'saturation', 'edge']].median()
rng = m.groupby('label')[['brightness', 'saturation']].agg(['min', 'max'])
pct = lambda x, y=N: f'{100 * x / y:.1f}'.replace('.', ',')
grp = fm.groupby('group_id').agg(n=('label', 'size'))
top2 = int(grp.n.nlargest(2).sum())
near = pairs[pairs.distance <= 4]
png = fm[fm.filepath.str.endswith('.png')].label.value_counts()
splits = {s: fm[fm.split == s] for s in ('train', 'val', 'test')}


def outliers(col):
    k = 0
    for _, g in m.groupby('label'):
        q1, q3 = g[col].quantile([.25, .75])
        k += int(((g[col] < q1 - 1.5 * (q3 - q1)) | (g[col] > q3 + 1.5 * (q3 - q1))).sum())
    return k


# ---------- cover ----------
story += [Paragraph('UJIAN TENGAH SEMESTER — IS794 Deep Learning (Theory)', SUB), Spacer(1, 6),
          Paragraph('Tomato Vision: Klasifikasi Tingkat Kesegaran Tomat (Segar, Tidak Segar, Busuk) Menggunakan Convolutional Neural Network (CNN) Berbasis TensorFlow yang Dilatih dari Nol', TITLE),
          Paragraph('Vassel Goleyu • NIM 00000138850<br/>Universitas Multimedia Nusantara • Semester Ganjil 2026/2027', SUB),
          Spacer(1, 10)]

P(f'<b>Ringkasan.</b> Laporan ini memaparkan dataset primer proyek akhir saya: {N} foto tomat yang saya ambil '
  f'sendiri, seluruhnya berlabel dan dikelompokkan menjadi tiga kondisi visual (segar {lab["segar"]}, tidak_segar '
  f'{lab["tidak_segar"]}, busuk {lab["busuk"]}). Sepuluh visualisasi EDA menunjukkan empat hal penting untuk '
  f'pemodelan: kelas tidak sepenuhnya seimbang, foto menumpuk pada dua grup besar, 22 JPEG semuanya segar, dan '
  f'statistik warna/tekstur global saja tidak memisahkan kelas. Temuan ini mendukung pemakaian CNN, tetapi juga '
  f'menuntut split berbasis grup dan evaluasi yang hati-hati karena jumlah buah independen tidak diketahui.')

# ---------- 1 ----------
H('1. Dataset Description')
h('1.1 Latar belakang studi kasus')
P('Tomat mudah rusak: dalam beberapa hari kulitnya dapat berubah dari halus dan kencang menjadi keriput, '
  'lalu muncul bercak dan jamur. Penyortiran kesegaran umumnya dilakukan dengan mata, sehingga hasilnya '
  'bergantung pada penilai. Klasifikasi citra otomatis dapat membantu penyaringan awal berdasarkan penampilan '
  'luar. Mukhiddinov et al. (2022) menunjukkan bahwa deep learning dapat mengenali kesegaran buah dan sayur, '
  'sekaligus mencatat bahwa perubahan kondisi pengambilan gambar menyulitkan model.')
P('Pertanyaan proyek: apakah pola visual pada foto tomat dapat dipelajari CNN untuk membedakan tiga kondisi '
  '(segar, tidak_segar, busuk)? Target adalah kondisi <i>visual</i>, bukan umur simpan, kandungan mikroba, atau '
  'keamanan konsumsi. Kematangan (hijau/merah) juga berbeda dari kesegaran, jadi warna merah saja tidak dipakai '
  'sebagai kriteria.')

h('1.2 Cara pengumpulan data')
P('Metode: <b>observasi dan fotografi langsung</b>. Saya memotret tomat sendiri dari kondisi segar sampai busuk, '
  'dengan latar antara lain meja kayu, ubin, piring, kardus, dan beberapa permukaan lain. Seluruh foto adalah milik saya '
  '(pernyataan saya sebagai pemilik); sebagian besar berformat PNG dan 22 berformat JPEG. Berkas tidak memiliki EXIF, '
  'sehingga asal pemotretan tidak dapat dibuktikan dari metadata; audit checksum hanya membuktikan konsistensi berkas.')
table([['Komponen', 'Isi'],
       ['Sampel', f'{N} foto unik: {S["dimensions"][1]["n"]} PNG (1254 × 1254) dan {S["dimensions"][0]["n"]} JPEG (1200 × 1600)'],
       ['Label', f'segar {lab["segar"]} • tidak_segar {lab["tidak_segar"]} • busuk {lab["busuk"]}'],
       ['Variabel', 'citra RGB (input); label 3 kelas (target); metadata: file, SHA-256, grup, split, format, ukuran'],
       ['Grup metadata', f'{fm.group_id.nunique()} grup berdasarkan latar/sesi/adegan (bukan jumlah buah)'],
       ['Duplikat', f'{S["duplicate_excess"]} salinan identik dikecualikan ({S["source_files"]} berkas sumber → {S["unique_source_bytes"]} unik)']],
      [3.5 * cm, 12.5 * cm])
P('<b>Keterbatasan yang jujur.</b> Tanggal pemotretan tidak tercatat, jumlah buah fisik tidak dapat saya pastikan, '
  'dan identitas buah antarfoto tidak dicatat. Karena itu satuan yang dihitung adalah <i>foto</i>, bukan buah '
  'independen. Proses pelabelan: sebagian besar label berasal dari anotasi proyek saya dan nama berkas; label 22 JPEG '
  '(semuanya segar) ditetapkan lewat inspeksi visual per foto atas permintaan saya. Tidak ada '
  'pemeriksaan fisik maupun penilai kedua, dan sebagian label belum saya periksa satu per satu, jadi semua label saya '
  'perlakukan sebagai anotasi visual, bukan ground truth. Kriteria label: segar = kulit relatif halus/kencang tanpa '
  'pembusukan tampak; tidak_segar = keriput/layu tanpa pembusukan nyata; busuk = lesi pembusukan atau jamur yang jelas.')

h('1.3 Relevansi dataset dan alasan memakai neural network')
P('Dataset ini relevan karena input (foto tomat) dan target (kondisi visual) berhubungan langsung dengan masalah, '
  'dan foto adalah data primer milik saya. Tanda kerusakan seperti keriput dan lesi bersifat '
  '<i>lokal</i>, dapat muncul di posisi mana pun, dan berupa kombinasi bentuk, warna, serta tekstur. CNN cocok '
  'karena konvolusi berbagi parameter pada seluruh posisi gambar dan lapisan bertingkat menggabungkan pola lokal '
  'sederhana menjadi pola yang lebih kompleks (Goodfellow et al., 2016; Chollet, 2021). Jaringan fully connected '
  'yang meratakan piksel membuang struktur spasial itu, sedangkan rerata warna saja menghilangkan lokasi kerusakan; '
  'EDA di bawah juga memperlihatkan bahwa deskriptor global saling tumpang tindih antarkelas.')
P(f'Kesesuaian ini bukan jaminan keberhasilan. Dengan {N} foto dari buah independen yang jumlahnya tidak diketahui, '
  'CNN berisiko menghafal latar atau objek. Rencana awalnya: model berkapasitas kecil, regularisasi, augmentasi '
  'ringan hanya pada training, dan baseline sederhana berbasis fitur warna/tekstur sebagai pembanding.')

# ---------- 2 ----------
H('2. Data Preparation')
h('2.1 Langkah preprocessing yang dijalankan')
table([['Langkah', 'Keterangan'],
       ['1. Inventaris & audit', f'{S["inventory_files"]} berkas di empat lokasi dibuka dan didekode; {S["corrupt"]} rusak. Salinan hanya dihitung sekali.'],
       ['2. Deduplikasi', f'Hash byte dan hash piksel: {S["duplicate_excess"]} salinan identik (tomat_ref_06, 07, 13, 22) dikecualikan; tidak ada duplikat di antara {N} foto.'],
       ['3. Validasi label', f'Semua {N} label berada pada tiga kategori; checksum cocok dengan manifest. Label dan grup dicatat per foto.'],
       ['4. Koreksi orientasi & RGB', 'Orientasi EXIF diterapkan; semua gambar dikonversi ke RGB.'],
       ['5. Resize + padding', 'Resize proporsional (bilinear) ke 128 × 128, sisa area diisi abu-abu (127,127,127); bentuk tomat tidak diregangkan.'],
       ['6. Normalisasi', 'float32, dibagi 255 → rentang [0,1]. Tidak ada statistik yang dihitung dari validation/test.'],
       ['7. Split berbasis grup', 'Foto dari grup yang sama tidak dipisah antarsplit; tidak ada group_id/hash yang muncul di lebih dari satu split.']],
      [4.2 * cm, 11.8 * cm])
P('Ukuran 128 adalah konfigurasi proyek, bukan ukuran optimal yang sudah dibuktikan; lesi kecil dapat hilang '
  'saat diperkecil, dan padding pada foto portrait dapat menjadi petunjuk sumber. Augmentasi (flip, rotasi, zoom, '
  'kontras, brightness) tersedia di kode proyek tetapi tidak dijalankan untuk laporan ini; bila dipakai, hanya pada '
  'training setelah split (Shorten &amp; Khoshgoftaar, 2019).')

h('2.2 Dataset bersih')
P(f'Versi bersih tersedia sebagai <b>clean_dataset_128.npz</b> (tensor {N} × 128 × 128 × 3, label, split) '
  'dan <b>clean_manifest.csv</b> (indeks → file sumber, label, grup, split, SHA-256, asal anotasi), '
  'keduanya di folder submission/uts. Label: 0 = segar, 1 = tidak_segar, 2 = busuk.')
rows = [['Split', 'segar', 'tidak_segar', 'busuk', 'Total', 'Grup']]
for nm, key in (('Training', 'train'), ('Validation', 'val'), ('Test', 'test')):
    d = splits[key]; c = d.label.value_counts()
    rows.append([nm, c['segar'], c['tidak_segar'], c['busuk'], len(d), d.group_id.nunique()])
rows.append(['<b>Total</b>', lab['segar'], lab['tidak_segar'], lab['busuk'], N, fm.group_id.nunique()])
table(rows, [3.2 * cm, 2.4 * cm, 2.8 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm])
nv = len(splits['val']); nt = len(splits['test']); ntr = len(splits['train'])
P(f'Proporsi aktual {pct(ntr)}% / {pct(nv)}% / {pct(nt)}%, bukan 70/15/15, karena split mengikuti grup (dua grup besar tidak dapat '
  f'dipecah) dan test tetap enam foto yang sudah terkunci dari eksperimen sebelumnya. Validation berisi {nv} foto dari '
  f'{splits["val"].group_id.nunique()} grup, test enam foto dari dua grup, sehingga satu kesalahan pada test mengubah akurasi 16,7 poin '
  f'persentase. EDA piksel di bawah hanya memakai {len(m)} foto training; validation/test tidak dipakai untuk memilih strategi.')

# ---------- 3 ----------
H('3. Exploratory Data Analysis')
h('3.1 Statistik deskriptif')
P(f'Deskriptor dihitung dari {len(m)} foto training (thumbnail sisi maksimum 256 px): <b>luminans</b> = 0,299R + '
  '0,587G + 0,114B; <b>saturasi</b> = rerata kanal S (HSV); <b>gradien</b> = rerata selisih absolut luminans piksel '
  'bertetangga (deskriptor tekstur kasar, bukan ukuran blur). Foto dalam satu grup tidak independen, jadi tidak '
  'dilakukan uji signifikansi.')
rows = [['Deskriptor', 'Kelas', 'n', 'Mean', 'Std', 'Min', 'Median', 'Max']]
for col, nm in [('brightness', 'Luminans'), ('saturation', 'Saturasi'), ('edge', 'Gradien')]:
    for lb in order:
        v = m[m.label == lb][col]
        rows.append([nm, lb, len(v), f4(v.mean()), f4(v.std(ddof=1)), f4(v.min()), f4(v.median()), f4(v.max())])
table(rows, [2.4 * cm, 2.6 * cm, 1.2 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm])

h('3.2 Visualisasi dan interpretasi')
P(f'<b>Gambar 1 — Distribusi kelas.</b> Segar {lab["segar"]} ({pct(lab["segar"])}%), tidak_segar {lab["tidak_segar"]} dan '
  f'busuk {lab["busuk"]} (masing-masing {pct(lab["busuk"])}%); rasio terbesar:terkecil {lab["segar"] / lab["busuk"]:.2f}. '
  'Ketidakseimbangannya ringan, tetapi kelas segar lebih besar karena 22 JPEG seluruhnya berlabel segar; ini artefak '
  'komposisi sumber, bukan bukti bahwa tomat segar lebih umum. Model yang selalu menebak segar mendapat akurasi '
  f'{pct(lab["segar"])}% tanpa mengenali kerusakan, jadi akurasi saja menyesatkan; saya akan memakai macro-F1, '
  'recall per kelas, dan confusion matrix.')
fig(1, 'Gambar 1. Jumlah foto unik per label.', 9.5)
P(f'<b>Gambar 2 — Label per kelompok grup.</b> Grup a_meja_kayu berisi 30 foto (10 per kelas), grup r_reference (22 JPEG) '
  f'berisi 22 foto segar, sepuluh grup b dan 49 adegan scene masing-masing berisi tiga foto (satu per kelas). Dua grup '
  f'besar memuat {top2} dari {N} foto ({pct(top2)}%), jadi sebagian besar keragaman berasal dari puluhan grup kecil. '
  'Grup r_reference hanya berlabel segar, sehingga latar/gaya foto itu berasosiasi dengan satu kelas — jalan pintas '
  'yang mudah dipelajari CNN (Geirhos et al., 2020). Grup yang berisi ketiga kelas pada satu latar membantu memutus '
  f'asosiasi latar–label, tetapi belum membuktikan bahwa buahnya berbeda; jumlah buah sebenarnya tidak diketahui, '
  f'jadi {S["groups"]} grup bukan {S["groups"]} buah.')
fig(2, 'Gambar 2. Komposisi label menurut kelompok grup.', 12)
trc = splits['train'].label.value_counts()
vac = splits['val'].label.value_counts()
P(f'<b>Gambar 3 — Komposisi split.</b> Training {trc["segar"]}/{trc["tidak_segar"]}/{trc["busuk"]} ({ntr} foto, '
  f'{splits["train"].group_id.nunique()} grup), validation {vac["segar"]}/{vac["tidak_segar"]}/{vac["busuk"]} '
  f'({nv} foto, {splits["val"].group_id.nunique()} grup), dan test 2/2/2 ({nt} foto, 2 grup). Validation dan test '
  'seimbang antarkelas, sedangkan training condong ke segar; akurasi antarsplit tidak langsung sebanding. Test hanya enam '
  'foto dari dua grup, sehingga satu kesalahan mengubah akurasi 16,7 poin persentase; hasil harus dilaporkan per kelas '
  'dengan jumlah grup, dan test tidak boleh dipakai untuk memilih hyperparameter.')
fig(3, 'Gambar 3. Jumlah label pada training, validation, dan test.', 9.5)
P(f'<b>Gambar 4 — Dimensi dan format.</b> Semua {S["dimensions"][1]["n"]} PNG tepat 1254 × 1254 dan semua '
  f'{S["dimensions"][0]["n"]} JPEG 1200 × 1600. PNG seimbang ({png["segar"]}/{png["tidak_segar"]}/{png["busuk"]}), '
  'sehingga format PNG tidak memprediksi satu kelas; namun seluruh JPEG berlabel segar, jadi format portrait menjadi '
  'bocoran label lewat sumber. Ini bias akuisisi: model bisa "mengenali segar" dari bingkai portrait atau padding hasil '
  'resize. Format, nama file, dan ukuran asli tidak boleh menjadi fitur; mitigasi terbaik adalah menambah foto dengan '
  'format/rasio beragam pada semua kelas dan menguji crop objek yang seragam.')
fig(4, 'Gambar 4. Lebar dan tinggi gambar sumber menurut format.', 9.5)
P('<b>Gambar 5 — Contoh per kelas.</b> Contoh dipilih deterministik dari training, satu gambar per grup. Segar tampak '
  'mulus, tidak_segar tampak keriput, dan busuk memperlihatkan area gelap atau bercak. Tanda ini lokal dan kecil terhadap '
  'bingkai, sehingga mendukung CNN dibanding rerata warna. Latar (kayu, ubin, piring, kardus) dan skala objek berbeda antar '
  'grup, jadi skala dan latar dapat ikut menentukan pola. Crop tidak boleh memotong lesi di tepi buah.')
fig(5, 'Gambar 5. Contoh foto training per kelas (kolom: grup berbeda).', 14.5)
P(f'<b>Gambar 6 — Luminans.</b> Median segar {f4(med.loc["segar","brightness"])}, tidak_segar '
  f'{f4(med.loc["tidak_segar","brightness"])}, busuk {f4(med.loc["busuk","brightness"])}; rentangnya tumpang tindih '
  f'(sekitar {f4(rng["brightness"]["min"].min())}–{f4(rng["brightness"]["max"].max())}) dan selisih median jauh lebih kecil '
  f'dari simpangan baku per kelas (≈ 0,07–0,08). Urutannya tidak turun monoton dari segar ke busuk, sehingga aturan '
  f'"lebih gelap = lebih busuk" tidak didukung data. Ada {outliers("brightness")} titik di luar whisker (aturan 1,5 IQR); '
  'penyebabnya lebih mungkin pencahayaan dan latar daripada kerusakan, sehingga gambar ekstrem perlu ditinjau, bukan '
  'dibuang otomatis. Augmentasi brightness ringan masuk akal; equalization agresif tidak karena dapat mengubah tampilan lesi.')
fig(6, 'Gambar 6. Luminans per kelas (titik = foto training).', 9.5)
P(f'<b>Gambar 7 — Saturasi.</b> Median segar {f4(med.loc["segar","saturation"])}, tidak_segar '
  f'{f4(med.loc["tidak_segar","saturation"])}, busuk {f4(med.loc["busuk","saturation"])}: segar sedikit lebih jenuh '
  f'daripada dua kelas lain dan urutannya segar &gt; tidak_segar &gt; busuk, tetapi selisihnya (≈ 0,02–0,06) jauh di bawah '
  'simpangan baku (≈ 0,10–0,12), sehingga tumpang tindih besar. Asumsi "makin jenuh makin segar" hanya didukung lemah. Saturasi global bercampur dengan latar dan white balance, jadi hanya cocok sebagai '
  'baseline atau fitur diagnostik. CNN sebaiknya tetap menerima RGB penuh, dan augmentasi hue/saturasi kuat dihindari '
  'agar sinyal warna pembusukan tidak hilang.')
fig(7, 'Gambar 7. Distribusi saturasi rata-rata per kelas.', 9.5)
P(f'<b>Gambar 8 — Gradien (tekstur kasar).</b> Median {f4(med.loc["segar","edge"])} / {f4(med.loc["tidak_segar","edge"])} / '
  f'{f4(med.loc["busuk","edge"])} untuk segar / tidak_segar / busuk — selisih ≈ 0,001 dengan simpangan baku ≈ 0,006–0,007, '
  f'jadi hampir tidak ada pemisahan; ada {outliers("edge")} titik di luar whisker. Keriput dan lesi bersifat lokal '
  'sehingga tidak terangkat oleh rerata global, sementara tepi piring atau serat kayu justru menaikkan nilainya. '
  'Kesimpulannya: butuh fitur spasial, bukan statistik keseluruhan gambar. Gambar bergradien rendah tidak dibuang '
  'otomatis karena dapat mengubah distribusi kelas.')
fig(8, 'Gambar 8. Rerata gradien absolut per kelas.', 9.5)
c = corr
P(f'<b>Gambar 9 — Korelasi deskriptor.</b> Luminans sangat berkorelasi dengan rerata hijau ({c.loc["brightness","green_mean"]:.2f}), '
  f'merah ({c.loc["brightness","red_mean"]:.2f}), dan biru ({c.loc["brightness","blue_mean"]:.2f}), wajar karena luminans '
  f'adalah kombinasi kanal RGB; saturasi berkorelasi negatif dengan rerata biru ({c.loc["saturation","blue_mean"]:.2f}) dan '
  f'cukup dengan luminans ({c.loc["saturation","brightness"]:.2f}). Gradien hampir tidak berkorelasi dengan deskriptor warna '
  f'(|r| ≤ {c.loc["edge", ["brightness","saturation","red_mean","green_mean","blue_mean"]].abs().max():.2f}), jadi ia membawa informasi berbeda '
  'tetapi hampir tidak berbeda antarkelas. Untuk baseline tabular, fitur yang sangat berkorelasi redundan dan perlu '
  'diseleksi dari training saja. Untuk CNN hal ini bukan alasan membuang kanal. Korelasi bersifat deskriptif dan '
  'bisa diperkuat oleh banyaknya foto serupa dalam satu grup.')
fig(9, f'Gambar 9. Korelasi Pearson antardeskriptor ({len(m)} foto training).', 10.5)
a_, b_ = len(pairs), len(near)
a_ = f'{a_:,}'.replace(',', ' ')
P(f'<b>Gambar 10 — Kemiripan visual (dHash).</b> Dari {a_} pasangan foto training, {b_} pasangan berjarak dHash ≤ 4: '
  'enam dari grup JPEG r_reference, satu (TOM042 tidak_segar – TOM052 busuk) dari grup piring, dan satu (own_067 segar – '
  'own_069 busuk) dari satu adegan scene. Semuanya berada dalam grup yang sama, sesuai dugaan bahwa satu grup memuat '
  'foto yang mirip. Dua pasangan lintas label itu bisa karena latar dan komposisi sama, atau karena buah yang sama difoto '
  'pada kondisi berbeda — belum dapat dipastikan tanpa ID buah. Saya menjadikannya kandidat tinjauan, bukan duplikat '
  'otomatis; tidak ada pasangan ≤ 4 yang melintasi split.')
fig(10, 'Gambar 10. Distribusi jarak dHash 64-bit pada semua pasangan training.', 9.5)

h('3.3 Insight untuk pemilihan fitur dan strategi modeling')
for t in [
    '<b>Split per grup/buah, bukan per foto.</b> Foto serupa dalam satu grup akan bocor antarsplit bila diacak, dan evaluasi menjadi terlalu optimistis.',
    '<b>Metrik per kelas.</b> Karena kelas tidak sepenuhnya seimbang dan test hanya enam foto, gunakan macro-F1, recall kelas busuk, dan confusion matrix; hindari kesimpulan dari satu angka akurasi.',
    '<b>Cegah shortcut.</b> Format JPEG, ukuran, nama file, dan latar berasosiasi dengan label; pakai input RGB seragam, crop objek konsisten, dan tambah data segar dari format/latar yang lebih beragam.',
    '<b>Fitur spasial.</b> Luminans, saturasi, dan gradien tumpang tindih antarkelas, jadi ambang global sederhana tidak cukup; ini alasan empiris untuk mencoba CNN, dengan baseline fitur warna/tekstur sebagai pembanding.',
    '<b>Kualitas label.</b> Sebagian label belum diperiksa satu per satu; foto yang ambigu antara segar dan tidak_segar perlu ditinjau ulang sebelum evaluasi akhir.',
    '<b>Kapasitas kecil + regularisasi.</b> Dengan buah independen yang tidak diketahui, utamakan model kecil, dropout/weight decay, augmentasi ringan pada training, dan transfer learning bila diizinkan.',
    '<b>Prioritas data.</b> Menambah buah baru dengan ID buah tercatat lebih bernilai daripada menggandakan foto lewat augmentasi.']:
    story.append(Paragraph('• ' + t, ParagraphStyle('li', parent=BODY, leftIndent=12, firstLineIndent=-8, spaceAfter=3)))

# ---------- 4 ----------
H('4. Reflection')
P('<b>Kendala nyata.</b> Saat memotret, saya kadang lupa mengambil foto secara rutin. Akibatnya perubahan kondisi tidak '
  'terekam dengan interval yang konsisten, tahap peralihan (misalnya segar → mulai keriput) kurang terwakili, dan '
  'saya tidak lagi dapat memastikan jumlah buah serta tanggal tiap foto. Kendala lain: sebagian label berasal dari nama '
  'berkas dan belum saya periksa satu per satu. Bagi model, ini berarti batas antarkelas yang kabur (label noise), '
  'distribusi foto yang tidak merata antarbuah, dan risiko foto buah yang sama tersebar di split berbeda tanpa terdeteksi.')
table([['Tantangan yang mungkin terjadi', 'Dampak pada model', 'Mitigasi'],
       ['Buah yang sama difoto ulang tanpa ID', 'Leakage; akurasi terlalu optimistis', 'Catat ID buah tetap; gabungkan grup yang meragukan dalam satu split'],
       ['Batas segar vs tidak_segar sulit ditentukan', 'Label noise, batas keputusan tidak konsisten', 'Pedoman observasi tertulis; tinjau foto ambigu; penilai kedua'],
       ['Sebagian label belum diperiksa per foto', 'Label salah merusak training dan metrik', 'Periksa label per foto; laporkan hasil per kelas'],
       ['Latar/pencahayaan berbeda menurut kelas', 'Model belajar shortcut', 'Ambil semua kelas pada beragam latar dan cahaya'],
       ['Sedikit buah, satu varietas, satu kamera', 'Menghafal; gagal di domain baru', 'Tambah buah, varietas, dan perangkat'],
       ['Lesi kecil hilang saat resize', 'Informasi target hilang', 'Uji crop/ukuran input pada validation'],
       ['Tanggal dan kamera tidak dicatat', 'Sulit menelusuri bias dan mereproduksi', 'Log akuisisi: tanggal, perangkat, file, kondisi']],
      [5 * cm, 4.6 * cm, 6.4 * cm])
P('Mitigasi di atas sebagian besar masih rencana untuk pengumpulan berikutnya. Yang sudah dilakukan: audit '
  'berkas, deduplikasi, split berbasis grup, preprocessing seragam, dan pembatasan EDA piksel pada training. Yang belum: '
  'ID buah lintas sesi, pemeriksaan label per foto dan label 22 JPEG secara fisik, penilai kedua, dan data tambahan. '
  'Model kelak hanya menilai kesesuaian <i>tampilan</i> dengan label, bukan kondisi dalam buah atau keamanan pangan.')

# ---------- 5 ----------
H('5. Referensi')
refs = [
    'Chollet, F. (2021). <i>Deep Learning with Python</i> (2nd ed.). Manning Publications.',
    'Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., &amp; Wichmann, F. A. (2020). Shortcut learning in deep neural networks. <i>Nature Machine Intelligence, 2</i>, 665–673. https://doi.org/10.1038/s42256-020-00257-z',
    'Goodfellow, I., Bengio, Y., &amp; Courville, A. (2016). <i>Deep Learning</i>. MIT Press. https://www.deeplearningbook.org/',
    'Ketkar, N., &amp; Santana, E. (2017). <i>Deep Learning with Python: A Hands-on Introduction</i>. Apress.',
    'Mukhiddinov, M., Muminov, A., &amp; Cho, J. (2022). Improved classification approach for fruits and vegetables freshness based on deep learning. <i>Sensors, 22</i>(21), 8192. https://doi.org/10.3390/s22218192',
    'scikit-learn developers. (n.d.). Cross-validation: evaluating estimator performance. https://scikit-learn.org/stable/modules/cross_validation.html',
    'Shorten, C., &amp; Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. <i>Journal of Big Data, 6</i>, 60. https://doi.org/10.1186/s40537-019-0197-0',
    'TensorFlow. (n.d.). Data augmentation. https://www.tensorflow.org/tutorials/images/data_augmentation',
]
for i, r in enumerate(refs, 1):
    story.append(Paragraph(f'{i}. {r}', REF))


def footer(c, d):
    c.saveState()
    c.setFont('A', 8)
    c.setFillColor(colors.grey)
    c.drawString(2 * cm, 1.2 * cm, 'IS794 UTS — Vassel Goleyu (00000138850)')
    c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f'Halaman {d.page}')
    c.restoreState()


SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
                  title='UTS IS794 — Tomato Vision', author='Vassel Goleyu').build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)
