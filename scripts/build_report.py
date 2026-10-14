"""Bangun laporan Word IS794 (format template tugas) dari artefak eksperimen yang ada.

Tidak melatih model, tidak melabel ulang data, dan tidak membuka test terkunci.
Semua angka dibaca dari outputs/experiments/own_v4_ref22, manifest, dan labels.csv.
Jalankan: python -m scripts.build_report
"""
import csv
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/.matplotlib'))
OUT = ROOT / 'submission'
FIG = OUT / 'figures'
RUN = ROOT / 'outputs/experiments/own_v4_ref22'
PREPARED = ROOT / 'data/prepared/ff822adef8db3ae5'
CLASSES = ['segar', 'tidak_segar', 'busuk']
SELECTED = json.loads((RUN / 'selection.json').read_text())['selected'].rsplit('/', 1)[1]
TARGET = OUT / 'IS794_Laporan_TomatoVision.docx'
DATE = '9 Oktober 2026'

MEMBERS = ['1. Vassel Goleyu / 00000138850', '2. Leonarto / 00000126969', '3. Javier / 00000124728', '4. Gary / 00000081808']
TITLE = ('Tomato Vision: Klasifikasi Tingkat Kesegaran Tomat (Segar, Tidak Segar, Busuk) Menggunakan '
         'Convolutional Neural Network (CNN) Berbasis TensorFlow yang Dilatih dari Nol')

REFERENCES = [
 'Y. Shu, J. Zhang, Y. Wang, and Y. Wei, “Fruit Freshness Classification and Detection Based on the ResNet-101 Network and Non-Local Attention Mechanism,” Foods, vol. 14, art. 1987, 2025, doi: 10.3390/foods14111987.',
 'S. Demirel and O. Yildiz, “VisDist-Net: A New Lightweight Model for Fruit Freshness Classification,” Food Analytical Methods, vol. 18, no. 2, pp. 229–244, 2025, doi: 10.1007/s12161-024-02716-4.',
 'Y. A. Mokhtar and E. H. Seddik, “Real-time object detection and diagnosis of tomato quality using YOLO,” Robotics, Integration, Manufacturing and Control, vol. 2, no. 2, art. 1846, 2025, doi: 10.21622/RIMC.2025.02.2.1846.',
 'M. Mukhiddinov, A. Muminov, and J. Cho, “Improved Classification Approach for Fruits and Vegetables Freshness Based on Deep Learning,” Sensors, vol. 22, no. 21, art. 8192, 2022, doi: 10.3390/s22218192.',
 'C. Shorten and T. M. Khoshgoftaar, “A survey on Image Data Augmentation for Deep Learning,” Journal of Big Data, vol. 6, art. 60, 2019, doi: 10.1186/s40537-019-0197-0.',
 'A. Khan et al., “Tomato Maturity Recognition with Convolutional Transformers,” arXiv:2307.01530, 2023 (v2 2024, preprint).',
 'P. Ramos, R. Ramos, and N. Garcia, “Data Leakage in Visual Datasets,” arXiv:2508.17416, 2025 (preprint).',
 'Y. Wu and K. He, “Group Normalization,” in Proc. ECCV, 2018, pp. 3–19.',
 'F. Chollet, “Xception: Deep Learning with Depthwise Separable Convolutions,” in Proc. IEEE CVPR, 2017, pp. 1251–1258.',
 'K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” in Proc. IEEE CVPR, 2016, pp. 770–778.',
 'D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization,” in Proc. ICLR, 2015.',
 'C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On Calibration of Modern Neural Networks,” in Proc. ICML, PMLR 70, 2017, pp. 1321–1330.',
 'S. Ioffe and C. Szegedy, “Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift,” in Proc. ICML, PMLR 37, 2015, pp. 448–456.',
]


def pct(x, d=1):
    return f'{x * 100:.{d}f}'.replace('.', ',')


def num(x, d=3):
    return f'{x:.{d}f}'.replace('.', ',')


def load():
    rows = list(csv.DictReader((OUT / 'dataset/labels.csv').open()))
    manifest = list(csv.DictReader((PREPARED / 'manifest.csv').open()))
    dataset = json.loads((PREPARED / 'dataset.json').read_text())
    runs = {n: json.loads((RUN / n / 'run.json').read_text()) for n in ['regularized', 'baseline_augmented']}
    hist = {n: json.loads((RUN / n / 'history.json').read_text()) for n in runs}
    preds = {n: list(csv.DictReader((RUN / n / 'validation_predictions.csv').open())) for n in runs}
    selective = json.loads((ROOT / 'outputs/assignment_audit/selective_validation_own_v4.json').read_text())['policies']
    assert Counter(r['split'] for r in manifest) == {'train': 70, 'val': 6, 'test': 6}
    assert len(rows) == 229 and all(r['source'] == 'own' for r in rows)
    return rows, manifest, dataset, runs, hist, preds, selective


# ---------------------------------------------------------------- figures
def figures(manifest, runs, hist):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    FIG.mkdir(exist_ok=True)
    for old in FIG.glob('*.png'):
        old.unlink()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = ['#c0392b', '#e09a2c', '#5b3a33']
    colors = {'segar': '#c0392b', 'tidak_segar': '#e09a2c', 'busuk': '#5b3a33'}

    # Gambar 1: distribusi kelas per split
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    x = np.arange(3)
    for i, c in enumerate(CLASSES):
        vals = [sum(r['split'] == s and r['label'] == c for r in manifest) for s in ['train', 'val', 'test']]
        bars = ax.bar(x + (i - 1) * .26, vals, .26, label=c, color=colors[c])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + .6, str(v), ha='center', fontsize=8)
    ax.set_xticks(x, ['train (8 sesi)', 'val (2 sesi)', 'test terkunci (2 sesi)'])
    ax.set_ylabel('Jumlah foto'); ax.set_title('Distribusi kelas per split (82 foto asli berlabel, split per sesi)', fontsize=9)
    ax.legend(fontsize=8, frameon=False); ax.set_ylim(0, 44)
    fig.tight_layout(); fig.savefig(FIG / 'fig1_distribusi.png', dpi=170); plt.close(fig)

    # Gambar 2: contoh foto per kelas dari empat sesi berbeda
    sessions = ['own:a_meja_kayu', 'own:b01_ubin', 'own:b04_granit_hitam', 'own:b05_linen']
    fig, axes = plt.subplots(3, 4, figsize=(6.2, 4.9))
    for r_i, c in enumerate(CLASSES):
        for c_i, g in enumerate(sessions):
            row = next(r for r in manifest if r['label'] == c and r['group_id'] == g)
            with Image.open(ROOT / row['filepath']) as im:
                axes[r_i, c_i].imshow(im.convert('RGB'))
            axes[r_i, c_i].set_title(f"{c} · {g.split(':')[1]} ({row['split']})", fontsize=6)
            axes[r_i, c_i].axis('off')
    fig.tight_layout(); fig.savefig(FIG / 'fig2_contoh.png', dpi=140); plt.close(fig)

    # Gambar 3: kerangka penelitian
    steps = [('1 Pengumpulan', 'Foto tomat + 22 foto\nimpor; catat sumber'), ('2 Kurasi & label', 'segar / tidak_segar /\nbusuk; grup sesi'),
             ('3 QC data', 'SHA-256, hash piksel,\ndHash ≤ 4'), ('4 Split sesi', '70 / 6 / 6; test\nterkunci, split beku'),
             ('5 Preprocessing', 'EXIF → RGB → letterbox\n128×128 → [0,1]'), ('6 Model', 'Baseline CNN vs\nresidual separable CNN'),
             ('7 Training', 'Adam, early stopping,\nsampling seimbang'), ('8 Seleksi', 'Validation macro-F1\n(tie: val loss)'),
             ('9 Evaluasi', 'Validation: akurasi, F1, CM,\nECE; test belum dibuka'), ('10 Deployment', 'Demo browser, CLI,\napp lokal')]
    fig, ax = plt.subplots(figsize=(6.4, 2.9)); ax.axis('off'); ax.set_xlim(0, 5); ax.set_ylim(0, 2)
    for i, (t, d) in enumerate(steps):
        cx, cy = i % 5 + .5, 1.5 - i // 5
        face = '#e6eef8' if i < 5 else '#fbeee3'
        ax.add_patch(plt.Rectangle((cx - .45, cy - .38), .9, .76, fc=face, ec='#5a7da8' if i < 5 else '#c98b4e', lw=.8))
        ax.text(cx, cy + .2, t, ha='center', va='center', fontsize=6.6, fontweight='bold')
        ax.text(cx, cy - .08, d, ha='center', va='center', fontsize=5.4)
        if i % 5 < 4: ax.annotate('', xy=(cx + .56, cy), xytext=(cx + .45, cy), arrowprops=dict(arrowstyle='->', lw=.8))
    fig.tight_layout(); fig.savefig(FIG / 'fig3_kerangka.png', dpi=190); plt.close(fig)

    # Gambar 4: validation loss/accuracy kedua kandidat
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7))
    for n, c in [('baseline_augmented', '#243c34'), ('regularized', '#c0392b')]:
        e = range(1, len(hist[n]['loss']) + 1)
        axes[0].plot(e, hist[n]['val_loss'], label=n, color=c); axes[1].plot(e, hist[n]['val_accuracy'], label=n, color=c)
        be = hist[n]['best_epoch']
        axes[0].axvline(be if be > 0 else 1, color=c, ls=':', lw=.8)
    axes[0].set_title('Validation loss'); axes[1].set_title('Validation accuracy'); axes[0].set_yscale('log')
    for a in axes: a.set_xlabel('Epoch'); a.grid(alpha=.15)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / 'fig4_validation.png', dpi=170); plt.close(fig)

    # Gambar 5: confusion matrix validation dua kandidat
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9))
    for ax, n in zip(axes, ['baseline_augmented', 'regularized']):
        cm = np.array(runs[n]['validation']['confusion_matrix'])
        ax.imshow(cm, cmap='Blues', vmin=0, vmax=2)
        ax.set_xticks(range(3), ['segar', 'tidak\nsegar', 'busuk'], fontsize=7); ax.set_yticks(range(3), ['segar', 'tidak_segar', 'busuk'], fontsize=7)
        for i in range(3):
            for j in range(3): ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=13, color='white' if cm[i, j] else '#555')
        ax.set_xlabel('Prediksi', fontsize=8); ax.set_title(f'{n}: {int(np.trace(cm))}/6 benar', fontsize=8)
    axes[0].set_ylabel('Label', fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / 'fig5_confusion.png', dpi=170); plt.close(fig)

    # Gambar 6: learning curve model terpilih (train vs validation)
    h = hist[SELECTED]; e = range(1, len(h['loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7))
    axes[0].plot(e, h['accuracy'], label='Train (seimbang+augmentasi)'); axes[0].plot(e, h['val_accuracy'], label='Validation (6 foto)')
    axes[1].plot(e, h['loss'], label='Train'); axes[1].plot(e, h['val_loss'], label='Validation'); axes[1].set_yscale('log')
    axes[0].set_title(f'Accuracy ({SELECTED})'); axes[1].set_title('Loss')
    for a in axes: a.set_xlabel('Epoch'); a.grid(alpha=.15); a.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / 'fig6_learning_curve.png', dpi=170); plt.close(fig)


# ---------------------------------------------------------------- docx
BLUE_HEAD, BLUE_SUB, GRID = 'BDD7EE', 'DCE9F5', 'A6A6A6'


def build_docx(blocks_fn):
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt, RGBColor

    d = Document()
    sec = d.sections[0]
    sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)
    sec.left_margin = sec.right_margin = Inches(.85); sec.top_margin = Inches(.9); sec.bottom_margin = Inches(.75)
    sec.header_distance = Inches(.35)
    st = d.styles['Normal']; st.font.name = 'Calibri'; st.font.size = Pt(10)
    st.element.rPr.rFonts.set(qn('w:eastAsia'), 'Calibri')
    st.paragraph_format.space_after = Pt(4); st.paragraph_format.space_before = Pt(0); st.paragraph_format.line_spacing = 1.05

    def shade(cell, fill):
        pr = cell._tc.get_or_add_tcPr(); sh = OxmlElement('w:shd')
        sh.set(qn('w:val'), 'clear'); sh.set(qn('w:color'), 'auto'); sh.set(qn('w:fill'), fill); pr.append(sh)

    def borders(table, color=GRID, sz='6'):
        tblPr = table._tbl.tblPr; b = OxmlElement('w:tblBorders')
        for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
            e = OxmlElement(f'w:{edge}'); e.set(qn('w:val'), 'single'); e.set(qn('w:sz'), sz); e.set(qn('w:color'), color); b.append(e)
        tblPr.append(b)

    def cell_margins(table, top=50, bottom=50, left=90, right=90):
        tblPr = table._tbl.tblPr; m = OxmlElement('w:tblCellMar')
        for k, v in (('top', top), ('left', left), ('bottom', bottom), ('right', right)):
            e = OxmlElement(f'w:{k}'); e.set(qn('w:w'), str(v)); e.set(qn('w:type'), 'dxa'); m.append(e)
        tblPr.append(m)

    ORDER = ['tblStyle', 'tblpPr', 'tblOverlap', 'bidiVisual', 'tblStyleRowBandSize', 'tblStyleColBandSize', 'tblW', 'jc',
             'tblCellSpacing', 'tblInd', 'tblBorders', 'shd', 'tblLayout', 'tblCellMar', 'tblLook']

    def sort_tblpr(tblPr):
        kids = list(tblPr)
        for k in kids: tblPr.remove(k)
        kids.sort(key=lambda e: ORDER.index(e.tag.split('}')[1]) if e.tag.split('}')[1] in ORDER else 99)
        for k in kids: tblPr.append(k)

    def fix_width(t, widths):
        t.autofit = False
        tblPr = t._tbl.tblPr
        for tag in ('w:tblW', 'w:tblLayout'):
            for e in tblPr.findall(qn(tag)): tblPr.remove(e)
        w = OxmlElement('w:tblW'); w.set(qn('w:w'), str(int(sum(widths) * 1440))); w.set(qn('w:type'), 'dxa'); tblPr.append(w)
        lay = OxmlElement('w:tblLayout'); lay.set(qn('w:type'), 'fixed'); tblPr.append(lay)
        for col, wd in zip(t.columns, widths): col.width = Inches(wd)
        for row in t.rows:
            for cell, wd in zip(row.cells, widths): cell.width = Inches(wd)
        sort_tblpr(tblPr)

    def runs(par, text, size=None, color=None, italic=False, bold=False):
        for part in re.split(r'(\*\*.+?\*\*|`.+?`)', text):
            if not part: continue
            if part.startswith('**'): r = par.add_run(part[2:-2]); r.bold = True
            elif part.startswith('`'):
                r = par.add_run(part[1:-1]); r.font.name = 'Consolas'; r._element.rPr.rFonts.set(qn('w:eastAsia'), 'Consolas')
                r.font.size = Pt((size or 10) - 1)
            else: r = par.add_run(part); r.bold = bold or None
            if size and not part.startswith('`'): r.font.size = Pt(size)
            if color: r.font.color.rgb = RGBColor.from_string(color)
            if italic: r.italic = True
        return par

    def put_ppr(pPr, el):
        # pBdr/shd harus berada sebelum spacing/jc menurut skema; sisipkan setelah pStyle/keepNext bila ada
        idx = 0
        for i, k in enumerate(pPr):
            if k.tag.split('}')[1] in ('pStyle', 'keepNext', 'keepLines', 'pageBreakBefore', 'framePr', 'widowControl', 'numPr', 'suppressLineNumbers'): idx = i + 1
        pPr.insert(idx, el)

    # header & footer
    hdr = sec.header
    lines = [('IS 794 DEEP LEARNING FINAL PROJECT REPORT', 14), ('ODD SEMESTER ACADEMIC YEAR 2026/2027', 7), ('INFORMATION SYSTEM STUDY PROGRAM', 7),
             ('FACULTY OF ENGINEERING AND INFORMATICS', 7), ('UNIVERSITAS MULTIMEDIA NUSANTARA', 7)]
    for i, (t, s) in enumerate(lines):
        p = hdr.paragraphs[0] if i == 0 else hdr.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(0)
        r = p.add_run(t); r.bold = True; r.font.size = Pt(s)
    pPr = p._p.get_or_add_pPr(); bd = OxmlElement('w:pBdr'); bt = OxmlElement('w:bottom')
    for k, v in (('val', 'single'), ('sz', '10'), ('space', '4'), ('color', '000000')): bt.set(qn(f'w:{k}'), v)
    bd.append(bt); put_ppr(pPr, bd)
    fp = sec.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT

    def field(par, instr):
        f = OxmlElement('w:fldSimple'); f.set(qn('w:instr'), instr)
        r = OxmlElement('w:r'); rpr = OxmlElement('w:rPr')
        for tag in ('w:b', 'w:i'): rpr.append(OxmlElement(tag))
        sz = OxmlElement('w:sz'); sz.set(qn('w:val'), '16'); rpr.append(sz); r.append(rpr)
        t = OxmlElement('w:t'); t.text = '1'; r.append(t); f.append(r); par._p.append(f)
    for t in ('Halaman ', None, ' dari ', None):
        if t is None: field(fp, 'PAGE' if not fp.runs or fp.runs[-1].text.strip() == 'Halaman' else 'NUMPAGES')
        else:
            r = fp.add_run(t); r.bold = True; r.italic = True; r.font.size = Pt(8)


    class Cell:
        def __init__(self, cell): self.c = cell; self.first = True; self.last_table = False

        def par(self):
            if self.first: self.first = False; return self.c.paragraphs[0]
            return self.c.add_paragraph()

        def p(self, text, align=None, size=None, after=None, italic=False):
            x = self.par(); runs(x, text, size=size, italic=italic)
            if align == 'j': x.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            if after is not None: x.paragraph_format.space_after = Pt(after)
            return x

        def bullets(self, items):
            for it in items:
                x = self.par(); x.paragraph_format.left_indent = Inches(.22); x.paragraph_format.first_line_indent = Inches(-.16)
                runs(x, '• ' + it); x.paragraph_format.space_after = Pt(2)

        def code(self, text):
            for line in text.strip('\n').split('\n'):
                x = self.par(); x.paragraph_format.space_after = Pt(0); x.paragraph_format.line_spacing = 1.0
                r = x.add_run(line or ' '); r.font.name = 'Consolas'; r.font.size = Pt(6.8)
                pPr = x._p.get_or_add_pPr(); sh = OxmlElement('w:shd'); sh.set(qn('w:val'), 'clear'); sh.set(qn('w:fill'), 'F2F2F2'); put_ppr(pPr, sh)
            self.par().paragraph_format.space_after = Pt(0)

        def image(self, name, caption, width=6.0):
            x = self.par(); x.alignment = WD_ALIGN_PARAGRAPH.CENTER; x.paragraph_format.space_after = Pt(1)
            x.add_run().add_picture(str(FIG / name if '/' not in name else ROOT / name), width=Inches(width))
            c = self.c.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
            runs(c, caption, size=8.5, italic=True)

        def table(self, headers, rows, widths=None, size=8.5):
            if self.first:  # tabel tidak boleh jadi elemen pertama sel
                self.first = False; self.c.paragraphs[0].paragraph_format.space_after = Pt(0)
            t = self.c.add_table(rows=1, cols=len(headers)); borders(t, 'A6A6A6', '4'); cell_margins(t, 25, 25, 60, 60)
            for cell, h in zip(t.rows[0].cells, headers):
                shade(cell, BLUE_SUB); cell.paragraphs[0].paragraph_format.space_after = Pt(0)
                runs(cell.paragraphs[0], h, size=size, bold=True)
            hr = t.rows[0]._tr.get_or_add_trPr(); hr.append(OxmlElement('w:tblHeader'))
            for row in rows:
                cells = t.add_row().cells
                for cell, v in zip(cells, row):
                    cell.paragraphs[0].paragraph_format.space_after = Pt(0); runs(cell.paragraphs[0], str(v), size=size)
            if not widths: widths = [6.3 / len(headers)] * len(headers)
            scale = 6.25 / sum(widths)
            fix_width(t, [w * scale for w in widths])
            sp = self.c.add_paragraph(); sp.paragraph_format.space_after = Pt(2)
            for r_ in sp.runs: r_.font.size = Pt(2)
            sp.paragraph_format.line_spacing = Pt(4)

    def block(title, subtitle=None, body=True):
        d.add_paragraph().paragraph_format.space_after = Pt(0) if d.paragraphs[-1:] and len(d.paragraphs) > 1 else Pt(0)
        rows = 1 + (1 if subtitle else 0) + (1 if body else 0)
        t = d.add_table(rows=rows, cols=1); borders(t); cell_margins(t, 60, 60, 110, 110)
        fix_width(t, [6.55])
        h = t.rows[0].cells[0]; shade(h, BLUE_HEAD); h.paragraphs[0].paragraph_format.space_after = Pt(0)
        runs(h.paragraphs[0], title, size=10.5, bold=True)
        h.paragraphs[0].paragraph_format.keep_with_next = True
        idx = 1
        if subtitle:
            s = t.rows[1].cells[0]; shade(s, BLUE_SUB); s.paragraphs[0].paragraph_format.space_after = Pt(0)
            runs(s.paragraphs[0], subtitle, size=9, italic=True); s.paragraphs[0].paragraph_format.keep_with_next = True; idx = 2
        return Cell(t.rows[idx].cells[0]) if body else None

    blocks_fn(block)
    return d


# ---------------------------------------------------------------- content
def make_content(rows, manifest, dataset, runs, hist, preds, selective):
    b, r = runs['baseline_augmented'], runs['regularized']
    sel = runs[SELECTED]
    sv, bv, rv = sel['validation'], b['validation'], r['validation']
    n_pkg = len(rows)
    fingerprint = dataset['fingerprint']
    try:
        commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = ''
    cp = r['train_clean']['confusion_matrix']

    def policy(key, label):
        x = selective[key]
        ok = round(x['accepted_accuracy'] * x['accepted']) if x['accepted'] else 0
        return [label, f"{x['accepted']}/6", pct(x['coverage']) + '%', f"{ok}/{x['accepted']}", str(x['errors_accepted'])]

    def content(block):
        # --- identitas
        c = block('Project of Week:'); c.p('Final Project (Week 13–14)')
        c = block('Group Name and Class:'); c.p('UBUR UBUR IKAN LELE, Class CL')
        c = block('Member (Name / NIM):')
        for m in MEMBERS: c.p(m, after=1)
        c = block('Project Title'); c.p(f'**{TITLE}**')

        # --- background
        c = block('Background (100 – 500 words)', 'Include: problem(s), purpose, method(s)')
        c.p('Tomat adalah komoditas hortikultura bernilai tinggi yang mudah rusak. Penilaian kesegaran di pasar, dapur, dan rantai pasok umumnya '
            'dilakukan secara visual dan manual, sehingga subjektif, tidak konsisten antar penilai, dan sulit diskalakan. Pengamatan permukaan buah '
            '(kekencangan kulit, keriput, lesi pembusukan, jamur) merupakan pola visual yang dapat dipelajari model visi komputer.', 'j')
        c.p('**Masalah.** (1) Belum ada alat sederhana yang menilai kondisi visual satu tomat ke dalam tiga tingkat (segar, tidak segar, busuk). '
            '(2) Penelitian kesegaran buah umumnya memakai dataset publik besar dengan kemungkinan kebocoran antar split, sehingga akurasi tinggi '
            'belum tentu menggambarkan generalisasi [7]. (3) Data foto sendiri jumlahnya kecil dan banyak foto berasal dari sesi/latar yang sama, '
            'sehingga risiko overfitting dan kebocoran sesi (foto dari sesi yang sama muncul di train dan test) menjadi perhatian utama.', 'j')
        c.p('**Tujuan.** Membangun dan mengevaluasi sistem klasifikasi tiga kelas kondisi visual tomat dari foto sendiri dengan protokol yang jujur secara '
            'ilmiah: split berbasis sesi/identitas, tanpa augmentasi pada validation/test, pemilihan model hanya dengan validation, test terkunci, serta '
            'pelaporan keterbatasan. Sistem juga diwujudkan sebagai demo web yang berjalan di browser, CLI prediksi, dan aplikasi lokal.', 'j')
        nb, nr = f'{b["model_parameters"]:,}'.replace(',', '.'), f'{r["model_parameters"]:,}'.replace(',', '.')
        c.p(f'**Metode.** Dua CNN dilatih dari bobot acak tanpa transfer learning: (a) **baseline_augmented**, empat blok Conv–BatchNorm–ReLU–MaxPool '
            f'({nb} parameter) dengan augmentasi online, dan (b) **regularized**, residual separable convolution dengan Group Normalization '
            f'({nr} parameter). Data berupa 82 foto asli berlabel (60 foto lama + 22 foto primer impor berlabel segar), dibagi 70/6/6 '
            'per kelompok sesi; test (6 foto, 2 sesi) dikunci dan belum dievaluasi. Model dipilih dengan validation macro-F1.', 'j')
        c.p(f'**Temuan ringkas.** baseline_augmented terpilih: accuracy validation {pct(bv["accuracy"])}% (6/6) dan macro-F1 {num(bv["macro_f1"])}; '
            f'regularized {pct(rv["accuracy"])}% (4/6), macro-F1 {num(rv["macro_f1"])}. Validation hanya 6 foto dari 2 sesi dan juga memandu early stopping, '
            'sehingga angka ini indikasi awal, bukan bukti generalisasi. Hasil 100% pada test di versi laporan sebelumnya ditarik karena audit menemukan '
            'kebocoran sesi. Model diposisikan sebagai prototipe end-to-end, bukan sistem siap produksi.', 'j')

        # --- literatur
        c = block('Literature Study and Previous Work', 'Include: state-of-the-art, related previous works/articles from most recent international journals (minimum 5)')
        c.p('**State of the art.** Klasifikasi kesegaran buah modern bertumpu pada CNN dan transfer learning, kemudian diperkuat dengan atensi, distilasi '
            'pengetahuan, atau detektor objek. Tabel berikut merangkum karya terkait dan posisinya terhadap proyek ini. Angka performa karya lain sengaja '
            'tidak disalin karena tugas, data, dan protokolnya berbeda.', 'j')
        c.table(['Ref.', 'Karya (tahun)', 'Metode / data', 'Relevansi bagi proyek'], [
            ['[1]', 'Shu dkk., Foods (2025)', 'ResNet-101 + Non-Local Attention untuk klasifikasi dan deteksi kesegaran buah', 'Menegaskan pentingnya fitur tekstur/lesi halus; memakai backbone besar, sedangkan proyek ini melatih CNN kecil dari nol'],
            ['[2]', 'Demirel & Yildiz, Food Analytical Methods (2025)', 'VisDist-Net: distilasi vision transformer ke CNN hibrida ringan', 'Contoh model ringan untuk perangkat terbatas; proyek ini juga menekankan model kecil, tanpa distilasi'],
            ['[3]', 'Mokhtar & Seddik, Robotics, Integration, Manufacturing and Control (2025)', 'YOLO untuk deteksi dan diagnosis kualitas tomat', 'Domain paling dekat (tomat); proyek ini mengklasifikasi satu tomat per foto, bukan deteksi objek'],
            ['[4]', 'Mukhiddinov dkk., Sensors (2022)', 'YOLOv4 yang ditingkatkan untuk kesegaran buah dan sayur (segar vs busuk)', 'Menunjukkan klasifikasi segar/busuk dan perlunya variasi latar; hasil augmentasi tidak dianggap data independen'],
            ['[5]', 'Shorten & Khoshgoftaar, Journal of Big Data (2019)', 'Survei augmentasi citra untuk deep learning', 'Dasar augmentasi ringan: flip, rotasi kecil, translasi, zoom, kontras/brightness'],
            ['[6]', 'Khan dkk., arXiv:2307.01530 (2023/2024, preprint)', 'Convolutional transformer untuk pengenalan kematangan tomat (dataset KUTomaData dan lainnya)', 'Tren transformer pada tomat; fokusnya kematangan, bukan kesegaran'],
            ['[7]', 'Ramos dkk., arXiv:2508.17416 (2025, preprint)', 'Audit kebocoran data pada dataset visual; semua dataset yang dianalisis menunjukkan bentuk kebocoran', 'Motivasi split berbasis kelompok/sesi dan pemeriksaan duplikat'],
        ], widths=[.4, 1.55, 2.1, 2.3])
        c.p('**Dasar teknis arsitektur.** Residual connection [10] memudahkan optimisasi; depthwise separable convolution [9] mengurangi parameter; Group '
            'Normalization [8] tidak bergantung pada statistik batch sehingga cocok untuk batch/data kecil, sementara Batch Normalization [13] dipakai pada '
            'baseline. Optimizer Adam [11]; kalibrasi probabilitas dinilai dengan expected calibration error (ECE) mengikuti [12].', 'j')
        c.p('**Celah penelitian dan kontribusi.** Karya-karya di atas umumnya melaporkan akurasi tinggi pada dataset besar, sering tanpa menjelaskan kontrol '
            'kebocoran antar split [7]. Proyek ini berkontribusi pada sisi protokol: (i) split berbasis sesi/identitas dengan pemeriksaan hash dan dHash, '
            '(ii) model dari nol pada foto sendiri, (iii) pelaporan jumlah kelompok validation/test, status test terkunci, dan keterbatasan secara eksplisit, '
            'serta (iv) demo end-to-end yang berjalan di browser dan CLI.', 'j')

        # --- data
        c = block('Data Acquisition, Preparation and Pre-processing', 'Must include (not limited to): data collection, data curation, data cleaning, data transformation, normalization, etc. (EDA).')
        c.p('**1. Data collection**')
        c.p('Data adalah foto yang dicatat sebagai milik tim (`source=own`), bukan dataset publik. Dataset publik dan data dummy lama tidak dipakai pipeline. '
            'Terdapat dua kumpulan foto asli:', 'j')
        c.bullets([
            '`data/raw/`: **60 foto PNG** 1254×1254 piksel, berlabel dan disetujui, 20 per kelas. Foto dikelompokkan ke 11 sesi pemotretan teraudit '
            '(satu sesi besar A_meja_kayu berisi 30 foto dan sepuluh sesi b01–b10 masing-masing 3 foto). Dokumentasi tanggal/kamera per foto belum lengkap.',
            '`data/reference_import/`: **26 foto JPEG** 1200×1600 dari WhatsApp; **22 unik** berdasarkan SHA-256 (4 salinan identik tidak dihitung). Pada '
            f'{DATE} ke-22 foto unik digabung ke dataset berlabel sebagai `segar` atas arahan pemilik (tidak diperiksa satu per satu), satu grup `R_reference_meja_kayu`.'])
        c.p(f'Total foto: **{n_pkg}**, seluruhnya milik pemilik, diambil sendiri (pernyataan pemilik, 14 Oktober 2026) dan berlabel; sebagian label berasal dari nama berkas dan belum diperiksa per foto. '
            'Eksperimen yang dilaporkan di sini (own_v4_ref22) dijalankan pada 82 foto pertama (60 lama + 22 impor) dan belum dilatih ulang dengan seluruh foto. '
            f'Syarat tugas minimal 100 gambar primer terpenuhi oleh jumlah foto ({n_pkg}); lihat Analysis and Discussion.', 'j')
        c.p('**2. Data curation**')
        c.p('Setiap foto dilabeli manual berdasarkan kriteria tertulis: **segar** (kulit kencang, utuh, tanpa pembusukan), **tidak segar** (keriput/layu tanpa '
            'pembusukan nyata), **busuk** (lesi/pembusukan atau jamur jelas). Warna merah/hijau saja bukan dasar label (kesegaran ≠ kematangan). '
            'Setiap foto diberi `group_id` yang menggabungkan ID buah dan sesi secara konservatif; foto ambigu tidak disetujui. Label tersimpan di '
            '`data/annotations.csv` dan `submission/dataset/labels.csv` dengan flag approved. Label satu penilai; tidak ada pengukuran kesepakatan antar penilai.', 'j')
        c.p('**3. Data cleaning dan quality control**')
        c.bullets([
            'Pemeriksaan duplikat exact (SHA-256), duplikat piksel (hash piksel), dan near-duplicate (dHash 64-bit, jarak ≤ 4). Hasil: 4 salinan identik pada '
            'folder referensi dikecualikan; tidak ada near-duplicate merge pada 82 foto (`near_duplicate_merges` kosong). dHash tidak dapat membuktikan identitas buah.',
            'Gambar animasi, > 25 megapiksel, atau format tidak didukung ditolak; orientasi EXIF dikoreksi.',
            'Pipeline menolak dataset bila label hilang, kelas tidak lengkap, atau setiap split tidak memuat ketiga kelas.',
            'Audit sesi: pengelompokan lama per “buah” set01–set10 diganti pengelompokan per sesi teraudit karena foto dari sesi/latar yang sama tersebar '
            'di beberapa split pada versi sebelumnya.'])
        c.p('**4. Exploratory Data Analysis (EDA)**')
        c.table(['Aspek', 'Temuan'], [
            ['Jumlah foto berlabel', '82 (segar 42, tidak_segar 20, busuk 20). Kelas segar berlebih karena 22 foto impor seluruhnya segar.'],
            ['Kelompok/sesi', '12 grup: 11 sesi foto lama + 1 grup foto impor. Sesi A_meja_kayu berisi 30 foto (10 per kelas); sesi lain 3 foto (1 per kelas); grup impor 22 foto segar.'],
            ['Resolusi', 'Foto lama 1254×1254 PNG RGB; foto impor 1200×1600 JPEG.'],
            [f'Split beku (fingerprint {fingerprint[:8]}…)', 'Train 70 foto / 8 grup (segar 38, tidak_segar 16, busuk 16); Val 6 foto / 2 grup (2/2/2); Test terkunci 6 foto / 2 grup (2/2/2).'],
            ['Near-duplicate', 'Tidak ada (dHash ≤ 4)'],
            ['Risiko latar', '22 dari 38 foto segar di training berasal dari satu latar (meja kayu), sehingga model dapat mengaitkan latar dengan kelas.'],
        ], widths=[1.9, 4.5])
        c.image('fig1_distribusi.png', 'Gambar 1. Distribusi kelas pada train/validation/test. [data: data/prepared/ff822adef8db3ae5/manifest.csv]', 5.4)
        c.image('fig2_contoh.png', 'Gambar 2. Contoh foto tiap kelas dari empat sesi berbeda (split ditulis di judul).', 5.6)
        c.p('**Catatan EDA.** Dataset kecil dan tidak seimbang di training. Validation dan test masing-masing hanya 2 sesi dan 2 foto per kelas. Sesi '
            'difoto pada latar yang sama sehingga model dapat mempelajari ciri sesi, bukan hanya ciri kesegaran; risiko ini dibahas pada bagian Analysis.', 'j')
        c.p('**5. Data transformation dan normalization**')
        c.p('Satu loader dipakai identik saat training, evaluasi, dan inferensi (`src/preprocessing.py`): koreksi EXIF → konversi RGB → **letterbox** ke 128×128 '
            '(aspek rasio dipertahankan, padding abu-abu, bilinear) → normalisasi piksel ke [0,1] (÷255).', 'j')
        c.bullets([
            '**Augmentasi training-only:** flip horizontal, rotasi ±0,06 putaran, translasi 7%, zoom, kontras, dan brightness ringan; aktif pada kedua kandidat '
            'saat training. Validation/test tidak diaugmentasi.',
            '**Sampling seimbang:** tiap epoch memakai 120 undian per kelas (360 undian) dari 70 foto train; pengulangan bukan data baru.'])

        # --- metodologi
        c = block('Methodology', 'Add research framework, and explain in details')
        c.p('**Kerangka penelitian.** Alur kerja terdiri atas sepuluh tahap dari pengumpulan data hingga deployment (Gambar 3), diimplementasikan sebagai '
            'pipeline yang dapat diulang (`python -m src.train`, dataset beku di `data/prepared/ff822adef8db3ae5`).', 'j')
        c.image('fig3_kerangka.png', 'Gambar 3. Research framework Tomato Vision.', 6.0)
        c.p('**A. Arsitektur model (dari bobot acak, tanpa pretrained)**')
        c.table(['Aspek', 'baseline_augmented', 'regularized'], [
            ['Blok utama', '4× Conv2D (32/64/128/256) + BatchNorm + ReLU + MaxPool', 'Conv 24 (stride 2) + 3 blok residual separable conv (48/96/160)'],
            ['Normalisasi', 'Batch Normalization (momentum 0,9)', 'Group Normalization (8 grup)'],
            ['Head', 'GAP → Dense 128 → Dropout 0,5 → softmax 3', 'GAP → Dense 64 → Dropout 0,35 → softmax 3'],
            ['Regularisasi', 'Dropout; augmentasi online', 'L2 1e-4, spatial dropout, dropout, augmentasi di dalam model'],
            ['Parameter', f'{b["model_parameters"]:,}'.replace(',', '.'), f'{r["model_parameters"]:,}'.replace(',', '.')],
        ], widths=[1.2, 2.6, 2.6])
        c.p('**B. Pelatihan.** Adam, sparse categorical cross-entropy, learning rate 0,001, batch 16, seed 42 dengan operasi deterministik, 120 undian/kelas per epoch. '
            'Maksimum epoch 30 untuk baseline_augmented dan 20 untuk regularized (anggaran tidak setara). Early stopping memantau validation loss (patience 10); '
            'learning rate diturunkan saat plateau; bobot epoch terbaik dipulihkan.', 'j')
        c.p('**C. Skenario eksperimen (own_v4_ref22)**')
        c.table(['Kandidat', 'Arsitektur', 'Learning rate', 'Dropout', 'Epoch maks.', 'Augmentasi'], [
            ['baseline_augmented', 'Baseline', '0,001', '0,5 (tetap)', '30', 'Online'],
            ['regularized', 'Regularized', '0,001', '0,35', '20', 'Online (dalam model)'],
        ], widths=[1.5, 1.1, 1.0, 1.0, 0.9, 0.9])
        c.p('**D. Seleksi dan evaluasi.** Kandidat dipilih dengan validation macro-F1, tie-break validation loss. Metrik: accuracy, macro precision/recall/F1, '
            'confusion matrix, loss, ECE, serta prediksi selektif (ambang 0,7 tetap, tidak di-fit). Temperature scaling hanya dipasang dengan validation bila ≥ 20 '
            'gambar per kelas dan ≥ 5 kelompok; pada data ini tidak terpenuhi sehingga softmax mentah dipakai. Karena arsitektur, normalisasi, regularisasi, dan '
            'anggaran epoch berbeda sekaligus, perbandingan ini tidak mengisolasi pengaruh satu komponen.', 'j')
        c.p('**E. Pengendalian kebocoran data.** Split dibuat per kelompok sesi, bukan per foto, sehingga seluruh foto satu sesi berada pada satu split. '
            'Test terkunci disimpan sebagai daftar SHA-256 dan tidak dibuka dalam eksperimen ini; fingerprint dataset disimpan pada setiap run agar hasil dapat ditelusuri. '
            'Hyperparameter tidak dipilih berdasarkan test.', 'j')

        # --- results
        c = block('Results', 'Add screenshot of codes and results, or you can add notes like [the result refers to no. 1 in IPYNB “project_code”]')
        c.p('Hasil di bawah berasal dari run `outputs/experiments/own_v4_ref22/` (dua kandidat, split 70/6/6). Rujukan: `selection.json`, `*/run.json`, `*/history.json`, '
            '`*/validation_predictions.csv`, dan `summary.md`. Kode berupa modul Python di `src/` (notebook proyek memuat artefak versi sebelumnya).', 'j')
        c.p('**1. Kode utama (src/model.py, ringkas)**')
        c.code('''if c.architecture == "baseline":
    if c.augment: x = make_augmentation(c.seed)(x)      # aktif hanya saat training
    for filters in (32, 64, 128, 256):
        x = L.Conv2D(filters, 3, padding="same")(x)
        x = L.BatchNormalization(momentum=c.bn_momentum)(x); x = L.Activation("relu")(x)
        x = L.MaxPooling2D()(x)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dense(128, activation="relu")(x); x = L.Dropout(.5)(x)
else:
    x = make_augmentation(c.seed)(x)                    # aktif hanya saat training
    x = L.Conv2D(24, 3, strides=2, padding="same", use_bias=False)(x)
    for filters in (48, 96, 160):                       # blok residual separable
        residual = L.Conv2D(filters, 1, strides=2, padding="same", use_bias=False)(x)
        x = L.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)
        x = L.GroupNormalization(groups=8)(x); x = L.Activation("relu")(x)
        x = L.SeparableConv2D(filters, 3, strides=2, padding="same", use_bias=False)(x)
        x = L.Add()([L.GroupNormalization(groups=8)(x), residual])
    x = L.GlobalAveragePooling2D()(x); x = L.Dense(64, activation="relu")(x); x = L.Dropout(c.dropout)(x)
outputs = L.Dense(3, activation="softmax")(x)''')
        c.p('**2. Perbandingan kandidat pada validation (6 foto, 2 sesi)**')
        def cand(n, x): return [n, f'{x["model_parameters"]:,}'.replace(',', '.'), f'{x["epochs_run"]} / {x["best_epoch"]}', pct(x['train_clean']['accuracy']) + '%',
                              num(x['validation']['accuracy']), num(x['validation']['macro_f1']), num(x['validation']['loss'], 4), num(x['validation']['expected_calibration_error'], 4),
                              f'{x["elapsed_seconds"]:.0f} s']
        c.table(['Kandidat', 'Param.', 'Epoch / terbaik', 'Train acc', 'Val acc', 'Val macro-F1', 'Val loss', 'ECE', 'Waktu'],
                [cand('baseline_augmented', b), cand('regularized', r)], widths=[1.3, .6, .75, .6, .55, .75, .6, .55, .5], size=8)
        c.p(f'**baseline_augmented terpilih** (validation macro-F1 {num(bv["macro_f1"])} dibanding {num(rv["macro_f1"])}; epoch terbaik {b["best_epoch"]} dari {b["epochs_run"]}). '
            'Sebagai pembanding, run sebelumnya (own_v3, tanpa 22 foto impor) memberi 4/6 dan macro-F1 0,656 pada kedua kandidat.', 'j')
        c.image('fig4_validation.png', 'Gambar 4. Validation loss (skala log) dan accuracy kedua kandidat; garis titik = epoch terbaik. [data: */history.json]', 5.9)
        c.p('**3. Hasil model terpilih pada validation (baseline_augmented)**')
        c.table(['Metrik', 'Nilai'], [
            ['Accuracy', f'{num(sv["accuracy"])} ({round(sv["accuracy"]*6)}/6 benar)'],
            ['Macro precision / recall / F1', f'{num(sv["macro_precision"])} / {num(sv["macro_recall"])} / {num(sv["macro_f1"])}'],
            ['Validation loss (cross-entropy)', num(sv['loss'], 4)],
            ['Expected calibration error', f'{num(sv["expected_calibration_error"], 4)} (6 sampel; tidak stabil)'],
            ['Kelompok pada validation', '2 sesi (b05_linen, b08_beton)'],
            ['Train accuracy (bersih, 70 foto)', f'{num(sel["train_clean"]["accuracy"])} (37/38 segar benar; 1 segar → tidak_segar)'],
            ['Test terkunci', 'Belum dievaluasi (6 foto, 2 sesi)'],
        ], widths=[2.4, 4.0])
        c.image('fig5_confusion.png', 'Gambar 5. Confusion matrix validation: baseline_augmented (terpilih) dan regularized. Bukan hasil test.', 5.9)
        rows_pred = []
        for p_ in preds[SELECTED]:
            pr = [float(p_['prob_segar']), float(p_['prob_tidak_segar']), float(p_['prob_busuk'])]
            k = max(range(3), key=lambda i: pr[i])
            rows_pred.append([Path(p_['filepath']).name, p_['group_id'].split(':')[1], p_['label'], CLASSES[k], pct(pr[k]) + '%', '✓' if CLASSES[k] == p_['label'] else '✗'])
        c.table(['Foto validation', 'Sesi', 'Label', 'Prediksi', 'Probabilitas', 'Benar'], rows_pred, widths=[1.7, 1.1, 1.0, 1.0, 1.0, .6])
        c.p('**4. Kandidat regularized (pembanding)**')
        c.p(f'Pada validation regularized benar 4/6: kedua foto segar diprediksi tidak_segar (precision/recall segar 0,00/0,00; tidak_segar 0,50/1,00; busuk 1,00/1,00). '
            f'Pada data training sendiri ia memprediksi {cp[0][1]} dari {sum(cp[0])} foto segar sebagai tidak_segar (train accuracy {pct(r["train_clean"]["accuracy"])}%), '
            'tanda underfitting pada kelas segar; epoch terbaiknya adalah epoch terakhir sehingga anggaran 20 epoch mungkin membatasinya.', 'j')
        c.image('fig6_learning_curve.png', 'Gambar 6. Learning curve baseline_augmented. [data: baseline_augmented/history.json]', 5.9)

        # --- evaluation
        c = block('Evaluation')
        c.p('**Protokol.** Validation dipakai untuk memilih kandidat dan juga untuk early stopping; test terkunci tidak dibuka. Tidak ada penyetelan berdasarkan test. '
            'Karena itu estimasi pada validation cenderung optimistis dan bukan estimasi generalisasi.', 'j')
        c.bullets([
            f'**Hasil utama.** baseline_augmented: accuracy {num(bv["accuracy"])} dan macro-F1 {num(bv["macro_f1"])} pada validation; regularized: accuracy {num(rv["accuracy"])}, macro-F1 {num(rv["macro_f1"])}.',
            '**Stabilitas.** Validation accuracy baseline_augmented berosilasi antar epoch (mis. 0,50 → 0,83 → 0,67 pada epoch 17–23) dan loss validation melonjak '
            'pada beberapa epoch; epoch terbaik dipilih dengan validation loss. Angka 6/6 sangat bergantung pada pemilihan epoch dan enam foto tersebut.',
            f'**Kalibrasi.** ECE validation {num(bv["expected_calibration_error"])} (baseline_augmented) dan {num(rv["expected_calibration_error"])} (regularized). '
            'Temperature scaling tidak dipasang karena validation terlalu kecil; probabilitas softmax bukan peluang yang terkalibrasi.',
            '**Keterbatasan evaluasi.** Satu foto salah mengubah accuracy sebesar 16,7 poin. Perubahan 4/6 (own_v3) menjadi 6/6 setara dua foto dan berasal dari '
            'perubahan data training sekaligus pergantian kandidat terpilih, sehingga tidak dapat dikaitkan dengan 22 foto baru; bukan bukti peningkatan.',
            '**Uji robustness.** Uji perubahan pencahayaan dan blur dari versi laporan sebelumnya dilakukan pada model lama yang ditarik, sehingga angkanya tidak dibawa. '
            'Uji tersebut belum diulang pada own_v4_ref22.'])
        c.p('**Prediksi selektif pada validation (model terpilih; ambang 0,7 tetap).**')
        c.table(['Kebijakan', 'Diterima', 'Coverage', 'Benar dari diterima', 'Salah lolos'], [
            policy('classify_all', 'Semua prediksi'), policy('confidence_only_fixed', 'Confidence ≥ 0,7'),
            policy('confidence_quality_consistency_fixed', 'Confidence + kualitas + konsistensi')], widths=[2.4, .8, .8, 1.5, .9])
        c.p('Karena model terpilih tidak membuat kesalahan pada validation, aturan selektif hanya menurunkan coverage tanpa manfaat terukur; manfaatnya baru dapat dinilai '
            'pada data yang memuat kesalahan nyata.', 'j')
        c.bullets([
            '**Pengujian perangkat lunak.** Tes otomatis (`tests/`) mencakup kebocoran data, format gambar, label/indeks, sampling, prediksi selektif, ekspor model web, '
            'paritas JavaScript–Keras, CLI prediksi, impor, revisi anotasi, serta tes HTTP dan browser (desktop dan ponsel).'])

        # --- deployment
        c = block('Deployment (if any)')
        c.p('Sistem tersedia dalam tiga bentuk:', after=2)
        c.bullets([
            '**Demo web di browser** (GitHub Pages): https://sellebeww.github.io/tomato-vision/ . Bobot baseline_augmented diekspor ke JavaScript sehingga foto diproses '
            'sepenuhnya di browser dan tidak diunggah. Menu: Prediksi, Dataset, Training, dan Evaluasi. Paritas diuji: selisih maksimum keluaran jaringan JavaScript vs Keras '
            '8,3e-07 pada 13 kasus uji (batas 1e-04).',
            '**CLI TensorFlow:** `python -m src.predict foto.jpg` (satu/banyak foto, keluaran teks atau JSON) dan notebook Colab (belum diuji di Colab).',
            '**Aplikasi lokal:** `python -m src.app` (http://127.0.0.1:7860) untuk pelabelan, training, dan evaluasi; hanya bind ke localhost, memeriksa Host dan token sesi, '
            'membatasi ukuran upload, dan memakai Content Security Policy.'])
        c.p('Keluaran: dugaan kelas, probabilitas tiga kelas, peringatan kualitas foto, penanda **perlu tinjauan** (model ragu, hasil berubah pada variasi ringan, atau kualitas rendah), '
            'dan unduhan JSON. **Bukan layanan internet publik dengan jaminan;** model tidak memiliki kelas non-tomat sehingga tidak boleh dipakai sebagai detektor umum dan bukan penilaian keamanan pangan.', 'j')
        c.image('docs/screenshots/demo-2-prediksi.png', 'Gambar 7. Demo web: menu Prediksi. [docs/screenshots/demo-2-prediksi.png]', 5.6)
        c.image('docs/screenshots/demo-3-dataset.png', 'Gambar 8. Demo web: menu Dataset. [docs/screenshots/demo-3-dataset.png]', 5.6)
        c.image('docs/screenshots/demo-4-evaluasi.png', 'Gambar 9. Demo web: menu Evaluasi. [docs/screenshots/demo-4-evaluasi.png]', 5.6)
        c.p('**Reproduksibilitas.** Python 3.11.9, TensorFlow 2.16.1, Keras 3.15.1; `requirements.txt` dan lock macOS arm64. Kode: https://github.com/sellebeww/tomato-vision'
            + (f' (commit {commit} saat laporan dibuat; hasil own_v4_ref22 dihasilkan pada {DATE}).' if commit else '.'), 'j')

        # --- analysis
        c = block('Analysis and Discussion', 'Give your team depth analysis of the results, you may compare or benchmark with previous project/research.')
        c.p('**1. Mengapa baseline_augmented terpilih padahal lebih besar?**')
        c.p(f'baseline_augmented ({nb} parameter) benar 6/6 pada validation, sedangkan regularized ({nr} parameter) 4/6. ' +
            'Dugaan: dengan 70 foto train dari 8 sesi (22 di antaranya satu latar), model kecil dengan augmentasi, L2, dan dropout kuat underfit (train accuracy 75,7%, '
            'epoch terbaik = epoch terakhir), sedangkan baseline menangkap ciri warna/tekstur dan latar. Ini hipotesis; perbandingan tidak setara (anggaran epoch 30 vs 20, '
            'arsitektur dan normalisasi berbeda sekaligus) dan belum diuji dengan ablation terkontrol.', 'j')
        c.p('**2. Seberapa dapat dipercaya angka 100% pada validation?**')
        c.p('Kurang. Hanya 6 foto dari 2 sesi, validation juga dipakai untuk early stopping dan seleksi, dan akurasi antar epoch berosilasi. Tidak ada kesalahan sehingga '
            'tidak ada contoh kegagalan yang dapat dianalisis. Model mungkin memanfaatkan petunjuk sesi (latar, pencahayaan) selain kesegaran. Test terkunci belum dibuka, '
            'jadi generalisasi belum dibuktikan.', 'j')
        c.p('**3. Koreksi terhadap laporan versi sebelumnya**')
        c.p('Versi sebelumnya melaporkan accuracy dan macro-F1 1,00 pada test (12 foto) dengan split “per buah” set01–set10. Audit menemukan bahwa pengelompokan itu tidak '
            'mencerminkan sesi pemotretan: foto dari sesi/latar meja kayu yang sama tersebar di train, validation, dan test. Hasil tersebut kebocoran sesi dan **ditarik**; '
            'laporan ini memakai split sesi teraudit (70/6/6) dan angka validation di atas. Klaim sebelumnya bahwa tiap buah muncul di ketiga kelas dan terdapat 10 buah '
            'independen juga tidak dipertahankan karena identitas buah dalam satu sesi tidak pasti.', 'j')
        c.p('**4. Label impor dan risiko latar**')
        c.p('Ke-22 foto impor diberi label segar sekaligus oleh pemilik tanpa penilaian per foto. Pemeriksaan visual sebelumnya mencatat banyak buah tampak mulus, tetapi bila '
            'sebagian mulai layu, label itu mengajarkan kesalahan. Sebelum digabung, model lama (own_v3) memprediksi 18 dari 22 foto sebagai busuk dan 4 sebagai tidak_segar '
            '(tanpa satu pun segar), menandakan pergeseran domain (latar ramai, tomat kecil dalam bingkai); penyebabnya belum dibuktikan. Kini foto itu masuk training sehingga '
            'tidak dapat lagi dipakai menilai generalisasi, dan segar menjadi 38 dari 70 foto training dengan 22 dari satu latar.', 'j')
        c.p('**5. Perbandingan dengan penelitian terdahulu**')
        c.p('Karya [1]–[4] memakai tugas, data, dan protokol berbeda (biner fresh/rotten, banyak jenis buah, backbone pretrained atau detektor objek) pada dataset yang jauh lebih besar, '
            'sehingga angka mereka tidak dapat dibandingkan langsung dengan 6 foto validation. Perbedaan proyek ini ada pada tiga tingkat kesegaran satu jenis buah, pelatihan dari nol, '
            'dan pelaporan eksplisit jumlah kelompok serta status test. Klaim keunggulan atas karya lain tidak dapat dibuat dari data ini.', 'j')
        c.p('**6. Keterbatasan dan rencana**')
        c.bullets([
            f'Hasil model dilaporkan pada 82 foto; dataset kini {n_pkg} foto berlabel (sebagian label dari nama berkas, belum diperiksa per foto) dan model belum dilatih ulang. Asal pemotretan tiap foto perlu didokumentasikan.',
            'Hanya 12 sesi; validation dan test masing-masing 2 sesi; label satu penilai; kelas tidak_segar berbatasan dengan segar dan busuk; test terkunci belum dinilai.',
            'Langkah lanjut: verifikasi kondisi 22 foto impor satu per satu; tambah foto tidak_segar/busuk pada latar yang sama dan buah/latar/kamera/pencahayaan baru hingga '
            'jauh di atas 100 foto primer; cross-validation berkelompok; ablation (augmentasi, normalisasi, anggaran epoch); uji robustness blur/pencahayaan pada model terkini; '
            'satu evaluasi test setelah protokol dan model dibekukan; penilaian label oleh dua orang.'])
        c.p('**Kesimpulan.** Prototipe end-to-end berfungsi dan protokol evaluasinya mengendalikan kebocoran antar sesi. Hasil kuantitatif (6/6 pada validation) belum membuktikan '
            'generalisasi; nilai utama proyek ada pada pipeline yang dapat diaudit, bukan pada angka akurasi.', 'j')

        # --- references
        c = block('References')
        for i, ref in enumerate(REFERENCES, 1):
            c.p(f'[{i}] {ref}', size=9, after=3)

        # --- software and reference notes
        c = block('Perangkat lunak dan catatan rujukan')
        c.p('Rujukan [6]–[7] diperiksa di arXiv pada 9 Oktober 2026; [1]–[5] mengikuti verifikasi halaman penerbit pada audit proyek 7 Oktober 2026; '
            'rujukan [8]–[13] adalah literatur klasik yang belum diperiksa ulang terhadap sumbernya. Seluruh angka pada laporan dibaca dari file hasil run '
            '(`run.json`, `selection.json`, `history.json`, `validation_predictions.csv`, `manifest.csv`).', 'j')
        c.p('**Library/framework:** TensorFlow/Keras, NumPy, pandas, scikit-learn, Pillow, Matplotlib, python-docx.', 'j')
        c.p('Tim bertanggung jawab atas pengumpulan foto, pelabelan, serta peninjauan seluruh isi laporan.', 'j')

    return content


def main():
    rows, manifest, dataset, runs, hist, preds, selective = load()
    figures(manifest, runs, hist)
    d = build_docx(make_content(rows, manifest, dataset, runs, hist, preds, selective))
    d.save(TARGET)
    print('Saved', TARGET)


if __name__ == '__main__':
    main()
