"""Build IS794 Word reports and slides from existing, auditable experiment artifacts.
Does not train models, relabel data, open the locked test, or claim submission readiness.
"""
import csv
import json
import re
from pathlib import Path
import shutil
import sys
import os
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'submission'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/.matplotlib'))
RUN = ROOT / 'outputs/experiments/own_v4_ref22'
PREPARED = ROOT / 'data/prepared/ff822adef8db3ae5'
SELECTED = json.loads((RUN / 'selection.json').read_text())['selected'].rsplit('/', 1)[1]
CLASSES = ['segar', 'tidak_segar', 'busuk']
REFERENCES = [
 ('Shu, Y., Zhang, J., Wang, Y., & Wei, Y. (2025). Fruit Freshness Classification and Detection Based on the ResNet-101 Network and Non-Local Attention Mechanism. Foods, 14, 1987.', 'https://www.mdpi.com/2304-8158/14/11/1987', 'ResNet-101 dan attention untuk kesegaran buah; mendukung perhatian pada kerusakan permukaan. Data dan protokol berbeda, sehingga bukan pembanding angka langsung.'),
 ('Demirel, S., & Yıldız, O. (2025). VisDist-Net: A New Lightweight Model for Fruit Freshness Classification. Food Analytical Methods, 18, 229–244.', 'https://link.springer.com/article/10.1007/s12161-024-02716-4', 'Distilasi vision transformer ke CNN hibrida. Relevan untuk efisiensi model; proyek ini memakai CNN dari nol dan tidak melakukan distilasi.'),
 ('Mokhtar, Y. A., & Seddik, E. H. (2025). Real-time object detection and diagnosis of tomato quality using YOLO. Robotics: Integration, Manufacturing and Control, 2(2).', 'https://apc.aast.edu/ojs/index.php/RIMC/article/view/RIMC.2025.02.2.1846', 'Deteksi kualitas tomat dengan YOLO. Domain sama, tetapi deteksi objek berbeda dari klasifikasi satu tomat per gambar.'),
 ('Mukhiddinov, M., Muminov, A., & Cho, J. (2022). Improved Classification Approach for Fruits and Vegetables Freshness Based on Deep Learning. Sensors, 22, 8192.', 'https://www.mdpi.com/1424-8220/22/21/8192', 'YOLOv4 yang ditingkatkan untuk berbagai buah dan sayur, termasuk kategori segar/busuk. Menunjukkan kebutuhan variasi latar dan pencahayaan.'),
 ('Amin, U., Shahzad, M. I., Shahzad, A., Shahzad, M., Khan, U., & Mahmood, Z. (2023). Automatic Fruits Freshness Classification Using CNN and Transfer Learning. Applied Sciences, 13, 8087.', 'https://www.mdpi.com/2076-3417/13/14/8087', 'Fine-tuning AlexNet untuk kesegaran buah pada dataset publik. Transfer learning dapat menjadi eksperimen lanjutan, bukan metode yang telah diuji di proyek ini.'),
 ('Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on Image Data Augmentation for Deep Learning. Journal of Big Data, 6, 60.', 'https://link.springer.com/article/10.1186/s40537-019-0197-0', 'Dasar augmentasi geometris dan warna. Transformasi menambah variasi training, tetapi bukan pengamatan buah independen.'),
 ('Wu, Y., & He, K. (2018). Group Normalization. ECCV, pp. 3–19.', 'https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.php', 'Normalisasi per kelompok kanal tidak memakai statistik antar batch; dasar pemilihan normalisasi pada CNN regularized.'),
 ('He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR, pp. 770–778.', 'https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html', 'Shortcut residual mendasari penjumlahan jalur utama dan jalur pintas pada arsitektur proyek.')]


def load_facts():
    rows = list(csv.DictReader((OUT / 'dataset/labels.csv').open()))
    manifest = list(csv.DictReader((PREPARED / 'manifest.csv').open()))
    runs = {name: json.loads((RUN / name / 'run.json').read_text()) for name in ['regularized', 'baseline_augmented']}
    assert len(rows) >= 229 and len({r['sha256'] for r in rows}) == len(rows)
    assert Counter(r['split'] for r in manifest) == {'train':70, 'val':6, 'test':6}
    assert all(r['validation']['n_images'] == 6 for r in runs.values())
    return rows, manifest, runs


def figures(manifest, runs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    folder = OUT / 'figures'; folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10, 'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(1,2,figsize=(10,3.5))
    inventory = list(csv.DictReader((OUT / 'dataset/labels.csv').open()))
    synthetic_count = sum(r['source'] == 'synthetic_ai' for r in inventory)
    axes[0].bar(['Foto lama','Foto primer impor','Sintetis'],[60,22,synthetic_count],color=['#237c69','#7d8990','#cf8e31'])
    axes[0].set_title(f'{len(inventory)} gambar dalam paket'); axes[0].set_ylabel('Jumlah gambar')
    x=np.arange(3)
    for i,c in enumerate(CLASSES):
        values=[sum(r['split']==s and r['label']==c for r in manifest) for s in ['train','val','test']]
        axes[1].bar(x+(i-1)*.24,values,.24,label=c,color=['#237c69','#cf8e31','#bb4c47'][i])
    axes[1].set_xticks(x,['Training','Validation','Test terkunci']); axes[1].set_title('Split 82 foto asli berlabel'); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(folder/'eda.png',dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for name in runs:
        h=json.loads((RUN/name/'history.json').read_text())
        e=range(1,len(h['loss'])+1)
        axes[0].plot(e,h['val_loss'],label=name)
        axes[1].plot(e,h['val_accuracy'],label=name)
    for ax in axes: ax.set_xlabel('Epoch'); ax.legend(fontsize=8); ax.grid(alpha=.15)
    axes[0].set_title('Validation loss'); axes[1].set_title('Validation accuracy')
    fig.tight_layout(); fig.savefig(folder/'learning_curves.png',dpi=180); plt.close(fig)
    cm=np.array(runs[SELECTED]['validation']['confusion_matrix'])
    fig,ax=plt.subplots(figsize=(5,3.8)); ax.imshow(cm,cmap='Greens',vmin=0,vmax=2)
    ax.set_xticks(range(3),CLASSES); ax.set_yticks(range(3),CLASSES)
    for i in range(3):
        for j in range(3): ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=18)
    ax.set_xlabel('Prediksi'); ax.set_ylabel('Label anotasi'); ax.set_title(f'{SELECTED}: validation {int(np.trace(cm))}/6 benar, dua sesi',fontsize=9)
    fig.tight_layout(); fig.savefig(folder/'validation_confusion.png',dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(9,3))
    for ax,c in zip(axes,CLASSES):
        row=next(r for r in manifest if r['split']=='train' and r['label']==c)
        with Image.open(ROOT/row['filepath']) as im: ax.imshow(im)
        ax.set_title(c); ax.axis('off')
    fig.tight_layout(); fig.savefig(folder/'training_examples.png',dpi=140); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,3)); ax.axis('off')
    steps=['Akuisisi +\ncatat sumber','Kurasi +\nlabel + grup','Split sesi\n+ cek hash','CNN +\nvalidation','Analisis +\naplikasi']
    for i,t in enumerate(steps):
        ax.text(.1+i*.2,.62,t,ha='center',va='center',transform=ax.transAxes,bbox=dict(boxstyle='round,pad=.65',fc='#e7f1eb',ec='#237c69'))
        if i<4: ax.annotate('',xy=(.165+i*.2,.62),xytext=(.235+i*.2,.62),xycoords='axes fraction',arrowprops=dict(arrowstyle='<-',color='#237c69'))
    ax.text(.5,.14,'Test terkunci hanya dibuka setelah protokol dan model final.\nSintetis / label belum terkonfirmasi tidak masuk evaluasi sumber.',ha='center',transform=ax.transAxes)
    fig.tight_layout(); fig.savefig(folder/'framework.png',dpi=180); plt.close(fig)
    selective=json.loads((ROOT/'outputs/assignment_audit/selective_validation_own_v4.json').read_text())
    curve=selective['risk_coverage_curve']
    fig,ax=plt.subplots(figsize=(7,3.5))
    valid=[r for r in curve if r['selective_risk'] is not None]
    ax.plot([r['coverage'] for r in valid],[r['selective_risk'] for r in valid],'o-',color='#237c69')
    ax.set(xlabel='Coverage (proporsi prediksi diterima)',ylabel='Selective risk (proporsi salah)',title='Validation: trade-off coverage dan kesalahan',xlim=(0,1.05),ylim=(0,.55))
    ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(folder/'risk_coverage.png',dpi=180); plt.close(fig)


def content(runs):
    r=runs['regularized']; b=runs['baseline_augmented']; sel=runs[SELECTED]; v=sel['validation']
    inventory=list(csv.DictReader((OUT/'dataset/labels.csv').open()))
    n_total=len(inventory); n_syn=sum(x['source']=='synthetic_ai' for x in inventory)
    fingerprint=json.loads((PREPARED/'dataset.json').read_text())['fingerprint']
    correct=round(v['accuracy']*6); pct=lambda x:f'{x*100:.1f}'.replace('.',',')
    selective=json.loads((ROOT/'outputs/assignment_audit/selective_validation_own_v4.json').read_text())['policies']
    def policy_row(name,key):
        x=selective[key]; ok=round(x['accepted_accuracy']*x['accepted']) if x['accepted'] else 0
        return [name,f"{x['accepted']}/6",pct(x['coverage'])+'%',f"{ok}/{x['accepted']} ({pct(x['accepted_accuracy'])}%)" if x['accepted'] else 'tidak terdefinisi',str(x['errors_accepted'])]
    sections=[]
    def section(title,*items): sections.append((title,list(items)))
    def p(t): return ('p',t)
    def table(headers,rows): return ('table',headers,rows)
    def image(name,caption): return ('image',name,caption)
    section('Background (100–500 words)',p(
        'Tomato Vision mempelajari klasifikasi kondisi visual satu tomat menjadi segar, tidak segar, dan busuk. '
        'Masalah yang dikaji adalah apakah jaringan saraf konvolusional dapat membedakan tekstur kulit, keriput, '
        'dan kerusakan permukaan pada dataset proyek yang kecil. Penilaian dibatasi pada penampilan luar; '
        'warna merah tidak otomatis berarti segar dan sistem tidak mengukur keamanan pangan atau kerusakan internal. '
        'Tujuannya adalah membangun alur yang dapat ditelusuri, mulai dari inventaris, anotasi, pemisahan sesi, '
        'preprocessing, training, hingga prediksi dan analisis kesalahan. '
        'Dua CNN dilatih dari bobot acak: baseline dengan empat blok konvolusi serta model regularized '
        'dengan residual separable convolution dan Group Normalization. Eksperimen menggunakan 82 foto asli berlabel: '
        '60 anotasi lama dan 22 foto primer impor yang digabung dengan label segar atas arahan pemilik pada 9 Oktober 2026. '
        f'Split sesi menghasilkan 70 training, 6 validation, dan 6 test terkunci. Paket dataset berisi {n_total} gambar, '
        f'yaitu 82 foto asli dan {n_syn} gambar sintetis yang belum digunakan dalam hasil training yang dilaporkan. '
        'Pada validation, baseline augmented mencapai accuracy 100% (6/6) dan macro-F1 1,000, sedangkan regularized '
        'mencapai 66,7% (4/6) dan macro-F1 0,556. Validation hanya enam foto dari dua sesi dan juga menentukan early stopping, '
        'sehingga hasil ini eksploratif dan bukan bukti peningkatan dibanding hasil sebelumnya (kedua kandidat 4/6). '
        'Hasil historis 100% pada own_v1 tidak dipakai sebagai bukti karena audit menemukan kebocoran sesi; '
        'angka 6/6 sekarang berasal dari split yang telah diaudit tetapi tetap sampel yang sangat kecil. '
        'Kontribusi proyek adalah pipeline, aplikasi, dan artefak evaluasi yang dapat diperiksa. Pemenuhan minimal 100 gambar '
        'primer belum terbukti hanya dari jumlah paket; dokumentasi sumber dan penambahan pengamatan '
        'primer tetap diperlukan sebelum klaim kepatuhan penuh.'))
    section('Literature Study and Previous Work',p('Lima artikel jurnal terkait [1]–[5] dibandingkan berdasarkan tugas dan metode; [6]–[8] menjadi landasan teknis. Artikel yang lebih lama dipakai sebagai dasar metode, bukan disebut temuan terbaru. Ringkasan berasal dari halaman penerbit yang diperiksa pada 7 Oktober 2026.'),
        table(['Rujukan','Metode dan relevansi'],[[f'[{i+1}] '+ref[0].split('. (')[0],ref[2]] for i,ref in enumerate(REFERENCES)]),
        p('Posisi penelitian: proyek ini menguji tiga kondisi visual satu komoditas dengan CNN dari nol. Dataset, label, serta unit evaluasinya berbeda dari literatur, sehingga tidak ada klaim mengungguli hasil penelitian terdahulu. Perbandingan yang sah membutuhkan data uji dan protokol yang sama.'))
    section('Data Acquisition, Preparation and Pre-processing',
        p('Inventaris primer dan sumber. PDF tugas halaman 1 mengizinkan pengumpulan primer melalui scraping atau pemotretan manual dan meminta minimal 100 gambar. Paket berisi 82 foto asli berlabel: 60 foto lama (metadata menyebutnya own, tetapi itu belum verifikasi asal pemotretan) dan 22 foto primer yang diambil pemilik sendiri dan kini digabung dengan label segar; 26 file impor memuat 4 salinan identik. Selain itu ada ' + str(n_syn) + ' gambar sintetis. Dengan 82 foto asli, syarat minimal 100 gambar primer belum terpenuhi; gambar sintetis tidak dihitung sebagai data primer.'),
        table(['Komponen','Jumlah','Penggunaan saat ini'],[['Foto lama berlabel','60','Training 48, validation 6, test terkunci 6; asal pengambilan perlu bukti'],['Foto primer impor','22','Label segar dari pemilik (9 Okt 2026), satu grup, seluruhnya training'],['Sintetis','0','source=synthetic_ai; belum disetujui untuk training'],['Salinan impor identik','4','Disimpan, tidak dihitung sebagai sampel tambahan'],['Ekspor augmentasi own_v3','360','Turunan 48 induk; tidak dipakai lagi pada own_v4 (augmentasi dilakukan online)']]),
        p('Kurasi: segar berarti kulit relatif utuh dan kencang tanpa tanda pembusukan; tidak segar berarti keriput atau layu tanpa pembusukan nyata; busuk berarti lesi atau jamur yang tampak. Ini definisi operasional visual, bukan hasil pengujian laboratorium. Label ambigu tetap tidak disetujui. Label segar pada 22 foto impor diberikan pemilik untuk seluruh foto tanpa pemeriksaan per foto; prediksi model tidak menggantikan label manusia dan kondisi tiap foto sebaiknya diverifikasi.'),
        p('Cleaning: checksum SHA-256 mendeteksi file identik; hash piksel dan dHash dipakai untuk mendeteksi salinan atau kemiripan. ID buah dan sesi digabungkan secara konservatif. Metadata sesi tidak menjamin bahwa setiap sesi adalah buah independen. Seluruh foto impor dimasukkan satu kelompok sampai identitas buah lintas tanggal diketahui.'),
        image('eda.png','Gambar 1. Inventaris dan distribusi split; dihitung dari CSV, bukan jumlah augmentasi.'),
        table(['Split','Jumlah','Per kelas','Grup sesi'],[['Training','70','segar 38, tidak_segar 16, busuk 16','8'],['Validation','6','2','2'],['Test terkunci','6','2','2']]),
        p(f'Fingerprint sumber: {fingerprint}. Split sesi menggantikan pembagian set01–set10 yang bocor. Seluruh gambar dari grup sama berada pada split sama. Validation dan test terkunci identik dengan own_v3; seluruh 22 foto baru jatuh ke training karena satu grup. Segar menjadi 38 dari 70 foto training dan 22 di antaranya berasal dari satu latar, sehingga model dapat mengaitkan latar itu dengan kelas segar. Gambar sintetis tidak dimasukkan ke validation/test.'),
        image('training_examples.png','Gambar 2. Contoh berlabel dari split training sumber lama; bukan bukti asal pemotretan.'),
        p('Transformasi: koreksi EXIF, RGB, letterbox 128 × 128 dengan padding abu-abu dan interpolasi bilinear, lalu pembagian piksel dengan 255 ke float32 [0,1]. Loader identik digunakan saat training dan inferensi. Augmentasi training mencakup flip horizontal, rotasi ±0,06 putaran, translasi 7%, zoom, kontras, dan brightness ringan. Validation/test tidak diaugmentasi.'))
    section('Methodology',image('framework.png','Gambar 3. Kerangka penelitian dan batas penggunaan test.'),
        table(['Aspek','Baseline augmented','Regularized'],[['Blok','Conv2D 32/64/128/256, BatchNorm, ReLU, MaxPool','Conv 24 + residual separable conv 48/96/160'],['Head','Global average pooling, Dense 128, dropout 0,5','Global average pooling, Dense 64, dropout 0,35'],['Normalisasi','Batch Normalization','Group Normalization, 8 grup [7]'],['Parameter',str(b['model_parameters']),str(r['model_parameters'])],['Inisialisasi','Acak; tanpa pretrained','Acak; tanpa pretrained'],['Augmentasi','Aktif pada konfigurasi pembanding ini','Aktif saat training']]),
        p('Adam, sparse categorical cross-entropy, learning rate 0,001, batch 16, seed 42, maksimum epoch 20 untuk regularized dan 30 untuk baseline augmented (configs/augmented_baseline.json), sehingga anggaran tidak setara. Sampling seimbang 120 per kelas per epoch, yaitu 360 undian dari 70 sumber training. Early stopping memantau validation loss, patience 10; learning rate diturunkan pada plateau. Bobot epoch terbaik dipulihkan. Model regularized memakai L2 1e-4 dan spatial dropout; epoch terbaiknya adalah epoch terakhir (20), jadi anggaran epoch mungkin membatasinya.'),
        p('Seleksi model memakai validation macro-F1, lalu validation loss saat F1 sama. Dua kandidat menggunakan data sumber dan split yang sama. Karena arsitektur, normalisasi, serta regularisasi berbeda sekaligus, perbandingan ini tidak mengisolasi pengaruh satu komponen. Residual connection [8] dan augmentasi [6] merupakan pilihan desain yang masih perlu ablation terkontrol.'),
        p('Test terkunci dari protokol own_v2 tidak dievaluasi dalam eksperimen own_v4_ref22. Studi own_v2 yang belum lengkap tidak dilaporkan sebagai hasil final. Tidak ada pemilihan hyperparameter menggunakan test.'))
    section('Results',p('Bukti kode tersedia di notebook IS794_TomatoVision_Project.ipynb: bagian 2–4 untuk inventaris/EDA/preprocessing, bagian 5–6 untuk model dan training, serta bagian 7–10 untuk kurva, evaluasi, kesalahan, dan inferensi. Notebook memuat artefak own_v3; hasil own_v4_ref22 dihasilkan oleh python -m src.train pada manifest data/prepared/ff822adef8db3ae5 dan dirangkum di outputs/experiments/own_v4_ref22/summary.md.'),
        table(['Kandidat','Epoch / terbaik','Train acc','Val acc','Macro-F1','Val loss'],[[name,f"{x['epochs_run']} / {x['best_epoch']}",f"{x['train_clean']['accuracy']:.4f}",f"{x['validation']['accuracy']:.4f}",f"{x['validation']['macro_f1']:.4f}",f"{x['validation']['loss']:.4f}"] for name,x in runs.items()]),
        p('Baseline augmented dipilih karena macro-F1 validation lebih tinggi (1,000 dibanding 0,556). Sumber angka: outputs/experiments/own_v4_ref22/{regularized,baseline_augmented}/run.json. Loss pada history dan kurva training dapat mencakup penalti regularisasi. Validation loss di run.json dihitung sebagai log-loss prediksi (cross-entropy), sehingga definisinya berbeda dari loss pada kurva.'),
        image('learning_curves.png','Gambar 4. Kurva validation dua kandidat dari history.json.'),
        image('validation_confusion.png','Gambar 5. Confusion matrix model terpilih (baseline augmented) pada validation, bukan test.'),
        table(['Kelas','Precision','Recall','F1','Support'],[[c,*[f"{v['report'][c][k]:.3f}" for k in ['precision','recall','f1-score']],str(int(v['report'][c]['support']))] for c in CLASSES]),
        p('Model terpilih tidak membuat kesalahan pada enam foto validation. Regularized salah pada dua foto: kedua foto segar diprediksi tidak_segar; pada data training pun regularized memprediksi 16 dari 38 foto segar sebagai tidak_segar, tanda underfitting pada kelas segar. Identitas gambar dan probabilitas tersimpan di validation_predictions.csv pada masing-masing folder run.'))
    section('Evaluation',
        p(f"Accuracy validation model terpilih = jumlah benar / jumlah sampel = {correct}/6 = {v['accuracy']:.4f}. Macro-F1 adalah rata-rata F1 tiga kelas dengan bobot sama = {v['macro_f1']:.4f}. Macro precision = {v['macro_precision']:.4f}; macro recall = {v['macro_recall']:.4f}. ECE = {v['expected_calibration_error']:.4f}; hanya enam sampel sehingga estimasinya tidak stabil."),
        p('Kalibrasi temperatur tidak dipasang karena validation hanya dua contoh per kelas dan dua grup. Probabilitas softmax tidak dapat diperlakukan sebagai peluang benar yang terkalibrasi. Validation juga dipakai untuk early stopping dan seleksi model, sehingga estimasi generalisasi cenderung optimistis.'),
        p('Test independen final belum tersedia untuk eksperimen ini. Satu kesalahan pada enam gambar mengubah accuracy sebesar 16,7 poin persentase. Perubahan dari 4/6 (own_v3) ke 6/6 (own_v4, baseline augmented) setara dua foto dan berasal dari perubahan data training sekaligus pergantian kandidat terpilih, sehingga tidak dapat dikaitkan dengan 22 foto baru. Hasil own_v1 yang sebelumnya 100% ditarik dari hasil utama: audit menunjukkan sesi A_meja_kayu tersebar di train, validation, dan test. Menampilkan angka itu sebagai keberhasilan saat ini tidak sah.'),
        p('Diagnostik foto impor sebelum digabung: model yang tersedia saat itu, yang belum melihat foto tersebut, memprediksi 18 dari 22 foto sebagai busuk dan 4 sebagai tidak_segar, tanpa satu pun segar. Bila label segar dari pemilik benar, model lama salah pada seluruh 22 foto; ini mengindikasikan pergeseran domain (latar ramai, tomat kecil dalam bingkai), meski penyebabnya belum dibuktikan dan menjadi alasan penggabungan ini. Kini foto tersebut masuk training dan tidak dapat lagi dipakai untuk menilai generalisasi; uji berikutnya memerlukan foto dari latar dan sesi baru.'))
    section('Pembeda: prediksi selektif dan variasi terstruktur',
        p('Kontribusi proyek adalah kombinasi audit sesi, CNN ringan, dan evaluasi prediksi selektif pada tiga kelas kondisi tomat. Ini kontribusi penerapan dan evaluasi, bukan klaim menciptakan algoritma CNN atau abstention baru. Referensi [1]–[5] dibedakan menurut tugas dan metode, bukan dinyatakan tidak pernah memakai teknik serupa.'),
        p('Coverage adalah proporsi gambar yang menerima prediksi otomatis. Selective risk adalah proporsi kesalahan di antara prediksi yang diterima. Evaluasi memakai ambang 0,7 dari konfigurasi yang telah ada; tidak ada fitting ambang pada test. Jika tidak ada prediksi diterima, accuracy dan risk tidak terdefinisi, bukan dianggap sempurna.'),
        table(['Kebijakan','Diterima','Coverage','Benar dari diterima','Salah lolos'],[policy_row('Semua prediksi','classify_all'),policy_row('Confidence ≥0,7','confidence_only_fixed'),policy_row('Confidence + kualitas + konsistensi','confidence_quality_consistency_fixed')]),
        image('risk_coverage.png','Gambar 6. Kurva deskriptif dari enam validation; bukan optimasi ambang atau hasil test.'),
        p('Pada validation model terpilih tidak ada kesalahan yang dapat dirujuk, sehingga aturan selektif hanya menurunkan coverage (83,3% dan 50%) tanpa manfaat terukur. Selective risk 0 berasal dari enam gambar yang semuanya benar dan bukan jaminan keselamatan; manfaat guard baru dapat dinilai pada data yang memuat kesalahan nyata.'),
        p('Ekspansi gambar merancang variasi latar, pencahayaan, ukuran objek, sudut, bentuk, permukaan kering/basah, serta blur ringan. Faktor adegan dibagi pada kelas segar, tidak segar, dan busuk agar tetesan air atau latar tertentu tidak otomatis berarti busuk. Metadata adalah atribut yang diminta, bukan ukuran fisik yang telah diukur. Blur yang menghilangkan bukti kondisi harus ditinjau, bukan diberi label yakin.'),
        p('Uji pengaruh latar yang bersifat kausal memerlukan foto buah yang sama pada latar berbeda di waktu berdekatan. Gambar yang dihasilkan secara terpisah bukan pasangan buah identik. Seluruh grup adegan harus tetap dalam satu split; data tambahan belum menjadi bukti peningkatan akurasi model.'))
    section('Deployment (if any)',
        p('Aplikasi lokal memiliki menu Prediksi, Dataset & label, Training, dan Evaluasi. Jalankan model yang dibahas: python -m src.app --run outputs/experiments/own_v4_ref22/' + SELECTED + '. Inferensi CLI: python -m src.predict foto.jpg --run outputs/experiments/own_v4_ref22/' + SELECTED + '. Server lokal tersedia pada http://127.0.0.1:7860.'),
        p('Demo browser statis merupakan artefak terpisah dan belum diganti dengan model own_v4; demo masih memakai model historis. Oleh karena itu demo tersebut tidak dijadikan bukti kinerja model regularized saat ini. Rilis web perlu ekspor model dan uji paritas sebelum dinyatakan sesuai. Deployment adalah nilai tambah dalam PDF tugas, bukan pengganti evaluasi model atau syarat dataset.'),
        p('Batas sistem: classifier tiga kelas untuk satu tomat; tidak memiliki kelas non-tomat dan bukan detektor objek. Unggahan gambar yang salah dapat tetap memperoleh probabilitas tinggi. Tampilkan peringatan kualitas dan perlunya tinjauan pada presentasi.'))
    section('Analysis and Discussion',
        p('1. Mengapa belum dapat disebut sangat akurat? Baseline augmented benar pada enam dari enam foto validation (train 98,6%); regularized benar pada empat dari enam (train 75,7%). Pada own_v3 kedua kandidat 4/6. Selisih satu-dua foto dari enam foto dua sesi tidak cukup untuk menyimpulkan generalisasi. Regularized juga underfit pada training (16 dari 38 foto segar salah) dan memakai epoch terakhir sebagai terbaik, sehingga 20 epoch mungkin tidak cukup.'),
        p('2. Arti pemilihan baseline augmented. Model ini memiliki 423.619 parameter, sekitar 4,4 kali regularized (96.019). Pemilihan mengikuti aturan macro-F1 lalu loss. Perbandingan tidak setara karena arsitektur, normalisasi, regularisasi, dan anggaran epoch (30 vs 20) berbeda sekaligus, sehingga tidak membuktikan baseline lebih baik secara umum. Dibutuhkan ablation dengan data, seed, dan anggaran epoch yang sama.'),
        p('3. Ambiguitas label. Pada own_v3 kesalahan terjadi antar kelas berdekatan, dan 22 foto baru diberi label segar sekaligus oleh pemilik tanpa penilaian per foto; bila sebagian sudah mulai layu, label itu mengajarkan kesalahan. Keriput ringan atau lesi kecil dapat hilang saat resize. Anotasi lintas penilai, dokumentasi kondisi buah, dan pengukuran kesepakatan akan lebih informatif daripada mengubah label agar cocok dengan model.'),
        p(f'4. Jumlah file versus informasi. Augmentasi dan gambar sintetis menambah variasi visual, tetapi tidak membuktikan pengamatan primer baru. Dataset {n_total} gambar mencakup {n_syn} sintetis yang belum dipakai; dari 82 foto asli, 22 berasal dari satu sesi/latar dan satu kelas. Perlu membedakan jumlah file, jumlah anotasi disetujui, jumlah sesi, dan jumlah buah independen.'),
        p('5. Perbandingan literatur. Artikel [1]–[5] menggunakan tugas dan dataset berbeda. Model pretrained, distilasi, dan detektor objek dapat menjadi arah eksperimen lanjutan. Angka mereka tidak dijadikan benchmark langsung bagi 6 gambar validation proyek ini.'),
        p('6. Prioritas perbaikan. Verifikasi asal sumber; tambah data primer terdokumentasi sampai minimal 100, dengan sebaran kelas dan buah beragam; verifikasi kondisi 22 foto impor satu per satu dan tambahkan foto tidak_segar/busuk pada latar yang sama agar latar tidak menjadi sinyal kelas; ulangi split berdasarkan buah/sesi; tetapkan protokol dan model sebelum satu evaluasi test independen. Setelah data memadai, bandingkan scratch CNN dengan transfer learning yang memang diizinkan tugas.'))
    section('Kesimpulan dan status pengumpulan',
        p('Proyek telah memiliki CNN, pipeline data, hasil validation terukur, notebook proyek, aplikasi lokal, dan dokumentasi. Hasil yang dapat dipertanggungjawabkan saat ini adalah accuracy validation 100% (6/6) untuk baseline augmented dan 66,7% (4/6) untuk regularized pada enam foto, bukan klaim test. Kelengkapan berkas tidak otomatis berarti semua syarat akademik terpenuhi.'),
        table(['Ketentuan PDF','Status / tindakan'],[['Data primer ≥100 gambar','Belum terbukti; 82 foto asli (22 di antaranya satu sesi) dan sintetis tidak dihitung primer'],['Neural network','Terpenuhi: dua CNN TensorFlow/Keras dari nol'],['Metrik relevan','Terpenuhi untuk evaluasi validation; test final belum selesai'],['PPT + IPYNB + dataset','Dataset dan laporan diperbarui 9 Okt 2026; presentasi dan notebook belum diperbarui ke own_v4'],['Laporan PDF','Dokumen Word disediakan sesuai permintaan pengguna; ekspor PDF sebelum pengumpulan resmi'],['Nama ZIP sesuai kelas/kelompok','Ganti ClassXX/GroupXX setelah menggabungkan identitas terpisah'],['Judul tidak sama dengan kelompok lain','Perlu diperiksa di kelas oleh tim'],['Presentasi dan kontribusi seluruh anggota','Latih presentasi; identitas/kontribusi berasal dari dokumen tim']]),
        p('Batas kesesuaian template: PDF yang diberikan memuat ketentuan dan rubrik, tetapi tidak memuat template laporan terpisah. Struktur Word mengikuti heading template yang sudah ada dalam proyek. Bila dosen menyediakan template lain, pindahkan isi ke template tersebut. Identitas tim tidak direkayasa dan digabungkan dari file pemilik.'))
    section('References',*[p(f'[{i+1}] {ref[0]} {ref[1]}') for i,ref in enumerate(REFERENCES)])
    section('Bantuan perangkat lunak dan jejak reproduksi',
        p('Perangkat: Python 3.11, TensorFlow 2.16.1, Keras 3.15.1, NumPy, pandas, scikit-learn, Pillow, Matplotlib. Model dilatih dari nol. Perintah reproduksi, versi pustaka dokumen, dan notebook disertakan. Tidak ada angka hasil eksperimen yang dibuat untuk memenuhi target akurasi.'))
    return sections


def word_report(sections):
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    d=Document(); s=d.sections[0]
    s.top_margin=Inches(.8); s.bottom_margin=Inches(.75)
    normal=d.styles['Normal']; normal.font.name='Calibri'; normal.font.size=Pt(10.5)
    normal.paragraph_format.space_after=Pt(7)
    for style in ['Heading 1','Heading 2']: d.styles[style].font.color.rgb=RGBColor.from_string('216B58')
    s.header.paragraphs[0].text='IS794 • Deep Learning • Ganjil 2026–2027'
    footer=s.footer.paragraphs[0]; footer.text='Tomato Vision  |  '
    field=OxmlElement('w:fldSimple'); field.set(qn('w:instr'),'PAGE'); footer._p.append(field)
    d.add_paragraph('TOMATO VISION','Title')
    d.add_paragraph('Klasifikasi Kondisi Visual Tomat dengan CNN Ringan dan Evaluasi Prediksi Selektif pada Variasi Pengambilan Gambar','Subtitle')
    d.add_paragraph('Laporan proyek • revisi 9 Oktober 2026')
    d.add_paragraph('Project of Week: Final project (Week 13–14)\nGroup Name and Class / Member (Name / NIM): digabungkan dari dokumen identitas tim.')
    d.add_heading('Status hasil',1)
    total=len(list(csv.DictReader((OUT/'dataset/labels.csv').open())))
    d.add_paragraph(f'{total} gambar dalam paket • 82 foto asli berlabel digunakan dalam eksperimen • validation accuracy baseline augmented 100% (6/6), regularized 66,7% (4/6) • test final belum dievaluasi. Status data primer belum terverifikasi lengkap.')
    d.add_heading('Daftar bagian',1)
    for i,(title,_) in enumerate(sections,1): d.add_paragraph(f'{i}. {title}')
    d.add_page_break()
    for title,items in sections:
        d.add_heading(title,1)
        for item in items:
            if item[0]=='p': d.add_paragraph(item[1])
            elif item[0]=='image':
                d.add_picture(str(OUT/'figures'/item[1]),width=Inches(6.2))
                d.add_paragraph(item[2],'Caption')
            else:
                headers,rows=item[1:]
                table=d.add_table(rows=1,cols=len(headers)); table.style='Light Shading Accent 1'
                for cell,text in zip(table.rows[0].cells,headers): cell.text=text
                repeat=OxmlElement('w:tblHeader'); table.rows[0]._tr.get_or_add_trPr().append(repeat)
                for row in rows:
                    cells=table.add_row().cells
                    for cell,text in zip(cells,row): cell.text=str(text)
                    for cell in cells:
                        for p in cell.paragraphs:
                            for run in p.runs: run.font.size=Pt(9)
    target=OUT/'IS794_Laporan_TomatoVision.docx'
    legacy=OUT/'laporan/IS794_Laporan_TomatoVision.docx'
    backup=ROOT/'outputs/assignment_audit/IS794_Laporan_sebelum_audit.docx'
    backup.parent.mkdir(parents=True,exist_ok=True)
    if legacy.exists() and not backup.exists(): shutil.copy2(legacy,backup)
    d.save(target)
    # Keep the former entry point in sync, to avoid accidentally submitting stale 100% results.
    shutil.copy2(target,legacy)


def slides(runs):
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    entries=[
      ('Tomato Vision','Klasifikasi kondisi visual tomat', ['IS794 • Deep Learning • Ganjil 2026–2027','Segar / tidak segar / busuk','Identitas tim: gabungkan dari dokumen terpisah'],None,'Pembuka: masalah, ruang lingkup satu tomat, dan tujuan penelitian. Jangan menjanjikan keamanan pangan.'),
      ('Masalah & tujuan','Apa yang ingin dijawab?', ['Membedakan keriput dan kerusakan permukaan','Menguji CNN pada data kecil dengan split sesi','Menyediakan prediksi dan analisis yang dapat diperiksa'],None,'Jelaskan perbedaan kesegaran, kematangan, dan keamanan pangan.'),
      ('Peta literatur','Tugas sama belum berarti benchmark sama', ['[1] ResNet-101 + attention; [2] distilasi ViT → CNN','[3] YOLO kualitas tomat; [4] YOLOv4 buah/sayur','[5] AlexNet transfer learning; proyek ini CNN dari nol'],None,'Rujukan lengkap pada akhir slide. Bandingkan metode, bukan mengklaim lebih baik dari akurasi artikel.'),
      ('Inventaris dataset','127 gambar ≠ 127 data primer terverifikasi', ['60 sumber lama berlabel • 45 sintetis • 22 impor review','4 salinan impor tidak dihitung ulang','360 augmentasi tersimpan terpisah'], 'eda.png','Asal pemotretan sumber lama belum dikonfirmasi. Minimum 100 primer belum terbukti oleh jumlah file.'),
      ('Akuisisi & kurasi','Jejak sumber dan label', ['Simpan sumber, tanggal, buah, sesi, dan kondisi','Kriteria visual tiga kelas; label ambigu ditunda','Prediksi model tidak dijadikan ground truth'], 'training_examples.png','Contoh berasal dari training sumber lama. Jangan menyebut sumber sebagai foto kamera tanpa bukti.'),
      ('Cleaning & split','Satu buah/sesi tetap satu split', ['SHA-256 + hash piksel + pemeriksaan kemiripan','48 train / 6 validation / 6 test terkunci','Grup sesi: 7 / 2 / 2; split historis bocor ditinggalkan'],None,'Jelaskan bagaimana latar sama di train dan test dapat membuat akurasi semu.'),
      ('Preprocessing','Satu loader untuk training dan inferensi', ['EXIF → RGB → letterbox 128 × 128 → [0,1]','Flip, rotasi, zoom, kontras: training saja','Sampling 120/kelas bukan 120 buah baru'],None,'Alasan letterbox: menjaga aspek rasio. Warna tidak diganti ekstrem.'),
      ('Rencana akuisisi berikutnya','Penuhi jumlah dan keragaman', ['Target rencana: 120 gambar primer, sekitar 40 per kelas','Catat buah independen, sesi, kamera, cahaya, dan latar','Periksa duplikat dan konfirmasi label sebelum training'],None,'Ini rencana, bukan hasil yang sudah terkumpul. Minimal administratif tidak menjamin generalisasi.'),
      ('Kerangka penelitian','Dari inventaris hingga aplikasi', ['Sumber → kurasi → split → training → validation','Pemilihan model sebelum membuka test','Dataset tambahan belum masuk hasil eksperimen ini'], 'framework.png','Notebook memetakan seluruh tahap. Jangan mengubah test untuk memperbaiki skor.'),
      ('Dua CNN dari nol','Arsitektur dan efisiensi', ['Baseline augmented: 423.619 parameter, BatchNorm','Regularized: 96.019 parameter, residual separable conv','GroupNorm + dropout + L2; softmax tiga kelas'],None,'Regularized sekitar 77% lebih sedikit parameter. Ini tidak otomatis berarti akurasi lebih tinggi.'),
      ('Training & seleksi','Konfigurasi dapat diulang', ['Adam • LR 0,001 • batch 16 • seed 42 • maksimum 20 epoch','Early stopping berdasarkan validation loss','Pilih macro-F1, lalu loss jika F1 sama'],None,'Jelaskan training accuracy vs validation accuracy serta mengapa test tidak dipakai untuk seleksi.'),
      ('Hasil validation','4 dari 6 gambar benar', ['Kedua kandidat: accuracy 66,7%; macro-F1 0,656','Regularized loss 0,538; baseline augmented 0,904','Regularized dipilih; test final belum dinilai'], 'learning_curves.png','Data dari run.json dan history.json. Jangan gunakan angka historis 100%.'),
      ('Analisis kesalahan','Kelas berdekatan masih tertukar', ['Tidak segar → segar: 1 gambar','Busuk → tidak segar: 1 gambar','Validation hanya dua sesi; satu salah = 16,7 poin'], 'validation_confusion.png','Bahas implikasi salah klasifikasi busuk. Model tidak boleh menggantikan pemeriksaan manusia.'),
      ('Foto impor & generalisasi','Diagnostik tanpa label aktual', ['22 unik: 18 prediksi busuk, 4 tidak segar','14 perlu review; belum dapat dihitung akurasinya','Latar ramai / objek kecil: hipotesis domain shift'],None,'Tunjukkan bahwa confident prediction belum tentu benar. Label perlu observasi, bukan menyalin prediksi.'),
      ('Demo & reproduksi','Notebook dan aplikasi lokal', ['Notebook: EDA → CNN → training opsional → evaluasi → inferensi','Demo lokal memakai --run own_v3_augmented/regularized','Model situs historis bukan hasil baru'],None,'Demokan satu gambar dan probabilitas. Gunakan perintah lengkap dari laporan; siapkan model lokal bila internet gagal.'),
      ('Kesimpulan & batas','Prototipe berfungsi; bukti masih terbatas', ['CNN dan metrik relevan tersedia; 127 gambar terinventaris','Syarat 100 primer belum terbukti; 22 label belum dikonfirmasi','Lanjutkan akuisisi, verifikasi, dan test independen'],None,'Semua anggota menyampaikan bagian sesuai kontribusi nyata. Jangan mengada-ada kontribusi atau hasil.'),
      ('Referensi 1–4','Jurnal terkait', [f'[{i+1}] {ref[0]}\n{ref[1]}' for i,ref in enumerate(REFERENCES[:4])],None,'Referensi halaman penerbit diverifikasi pada 7 Oktober 2026.'),
      ('Referensi 5–8','Jurnal dan dasar metode', [f'[{i+5}] {ref[0]}\n{ref[1]}' for i,ref in enumerate(REFERENCES[4:])],None,'Artikel lama dipakai untuk fondasi metode. Pertanyaan: mengapa accuracy belum tinggi? Data dan validasi sangat kecil.')]
    rows=list(csv.DictReader((OUT/'dataset/labels.csv').open()))
    total=len(rows); synthetic=sum(r['source']=='synthetic_ai' for r in rows)
    def current(text): return re.sub(r'(?<![\d.])127(?![\d.])', str(total), text).replace('45 sintetis',f'{synthetic} sintetis')
    entries=[(current(t),current(s),[current(b) for b in bs],im,current(n)) for t,s,bs,im,n in entries]
    selective_slide=('Prediksi selektif','Accuracy subset selalu dibaca bersama coverage',
        ['Selalu klasifikasi: 4/6 benar, coverage 100%', 'Aturan review: 3/4 benar, coverage 66,7%', 'Satu kesalahan masih lolos; belum ada bukti manfaat guard tambahan'],
        'risk_coverage.png','Ambang 0,7 sudah ada di konfigurasi. Jangan menyebut 75% sebagai accuracy keseluruhan; hanya empat gambar yang diterima.')
    variation_slide=('Variasi untuk stress-test','Air, jarak, dan blur bukan label kesegaran',
        ['Permukaan kering/basah, blur ringan, objek dekat/jauh','Faktor adegan dibagi di ketiga kelas','Atribut rancangan perlu tinjauan; bukan bukti efek kausal'],
        None,'Tujuan menghindari petunjuk palsu: air tidak berarti busuk. Foto terlalu buram harus diambil ulang. Dampak pada akurasi belum dibuktikan.')
    final_entries=entries[:14]+[selective_slide,variation_slide]+entries[14:]
    for name,selected in [('IS794_Presentasi_Final.pptx',final_entries),('IS794_Presentasi_Week7.pptx',entries[:8]+entries[16:])]:
        prs=Presentation(); prs.slide_width=Inches(13.333); prs.slide_height=Inches(7.5)
        for index,(title,sub,bullets,picture,notes) in enumerate(selected,1):
            slide=prs.slides.add_slide(prs.slide_layouts[6])
            slide.background.fill.solid(); slide.background.fill.fore_color.rgb=RGBColor.from_string('F8F7F2')
            bar=slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,0,0,Inches(.18),prs.slide_height)
            bar.fill.solid(); bar.fill.fore_color.rgb=RGBColor.from_string('24755F'); bar.line.fill.background()
            def text(x,y,w,h,value,size,color='243C34',bold=False):
                box=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h)); tf=box.text_frame; tf.word_wrap=True
                for j,line in enumerate(value.split('\n')):
                    p=tf.paragraphs[0] if j==0 else tf.add_paragraph(); p.text=line
                    p.font.size=Pt(size); p.font.bold=bold; p.font.color.rgb=RGBColor.from_string(color)
                    p.space_after=Pt(14)
                return box
            text(.6,.35,12,.6,title,32,bold=True); text(.62,1.05,12,.5,sub,18,color='64776E')
            if title.startswith('Referensi'):
                for i,b in enumerate(bullets): text(.65,1.8+i*1.18,12,1.14,b,13)
            elif picture:
                text(.65,1.85,4.5,4.8,'\n'.join('• '+b for b in bullets),22)
                from PIL import Image
                with Image.open(OUT/'figures'/picture) as im: ratio=im.width/im.height
                width=min(7.1,4.8*ratio); height=width/ratio
                slide.shapes.add_picture(str(OUT/'figures'/picture), Inches(5.65+(7.1-width)/2), Inches(1.8+(4.8-height)/2),width=Inches(width),height=Inches(height))
            else: text(.65,1.95,11.8,4.8,'\n'.join('• '+b for b in bullets),25)
            text(.65,7.0,12,.3,f'IS794  •  Tomato Vision  |  7 Oktober 2026                                       {index:02d}',10,color='64776E')
            slide.notes_slide.notes_text_frame.text=notes
        prs.save(OUT/'presentasi'/name)


def main():
    rows,manifest,runs=load_facts()
    figures(manifest,runs)
    sections=content(runs)
    total=len(rows); count=sum(r['source']=='synthetic_ai' for r in rows)
    counts=Counter(r['label'] for r in rows if r['source']=='synthetic_ai')
    def update(value):
        if isinstance(value,str):
            return re.sub(r'(?<![\d.])127(?![\d.])', str(total), value).replace('45 gambar',f'{count} gambar').replace('45 sintetis',f'{count} sintetis').replace('15 per kelas',', '.join(f'{c}: {counts[c]}' for c in CLASSES)).replace('Sintetis baru','Sintetis')
        if isinstance(value,(list,tuple)): return type(value)(update(v) for v in value)
        return value
    sections=update(sections)
    for title,items in sections:
        for item in items:
            if item[0]=='table':
                for row in item[2]:
                    if row[0]=='Sintetis': row[1]=str(count)
    word_report(sections)
    if '--slides' in sys.argv: slides(runs)
    (OUT/'audit/references.json').write_text(json.dumps([{'reference':r[0],'url':r[1],'relevance':r[2],'verified_date':'2026-10-07'} for r in REFERENCES],ensure_ascii=False,indent=2)+'\n')
    print('Built Word report and figures from current metadata' + (' and PPTX decks.' if '--slides' in sys.argv else '.'))

if __name__=='__main__': main()
