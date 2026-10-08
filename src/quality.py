"""Transparent inference diagnostics; not an object detector or freshness oracle."""
import numpy as np

def assess_quality(rgb):
    gray=np.asarray(rgb,dtype=np.float32)@np.array([.2126,.7152,.0722],dtype=np.float32)
    brightness=float(gray.mean())
    contrast=float(gray.std())
    sharpness=content_sharpness(rgb,gray)
    warnings=[]
    if brightness<.08:
        warnings.append('Gambar sangat gelap; ambil ulang dengan pencahayaan lebih baik.')
    if brightness>.95:
        warnings.append('Gambar sangat terang; detail permukaan mungkin hilang.')
    if contrast<.025:
        warnings.append('Kontras sangat rendah atau gambar hampir polos.')
    if sharpness<SHARPNESS_MIN:
        warnings.append('Foto tampak buram atau detail tepi rendah; periksa fokus dan ukuran tomat dalam foto.')
    tomato_fraction=tomato_color_fraction(rgb)
    if tomato_fraction<TOMATO_COLOR_MIN:
        warnings.append('Warna merah/oranye khas tomat hampir tidak terdeteksi; foto mungkin bukan tomat, '
                        'tomat hijau, atau buah terlalu kecil di foto.')
    return {'brightness':brightness,'contrast':contrast,'edge_variance':sharpness,
            'tomato_color_fraction':tomato_fraction,'warnings':warnings,
            'method':'Heuristic on preprocessed image; not a non-tomato detector or validated blur classifier'}

# Laplacian variance inside the photo area only (letterbox padding excluded: its border is a false edge).
# Calibrated 2026-10-06: all 86 own photo files >= 2.0e-3; threshold = half of that minimum. Gaussian blur
# radius >= 16 px (1254 px scale) falls below it in 33/33 tested photos; radius 8 in most; radius 4 rarely.
SHARPNESS_MIN=1e-3
PAD_VALUE=np.float32(127/255)

def content_sharpness(rgb,gray):
    x=np.asarray(rgb,dtype=np.float32)
    padding=np.all(x==PAD_VALUE,axis=-1)
    rows=np.flatnonzero(~padding.all(axis=1))
    cols=np.flatnonzero(~padding.all(axis=0))
    if len(rows)<3 or len(cols)<3:
        return 0.0
    g=gray[rows[0]:rows[-1]+1,cols[0]:cols[-1]+1]
    laplacian=(g[1:-1,:-2]+g[1:-1,2:]+g[:-2,1:-1]+g[2:,1:-1]-4*g[1:-1,1:-1])
    return float(laplacian.var())

# Share of saturated red/orange pixels. Calibrated 2026-10-06: all 82 own tomato photos >= 0.165
# (median 0.52); UI screenshots, gray, blue/green objects <= 0.042. It cannot reject red non-tomato objects.
TOMATO_COLOR_MIN=0.05

def tomato_color_fraction(rgb):
    x=np.asarray(rgb,dtype=np.float64)
    high,low=x.max(-1),x.min(-1)
    delta=high-low
    saturation=np.where(high>0,delta/np.maximum(high,1e-12),0)
    r,g,b=x[...,0],x[...,1],x[...,2]
    safe=np.where(delta>0,delta,1)
    hue=np.where(high==r,((g-b)/safe)%6,np.where(high==g,(b-r)/safe+2,(r-g)/safe+4))*60
    mask=(delta>0)&(saturation>=.35)&(high>=.2)&((hue<=40)|(hue>=330))
    return float(mask.mean())

def review_decision(probs,raw_probs,view_predictions,quality,mode,threshold=.7):
    ordered=np.sort(probs)
    candidate=int(np.argmax(probs))
    margin=float(ordered[-1]-ordered[-2])
    agreement=float(np.mean(np.asarray(view_predictions)==candidate))
    reasons=[]
    if min(float(probs[candidate]),float(raw_probs[candidate]))<threshold:
        reasons.append('Probabilitas prediksi belum melewati ambang tinjauan.')
    if margin<.15:
        reasons.append('Dua kelas teratas memiliki probabilitas berdekatan.')
    if agreement<1:
        reasons.append('Prediksi berubah pada flip atau perubahan pencahayaan ringan.')
    reasons.extend(quality['warnings'])
    return {'decision':'review' if reasons else 'candidate', 'needs_review':bool(reasons),
            'review_reasons':reasons,'probability_margin':margin,
            'stability':{'agreement':agreement,'views':len(view_predictions),
                         'method':'Original, horizontal flip, brightness x0.9 and x1.1; diagnostic only'},
            'notice':'Hasil masih berupa dugaan; perlu pemeriksaan manual.' if reasons else
                     'Dugaan kondisi visual; probabilitas bukan jaminan dan bukan penilaian keamanan pangan.'}
