"""Transparent inference diagnostics; not an object detector or freshness oracle."""
import numpy as np

def assess_quality(rgb):
    gray=np.asarray(rgb,dtype=np.float32)@np.array([.2126,.7152,.0722],dtype=np.float32)
    brightness=float(gray.mean())
    contrast=float(gray.std())
    laplacian=(gray[1:-1,:-2]+gray[1:-1,2:]+gray[:-2,1:-1]+gray[2:,1:-1]-4*gray[1:-1,1:-1])
    sharpness=float(laplacian.var())
    warnings=[]
    if brightness<.08:
        warnings.append('Gambar sangat gelap; ambil ulang dengan pencahayaan lebih baik.')
    if brightness>.95:
        warnings.append('Gambar sangat terang; detail permukaan mungkin hilang.')
    if contrast<.025:
        warnings.append('Kontras sangat rendah atau gambar hampir polos.')
    if sharpness<.00003:
        warnings.append('Detail tepi rendah; periksa fokus dan ukuran tomat dalam foto.')
    return {'brightness':brightness,'contrast':contrast,'edge_variance':sharpness,
            'warnings':warnings,'method':'Heuristic on preprocessed image; not a non-tomato detector or validated blur classifier'}

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
