"""CLI/library prediction with saved class order, preprocessing and uncertainty flag.

  python -m src.predict foto.jpg                 # satu foto
  python -m src.predict a.jpg b.png folder/      # banyak foto dan/atau folder (tidak rekursif)
  python -m src.predict foto.jpg --json          # keluaran JSON lengkap

Model default: models/release/ (jaringan yang sama dengan demo web), atau outputs/selected_run.json bila tidak ada.
Preprocessing dan ambang "perlu tinjauan" identik dengan demo web (diuji di tests/test_predict_vs_web.py).
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image, UnidentifiedImageError
from src.config import CLASS_NAMES, MODELS_DIR, OUTPUTS_DIR, ExperimentConfig, resolve_path
from src.preprocessing import load_image
from src.evaluate import temperature_scale
from src.quality import assess_quality, review_decision

def selected_run():
    path=OUTPUTS_DIR/"selected_run.json"
    if not path.exists():
        raise FileNotFoundError("No selected run. Run src.experiment or specify --run.")
    return resolve_path(json.loads(path.read_text())["run"])

RELEASE_DIR=MODELS_DIR/"release"
IMAGE_SUFFIXES={".jpg",".jpeg",".png",".webp"}   # same formats the web demo accepts; used when scanning folders

def release_run():
    """Directory of the model published with the web demo (models/release/<default>), or None."""
    manifest=RELEASE_DIR/"release.json"
    if not manifest.exists():
        return None
    run=RELEASE_DIR/json.loads(manifest.read_text())["default"]
    return run if (run/"model.keras").exists() and (run/"run.json").exists() else None

def default_run():
    run=release_run()
    if run is not None:
        return run
    try:
        return selected_run()
    except FileNotFoundError:
        raise FileNotFoundError("Tidak ada model. Gunakan --run direktori_run, atau pastikan models/release/ ada "
                                "(dibuat oleh python -m src.export_web).") from None

def load_trained_model(model_path=None):
    import tensorflow as tf
    return tf.keras.models.load_model(model_path or selected_run()/"model.keras",compile=False)

class Predictor:
    def __init__(self,run=None):
        import tensorflow as tf
        self.run=resolve_path(run) if run else selected_run()
        self.meta=json.loads((self.run/"run.json").read_text())
        if self.meta["classes"]!=CLASS_NAMES:
            raise ValueError("Unsupported class order")
        config=ExperimentConfig(**self.meta['config'])
        if not np.isfinite(self.meta['temperature']) or self.meta['temperature']<=0:
            raise ValueError('Invalid saved temperature')
        self.model=tf.keras.models.load_model(self.run/"model.keras",compile=False)
        # Prefer a threshold calibrated on validation (out-of-fold) predictions when the run provides one.
        calibrated=(self.meta.get('review_threshold') or {}).get('threshold')
        self.review_threshold=float(calibrated if calibrated is not None else config.confidence_threshold)
        if tuple(self.model.input_shape[1:])!=(config.image_size,config.image_size,3) or self.model.output_shape[-1]!=3:
            raise ValueError('Saved model does not match input dimensions or class count')

    def predict(self,image):
        x=load_image(image,self.meta["config"]["image_size"])
        views=np.stack([x,x[:,::-1,:],np.clip(x*.9,0,1),np.clip(x*1.1,0,1)])
        raw=self.model(views,training=False).numpy()
        probabilities=temperature_scale(raw,self.meta["temperature"])
        probs=probabilities[0]
        idx=int(probs.argmax())
        confidence=float(probs[idx])
        quality=assess_quality(x)
        decision=review_decision(probs,raw[0],probabilities.argmax(axis=1),quality,
                                 self.meta['mode'],self.review_threshold)
        return {"label":CLASS_NAMES[idx],"confidence":confidence,
                "probabilities":dict(zip(CLASS_NAMES,map(float,probs))),
                "raw_probabilities":dict(zip(CLASS_NAMES,map(float,raw[0]))),
                "quality":quality,**decision,
                "calibration":self.meta.get('calibration',{'note':'Legacy validation-only temperature; reliability not established'}),
                "mode":self.meta["mode"],"production_ready":self.meta["production_ready"],
                "scope":"Satu tomat terlihat jelas. Tidak mendeteksi objek non-tomat atau menjamin keamanan pangan."}

def predict_image(model,image_path):
    size=int(model.input_shape[1])
    probs=model(load_image(image_path,size)[None],training=False).numpy()[0]
    idx=int(probs.argmax())
    return CLASS_NAMES[idx],float(probs[idx]),dict(zip(CLASS_NAMES,map(float,probs)))

def expand_inputs(paths):
    """Return (files, problems, skipped). Folders are scanned one level deep; explicit files are always attempted."""
    files,problems,skipped=[],[],0
    for value in paths:
        path=Path(value).expanduser()
        if path.is_dir():
            found=sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
            skipped+=sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() not in IMAGE_SUFFIXES and not p.name.startswith("."))
            if not found:
                problems.append({"image":str(path),"error":"Folder tidak berisi foto JPG/PNG/WebP."})
            files.extend(found)
        elif path.is_file():
            files.append(path)
        else:
            problems.append({"image":str(path),"error":"File atau folder tidak ditemukan."})
    return files,problems,skipped

def explain_error(error):
    if isinstance(error,UnidentifiedImageError):
        return "Bukan file gambar yang dapat dibaca. Gunakan JPG, PNG, atau WebP (foto HEIC dari iPhone: ekspor ke JPG dulu)."
    if isinstance(error,(OSError,ValueError)):
        return f"Foto tidak dapat diproses: {error}"
    raise error

def predict_files(predictor,files):
    rows=[]
    for path in files:
        try:
            rows.append({"image":str(path),**predictor.predict(path)})
        except (OSError,ValueError,Image.DecompressionBombError) as error:
            rows.append({"image":str(path),"error":explain_error(error)})
    return rows

def format_row(row,threshold):
    if "error" in row:
        return f"{row['image']}\n  GAGAL: {row['error']}"
    names=" | ".join(f"{name} {row['probabilities'][name]:6.1%}" for name in CLASS_NAMES)
    status="PERLU TINJAUAN" if row["needs_review"] else "kandidat (bukan jaminan)"
    lines=[row["image"],f"  Dugaan      : {row['label']} ({row['confidence']:.1%}) -> {status}",f"  Probabilitas: {names}"]
    lines.extend(f"  - {reason}" for reason in row["review_reasons"])
    return "\n".join(lines)

def main(argv=None):
    parser=argparse.ArgumentParser(description="Prediksi kesegaran tomat dengan model TensorFlow/Keras yang tersimpan.")
    parser.add_argument("paths",nargs="*",help="foto, banyak foto, atau folder")
    parser.add_argument("--image",nargs="+",default=[],help="sama seperti argumen posisi (kompatibilitas lama)")
    parser.add_argument("--run",help="direktori run (run.json + model.keras); default: models/release lalu outputs/selected_run.json")
    parser.add_argument("--json",action="store_true",help="cetak JSON lengkap, bukan tabel")
    parser.add_argument("--output",help="simpan JSON lengkap ke berkas")
    args=parser.parse_args(argv)
    inputs=[*args.paths,*args.image]
    if not inputs:
        parser.error("beri minimal satu foto atau folder")
    files,problems,skipped=expand_inputs(inputs)
    if not files:
        for problem in problems:
            print(f"GAGAL {problem['image']}: {problem['error']}",file=sys.stderr)
        return 2
    try:
        predictor=Predictor(args.run or default_run())
    except ModuleNotFoundError as error:
        print(f"{error}. Pasang dependensi: pip install -r requirements.txt",file=sys.stderr)
        return 2
    except (FileNotFoundError,ValueError) as error:
        print(f"Model tidak dapat dimuat: {error}",file=sys.stderr)
        return 2
    rows=problems+predict_files(predictor,files)
    text=json.dumps(rows,indent=2,ensure_ascii=False)
    if args.json:
        print(text)
    else:
        print(f"Model: {predictor.run}   ambang perlu tinjauan: {predictor.review_threshold:.0%}")
        print("\n\n".join(format_row(row,predictor.review_threshold) for row in rows))
        if skipped:
            print(f"\n({skipped} file non-gambar di folder dilewati)")
        print("\nHasil berupa dugaan kondisi visual satu tomat; bukan penilaian keamanan pangan.")
    if args.output:
        path=resolve_path(args.output)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(text)
    return 1 if any("error" in row for row in rows) else 0

if __name__=="__main__":
    sys.exit(main())
