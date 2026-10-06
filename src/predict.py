"""CLI/library prediction with saved class order, preprocessing and uncertainty flag."""
import argparse
import json
from pathlib import Path
import numpy as np
from src.config import CLASS_NAMES, OUTPUTS_DIR, ExperimentConfig, resolve_path
from src.preprocessing import load_image
from src.evaluate import temperature_scale
from src.quality import assess_quality, review_decision

def selected_run():
    path=OUTPUTS_DIR/"selected_run.json"
    if not path.exists():
        raise FileNotFoundError("No selected run. Run src.experiment or specify --run.")
    return resolve_path(json.loads(path.read_text())["run"])

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
                                 self.meta['mode'],self.meta['config']['confidence_threshold'])
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

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--image",required=True,nargs="+")
    parser.add_argument("--run")
    parser.add_argument("--output",help="Optional JSON output path")
    args=parser.parse_args()
    predictor=Predictor(args.run)
    rows=[{"image":path,**predictor.predict(path)} for path in args.image]
    text=json.dumps(rows,indent=2,ensure_ascii=False)
    print(text)
    if args.output:
        path=resolve_path(args.output)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(text)

if __name__=="__main__":
    main()
