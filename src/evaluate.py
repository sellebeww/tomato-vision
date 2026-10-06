"""Evaluate saved model against its immutable split. No re-splitting or test tuning."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from src.config import CLASS_NAMES, resolve_path
from src.manifest import load_bundle
from src.preprocessing import load_arrays

def temperature_scale(probs, temperature=1.0):
    probs=np.asarray(probs)
    if not np.isfinite(temperature) or temperature<=0:
        raise ValueError('Temperature must be finite and positive')
    if probs.ndim!=2 or probs.shape[1]!=3 or not np.isfinite(probs).all() or (probs<0).any():
        raise ValueError('Expected finite nonnegative probabilities for three classes')
    if not np.allclose(probs.sum(axis=1),1,atol=1e-4):
        raise ValueError('Class probabilities must sum to one')
    logits=np.log(np.clip(probs,1e-8,1.0))/temperature
    logits-=logits.max(axis=1,keepdims=True)
    exp=np.exp(logits)
    return exp/exp.sum(axis=1,keepdims=True)

def fit_temperature(labels,probs):
    grid=np.linspace(.5,4,71)
    return float(min(grid,key=lambda t:log_loss(labels,temperature_scale(probs,t),labels=range(3))))

def calibration_policy(labels,probs,groups):
    """Conservative eligibility floor, not a statistical guarantee of calibration."""
    counts=np.bincount(np.asarray(labels,dtype=int),minlength=3)
    eligible=counts.min()>=20 and len(set(groups))>=5
    return {'temperature':fit_temperature(labels,probs) if eligible else 1.0,
            'fitted':bool(eligible),'validation_images':len(labels),'validation_groups':len(set(groups)),
            'note':('Validation-only fit; calibration still needs independent assessment' if eligible else
                    'Not fitted: fewer than 20 images per class or 5 groups; raw softmax retained')}

def metrics(labels,probs):
    labels=np.asarray(labels)
    probs=np.asarray(probs,dtype=np.float64)
    if (probs.ndim!=2 or probs.shape!=(len(labels),len(CLASS_NAMES)) or len(labels)==0
            or not np.isfinite(probs).all() or (probs<0).any()
            or not np.allclose(probs.sum(axis=1),1,atol=1e-4)):
        raise ValueError('Metrics require nonempty, finite, normalized class probabilities')
    # Float32 predictions serialized to CSV can differ from one by a few ulps.
    # Normalize only after validation, without accepting malformed probabilities.
    probs=probs/probs.sum(axis=1,keepdims=True)
    predictions=probs.argmax(axis=1)
    report=classification_report(labels,predictions,labels=range(3),target_names=CLASS_NAMES,
                                 output_dict=True,zero_division=0)
    confidence=probs.max(axis=1)
    correct=predictions==labels
    ece=0.
    bins=np.minimum((confidence*10).astype(int),9)
    for index in range(10):
        mask=bins==index
        if mask.any():
            ece+=float(mask.mean()*abs(correct[mask].mean()-confidence[mask].mean()))
    return {"accuracy":float(accuracy_score(labels,predictions)),
            "macro_precision":report["macro avg"]["precision"],
            "macro_recall":report["macro avg"]["recall"],
            "macro_f1":report["macro avg"]["f1-score"],
            "loss":float(log_loss(labels,probs,labels=range(3))),
            "expected_calibration_error":ece,"report":report,
            "confusion_matrix":confusion_matrix(labels,predictions,labels=range(3)).tolist(),
            "n_images":len(labels)}

def group_bootstrap(df,probs,seed=42,repeats=300):
    rng=np.random.default_rng(seed)
    groups=df.group_id.unique()
    indexes=[np.flatnonzero(df.group_id.to_numpy()==g) for g in groups]
    scores=[]
    for _ in range(repeats):
        ids=np.concatenate([indexes[i] for i in rng.integers(0,len(groups),len(groups))])
        labels=df.label_idx.to_numpy()[ids]
        if len(np.unique(labels))==3:
            scores.append(metrics(labels,probs[ids])["macro_f1"])
    return {"group_count":len(groups),"valid_resamples":len(scores),
            "macro_f1_95_percent_interval":np.quantile(scores,[.025,.975]).tolist() if scores else None,
            "note":"Group bootstrap; unreliable with very few independent groups"}

def plot_confusion_matrix(y_true,y_pred,output_path,class_names=CLASS_NAMES):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm=confusion_matrix(y_true,y_pred,labels=range(len(class_names)))
    fig,ax=plt.subplots(figsize=(6,5))
    ax.imshow(cm,cmap="Blues")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j,i,str(cm[i,j]),ha="center",va="center")
    ax.set(xticks=range(3),yticks=range(3),xticklabels=class_names,yticklabels=class_names,
           xlabel="Prediksi",ylabel="Label",title="Confusion matrix")
    fig.tight_layout()
    fig.savefig(output_path,dpi=150)
    plt.close(fig)
    return cm

def predict_batches(model,images,batch=32):
    return np.concatenate([model(images[i:i+batch],training=False).numpy() for i in range(0,len(images),batch)])

def evaluate_run(run,robustness=True):
    import tensorflow as tf
    from PIL import Image, ImageFilter
    run=resolve_path(run)
    meta=json.loads((run/"run.json").read_text())
    df,dataset=load_bundle(run/"dataset")
    if 'development_fold' in dataset:
        raise ValueError('Development CV run: scoring the locked test is prohibited')
    if dataset["fingerprint"] != meta["dataset_fingerprint"] or meta["classes"] != CLASS_NAMES:
        raise ValueError("Model metadata does not match dataset/class order")
    model=tf.keras.models.load_model(run/"model.keras",compile=False)
    test=df[df.split=="test"].reset_index(drop=True)
    images,labels=load_arrays(test,meta["config"]["image_size"])
    probs=temperature_scale(predict_batches(model,images),meta["temperature"])
    result=metrics(labels,probs)
    result["mode"]=dataset["mode"]
    result["dataset_fingerprint"]=dataset["fingerprint"]
    result["generalization_claim"]="Own-photo held-out evaluation; assess group counts and capture diversity"
    result["uncertainty"]=group_bootstrap(test,probs,meta["config"]["seed"])
    if robustness:
        scenarios={"darker":np.clip(images*.8,0,1),"brighter":np.clip(images*1.2,0,1),
                   "blur":np.stack([np.asarray(Image.fromarray((im*255).astype("uint8")).filter(
                       ImageFilter.GaussianBlur(1)),dtype=np.float32)/255 for im in images])}
        result["robustness"]={name:metrics(labels,temperature_scale(predict_batches(model,x),meta["temperature"]))
                              for name,x in scenarios.items()}
    rows=test.copy()
    rows["prediction"]=[CLASS_NAMES[i] for i in probs.argmax(axis=1)]
    rows["confidence"]=probs.max(axis=1)
    for idx,name in enumerate(CLASS_NAMES):
        rows[f"prob_{name}"]=probs[:,idx]
    rows.to_csv(run/"predictions_test.csv",index=False)
    rows[rows.label!=rows.prediction].to_csv(run/"misclassifications.csv",index=False)
    plot_confusion_matrix(labels,probs.argmax(axis=1),run/"confusion_matrix.png")
    (run/"evaluation.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in {"robustness","report"}},indent=2))
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",required=True)
    parser.add_argument("--no-robustness",action="store_true")
    args=parser.parse_args()
    evaluate_run(args.run,not args.no_robustness)

if __name__=="__main__":
    main()
