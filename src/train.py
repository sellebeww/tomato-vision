"""Reproducible scratch CNN training with validation-only selection and frozen data."""
import argparse
from dataclasses import replace
import hashlib
import json
import platform
import shutil
import time
from pathlib import Path
import numpy as np
from src.config import ExperimentConfig, CLASS_NAMES, ROOT_DIR, OUTPUTS_DIR, resolve_path
from src.manifest import load_bundle, relative
from src.preprocessing import load_arrays, balanced_indices, array_dataset
from src.evaluate import metrics, calibration_policy, temperature_scale, predict_batches
from src.model import build_model

def plot_learning_curve(history,path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,key in zip(axes,("accuracy","loss")):
        ax.plot(history[key],label="Train (balanced/augmented)")
        ax.plot(history["val_"+key],label="Validation")
        ax.set(xlabel="Epoch",ylabel=key)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path,dpi=150)
    plt.close(fig)

def train_run(dataset,run,config):
    import tensorflow as tf
    import keras
    tf.keras.utils.set_random_seed(config.seed)
    tf.config.experimental.enable_op_determinism()
    try:
        tf.config.threading.set_intra_op_parallelism_threads(config.threads)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        print("TensorFlow was already initialized; keeping its existing thread settings",flush=True)
    df,data_meta=load_bundle(dataset)
    run=resolve_path(run)
    if run.exists():
        raise FileExistsError(f"Use a NEW run directory; refusing to overwrite {run}")
    run.mkdir(parents=True)
    shutil.copytree(resolve_path(dataset),run/"dataset")
    shutil.copytree(ROOT_DIR/"src",run/"source",ignore=shutil.ignore_patterns("__pycache__"))
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (run/"source").glob("*.py")}
    train=df[df.split=="train"]
    val=df[df.split=="val"]
    x,y=load_arrays(train,config.image_size)
    vx,vy=load_arrays(val,config.image_size)
    model=build_model(config=config)
    rng=np.random.default_rng(config.seed)
    history={k:[] for k in ("loss","accuracy","val_loss","val_accuracy","learning_rate")}
    best=float("inf")
    stale=0
    started=time.monotonic()
    # Epoch loop enables exactly balanced finite draws with fresh augmentation.
    for epoch in range(config.epochs):
        ids=balanced_indices(y,max(config.train_samples_per_class,int(np.bincount(y).max())),rng)
        h=model.fit(array_dataset(x[ids],y[ids],config.batch_size),
                    validation_data=array_dataset(vx,vy,config.batch_size),epochs=1,verbose=0,shuffle=False).history
        for key in ("loss","accuracy","val_loss","val_accuracy"):
            history[key].append(float(h[key][0]))
        lr=float(tf.keras.backend.get_value(model.optimizer.learning_rate))
        history["learning_rate"].append(lr)
        loss=history["val_loss"][-1]
        if not all(np.isfinite(history[key][-1]) for key in ('loss','accuracy','val_loss','val_accuracy')):
            raise ValueError('Non-finite training metrics; run stopped without publishing a model')
        print(f"{run.name} epoch {epoch+1}: loss={h['loss'][0]:.4f} val_loss={loss:.4f} "
              f"acc={h['accuracy'][0]:.3f} val_acc={h['val_accuracy'][0]:.3f}",flush=True)
        if loss<best-1e-5:
            best=loss
            stale=0
            model.save(run/"model.keras")
        else:
            stale+=1
            if stale%3==0:
                model.optimizer.learning_rate.assign(max(lr*.5,1e-6))
            if stale>=config.patience:
                break
        (run/"history.json").write_text(json.dumps(history,indent=2))
    (run/"history.json").write_text(json.dumps(history,indent=2))
    model=tf.keras.models.load_model(run/"model.keras",compile=False)
    vp=predict_batches(model,vx)
    calibration=calibration_policy(vy,vp,val.group_id.tolist())
    temperature=calibration['temperature']
    train_metrics=metrics(y,predict_batches(model,x))
    val_metrics=metrics(vy,vp)
    validation_rows=val[['filepath','sha256','group_id','label','label_idx']].copy()
    for i,name in enumerate(CLASS_NAMES):
        validation_rows['prob_'+name]=vp[:,i]
    validation_rows.to_csv(run/'validation_predictions.csv',index=False)
    meta={"config":config.to_dict(),"classes":CLASS_NAMES,"temperature":temperature,
          "calibration":calibration,
          "temperature_note":calibration['note'],
          "dataset_fingerprint":data_meta["fingerprint"],"mode":data_meta["mode"],
          "best_epoch":int(np.argmin(history["val_loss"])+1),"epochs_run":len(history["loss"]),
          "elapsed_seconds":time.monotonic()-started,"train_unique":len(train),
          "train_draws_per_epoch":len(ids),"model_parameters":model.count_params(),
          "train_clean":train_metrics,"validation":val_metrics,
          "validation_calibrated":metrics(vy,temperature_scale(vp,temperature)),
          "train_val_accuracy_gap":train_metrics["accuracy"]-val_metrics["accuracy"],
          "runtime":{"python":platform.python_version(),"tensorflow":tf.__version__,"keras":keras.__version__,
                     "platform":platform.platform()},
          "source_sha256":source_hashes,"weights":"random initialization; no pretrained weights",
          "production_ready":False}
    (run/"run.json").write_text(json.dumps(meta,indent=2))
    plot_learning_curve(history,run/"learning_curve.png")
    print(f"Saved {relative(run)}; selected using validation only",flush=True)
    return meta

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",required=True)
    parser.add_argument("--run",required=True)
    parser.add_argument("--config")
    parser.add_argument("--architecture",choices=["baseline","regularized"])
    parser.add_argument("--epochs",type=int)
    parser.add_argument("--samples-per-class",type=int)
    args=parser.parse_args()
    c=ExperimentConfig.load(args.config)
    overrides={k:v for k,v in {"architecture":args.architecture,"epochs":args.epochs,
                               "train_samples_per_class":args.samples_per_class}.items() if v is not None}
    train_run(args.dataset,args.run,replace(c,**overrides))

if __name__=="__main__":
    main()
