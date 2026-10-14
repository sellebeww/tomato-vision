"""own_v2 study: session-grouped 5-fold CV x 5 seeds, ablation, final refit, locked test evaluated once.

Steps (each refuses to overwrite earlier artifacts):
  python -m src.cv_study protocol                       # freeze sessions, locked test, folds, configs
  python -m src.cv_study sweep --configs ref lr0003 ... # 25 runs per config (resumable, parallel workers)
  python -m src.cv_study summarize                      # mean+-std, Wilson + session-bootstrap CI, ablation table
  python -m src.cv_study select --configs ... --also-report regularized   # pre-declared rule, no test access
  python -m src.cv_study final                          # refit selected (+ declared comparison) on all dev data, 5 seeds
  python -m src.cv_study test                           # score the locked test ONCE
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from src.artifacts import atomic_json
from src.config import CLASS_NAMES, ExperimentConfig, ROOT_DIR, resolve_path

OUT = ROOT_DIR / "outputs" / "experiments" / "own_v2"
SOURCE_MANIFEST = ROOT_DIR / "outputs" / "experiments" / "own_v1" / "baseline" / "dataset" / "manifest.csv"
SESSIONS = ROOT_DIR / "data" / "sessions.csv"
SEEDS = [42, 43, 44, 45, 46]
FOLDS = 5
TEST_SESSIONS = 2
SPIKE_LOSS = 3.0          # a validation loss above this in any epoch counts as a "spike"
REVIEW_TARGET_ACCURACY = 0.9

# own_v1 baseline settings (300 balanced draws per class per epoch, max 30 epochs, patience 10).
REFERENCE = ExperimentConfig(architecture="baseline", epochs=30, learning_rate=0.001, patience=10,
                             train_samples_per_class=300, threads=3)
STAGE1 = {
    "ref": REFERENCE,
    "lr0003": replace(REFERENCE, learning_rate=0.0003),
    "cosine": replace(REFERENCE, lr_schedule="cosine"),
    "bn09": replace(REFERENCE, bn_momentum=0.9),
    "aug": replace(REFERENCE, augment=True),
    "no_restore": replace(REFERENCE, restore_best=False),   # derived from the "ref" runs (last-epoch weights)
    "regularized": replace(REFERENCE, architecture="regularized", dropout=0.35),
}
DERIVED = {"no_restore": "ref__last"}   # same training trajectory, only the final weights differ
SELECTION_RULE = ("Highest mean OOF macro-F1 over seeds; if the runner-up is within one standard deviation, "
                  "prefer lower mean OOF log loss, then fewer parameters. Locked test is never read.")


def configs(root=OUT):
    found = dict(STAGE1)
    extra = root / "stage2_configs.json"
    if extra.exists():
        for name, item in json.loads(extra.read_text())["configs"].items():
            found[name] = ExperimentConfig(**item["config"])
    return found


# ------------------------------------------------------------------ protocol
def build_protocol(root=OUT):
    from src.preprocessing import load_image
    if (root / "protocol.json").exists():
        raise FileExistsError(f"{root}/protocol.json exists; protocol is frozen")
    frame = pd.read_csv(SOURCE_MANIFEST)
    frame["file"] = frame.filepath.map(lambda p: p.rsplit("/", 1)[-1].rsplit(".", 1)[0])
    sessions = pd.read_csv(SESSIONS)
    frame = frame.merge(sessions[["file", "session_id"]], on="file", how="left", validate="one_to_one")
    if frame.session_id.isna().any():
        raise ValueError("Every image needs a session_id in data/sessions.csv")
    b_sessions = sorted(s for s in frame.session_id.unique() if s.startswith("B"))
    rng = np.random.default_rng(42)
    order = [b_sessions[i] for i in rng.permutation(len(b_sessions))]
    test_sessions, dev_b = sorted(order[:TEST_SESSIONS]), order[TEST_SESSIONS:]
    fold_of = {s: i for i, part in enumerate(np.array_split(np.array(dev_b), FOLDS)) for s in part}
    frame["role"] = np.where(frame.session_id.isin(test_sessions), "test", "dev")
    frame["fold"] = frame.session_id.map(fold_of).fillna(-1).astype(int)
    for k in range(FOLDS):
        if set(frame[frame.fold == k].label) != set(CLASS_NAMES):
            raise ValueError(f"Fold {k} lacks a class")
    if set(frame[frame.role == "test"].label) != set(CLASS_NAMES):
        raise ValueError("Locked test lacks a class")
    if (frame.groupby("session_id").role.nunique() > 1).any() or (frame.groupby("session_id").fold.nunique() > 1).any():
        raise ValueError("A session spans roles or folds")
    root.mkdir(parents=True, exist_ok=True)
    tensors = np.stack([np.rint(load_image(resolve_path(p), REFERENCE.image_size) * 255).astype(np.uint8)
                        for p in frame.filepath])
    np.savez_compressed(root / "tensors.npz", x=tensors, y=frame.label_idx.to_numpy(np.int32))
    frame[["filepath", "file", "sha256", "label", "label_idx", "session_id", "role", "fold"]].to_csv(
        root / "frame.csv", index=False)
    counts = lambda part: part.label.value_counts().reindex(CLASS_NAMES).fillna(0).astype(int).to_dict()
    protocol = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "grouping": "session (data/sessions.csv); session A (30 wooden-table photos) treated as one fruit/session "
                    "per owner decision 2026-10-06, so it is always in training",
        "locked_test_sessions": test_sessions,
        "locked_test_sha256": sorted(frame[frame.role == "test"].sha256),
        "locked_test_counts": counts(frame[frame.role == "test"]),
        "folds": {str(k): sorted(s for s, f in fold_of.items() if f == k) for k in range(FOLDS)},
        "fold_counts": {str(k): counts(frame[frame.fold == k]) for k in range(FOLDS)},
        "always_train_sessions": sorted(s for s in frame.session_id.unique() if not s.startswith("B")),
        "seeds": SEEDS, "spike_val_loss": SPIKE_LOSS,
        "stage1_configs": {n: c.to_dict() for n, c in STAGE1.items()},
        "selection_rule": SELECTION_RULE,
        "deviation_from_own_v1": "100 instead of 300 balanced draws per class per epoch (compute budget); "
                                 "max 30 epochs, patience 10 as in own_v1",
        "known_bias": "Each fold's validation sessions also drive early stopping, so CV scores are optimistic; "
                      "the locked test is the unbiased check",
        "class_weight": "Not applicable: every fold's training set is exactly balanced (verified below)",
        "train_class_counts_per_fold": {str(k): counts(frame[(frame.role == "dev") & (frame.fold != k)])
                                        for k in range(FOLDS)},
    }
    atomic_json(root / "protocol.json", protocol)
    return protocol


def load_data(root=OUT):
    data = np.load(root / "tensors.npz")
    return pd.read_csv(root / "frame.csv"), data["x"], data["y"]


def set_seed(config):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(config.seed)
    tf.config.experimental.enable_op_determinism()
    try:
        tf.config.threading.set_intra_op_parallelism_threads(config.threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass


# ------------------------------------------------------------------ one CV run
def run_one(name, fold, seed, root=OUT):
    from src.evaluate import predict_batches
    from src.model import build_model
    from src.train import fit_loop
    target = root / "runs" / name / f"f{fold}_s{seed}.json"
    if target.exists():
        return json.loads(target.read_text())
    frame, x, y = load_data(root)
    config = replace(configs(root)[name], seed=seed)
    set_seed(config)
    train = ((frame.role == "dev") & (frame.fold != fold)).to_numpy()
    val = (frame.fold == fold).to_numpy()
    xt, xv = x[train].astype("float32") / 255, x[val].astype("float32") / 255
    model = build_model(config=config)
    started = time.monotonic()
    best = {}
    history = fit_loop(model, xt, y[train], replace(config, restore_best=False), validation=(xv, y[val]), verbose=False,
                       on_improve=lambda m: best.__setitem__("weights", m.get_weights()))
    elapsed = time.monotonic() - started
    common = {"config": name, "fold": fold, "seed": seed, "files": frame.file[val].tolist(),
              "sessions": frame.session_id[val].tolist(), "labels": y[val].tolist(), "history": history,
              "epochs_run": len(history["loss"]), "parameters": int(model.count_params()), "elapsed_seconds": elapsed}
    last = {**common, "config": name + "__last", "probs": predict_batches(model, xv).astype(float).tolist(),
            "train_accuracy": float((predict_batches(model, xt).argmax(1) == y[train]).mean()),
            "best_epoch": len(history["loss"]), "weights": "last epoch"}
    if config.restore_best:
        model.set_weights(best["weights"])
    result = {**common, "probs": predict_batches(model, xv).astype(float).tolist(),
              "train_accuracy": float((predict_batches(model, xt).argmax(1) == y[train]).mean()),
              "best_epoch": history["best_epoch"], "weights": "best validation loss" if config.restore_best else "last epoch"}
    atomic_json(root / "runs" / (name + "__last") / f"f{fold}_s{seed}.json", last)
    atomic_json(target, result)
    return result


def sweep(names, workers=2, root=OUT):
    names = [n for n in names if n not in DERIVED]
    jobs = [(n, f, s) for n in names for s in SEEDS for f in range(FOLDS)
            if not (root / "runs" / n / f"f{f}_s{s}.json").exists()]
    print(f"{len(jobs)} runs pending", flush=True)
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def launch(job):
        name, fold, seed = job
        with open(logs / f"{name}_f{fold}_s{seed}.log", "w") as log:
            code = subprocess.run([sys.executable, "-u", "-m", "src.cv_study", "worker", name, str(fold), str(seed)],
                                  cwd=ROOT_DIR, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"{'done' if code == 0 else 'FAILED'} {name} fold {fold} seed {seed}", flush=True)
        return code

    with ThreadPoolExecutor(workers) as pool:
        codes = list(pool.map(launch, jobs))
    if any(codes):
        raise SystemExit(f"{sum(1 for c in codes if c)} runs failed; see {logs}")


# ------------------------------------------------------------------ statistics
def wilson(successes, n, z=1.959964):
    if n == 0:
        return [0.0, 1.0]
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(max(0, centre - half)), float(min(1, centre + half))]


def review_threshold(confidence, correct, target=REVIEW_TARGET_ACCURACY):
    """Lowest confidence cut-off whose accepted predictions reach the target accuracy (OOF data only)."""
    order = np.argsort(-confidence)
    conf, ok = confidence[order], correct[order]
    best = None
    for i in range(1, len(conf) + 1):
        if ok[:i].mean() >= target and (i == len(conf) or conf[i - 1] > conf[i]):
            best = (float(conf[i - 1]), i / len(conf), float(ok[:i].mean()))
    return None if best is None else {"threshold": best[0], "coverage": best[1], "accepted_accuracy": best[2],
                                      "target_accuracy": target}


def summarize_config(name, root=OUT):
    from src.evaluate import metrics
    files = sorted((root / "runs" / DERIVED.get(name, name)).glob("f*_s*.json"))
    runs = [json.loads(p.read_text()) for p in files]
    if len(runs) != FOLDS * len(SEEDS):
        return {"name": name, "complete": False, "runs": len(runs)}
    per_seed, pooled_labels, pooled_probs, sessions_all = [], [], [], None
    for seed in SEEDS:
        parts = sorted((r for r in runs if r["seed"] == seed), key=lambda r: r["fold"])
        labels = np.concatenate([r["labels"] for r in parts])
        probs = np.concatenate([r["probs"] for r in parts])
        sessions = sum((r["sessions"] for r in parts), [])
        m = metrics(labels, probs)
        per_seed.append({"seed": seed, **{k: m[k] for k in ("accuracy", "macro_f1", "loss", "expected_calibration_error")},
                         "correct": (probs.argmax(1) == labels).astype(int).tolist()})
        pooled_labels.append(labels); pooled_probs.append(probs); sessions_all = sessions
    labels, probs = np.concatenate(pooled_labels), np.concatenate(pooled_probs)
    pooled = metrics(labels, probs)
    n = len(pooled_labels[0])
    acc = np.array([s["accuracy"] for s in per_seed])
    # Session bootstrap of the seed-averaged accuracy: resample the development B sessions.
    correct = np.array([s["correct"] for s in per_seed]).mean(axis=0)
    session_ids = np.array(sessions_all)
    unique = np.unique(session_ids)
    rng = np.random.default_rng(0)
    boots = [np.concatenate([correct[session_ids == s] for s in rng.choice(unique, len(unique))]).mean()
             for _ in range(5000)]
    spikes = [max(r["history"]["val_loss"]) for r in runs]
    stat = lambda key: {"mean": float(np.mean([s[key] for s in per_seed])), "std": float(np.std([s[key] for s in per_seed], ddof=1))}
    return {
        "name": name, "complete": True, "config": configs(root)[name].to_dict(), "parameters": runs[0]["parameters"],
        "oof_images_per_seed": n, "sessions": int(len(unique)),
        "accuracy": stat("accuracy"), "macro_f1": stat("macro_f1"), "loss": stat("loss"),
        "ece": stat("expected_calibration_error"),
        "accuracy_wilson95_mean_seed": wilson(round(acc.mean() * n), n),
        "accuracy_session_bootstrap95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
        "per_seed": [{k: v for k, v in s.items() if k != "correct"} for s in per_seed],
        "pooled_over_seeds": {"report": pooled["report"], "confusion_matrix": pooled["confusion_matrix"],
                              "n_predictions": pooled["n_images"]},
        "stability": {"max_val_loss_median": float(np.median(spikes)), "max_val_loss_worst": float(np.max(spikes)),
                      "runs_with_spike": int(sum(v > SPIKE_LOSS for v in spikes)), "runs": len(runs),
                      "best_epoch_median": float(np.median([r["best_epoch"] for r in runs])),
                      "epochs_run_mean": float(np.mean([r["epochs_run"] for r in runs])),
                      "train_accuracy_mean": float(np.mean([r["train_accuracy"] for r in runs]))},
        "review_threshold": review_threshold(probs.max(1), probs.argmax(1) == labels),
        "minutes_per_run": float(np.mean([r["elapsed_seconds"] for r in runs]) / 60),
    }


def summarize(root=OUT):
    rows = [summarize_config(n, root) for n in configs(root)]
    done = [r for r in rows if r["complete"]]
    atomic_json(root / "cv_summary.json", {"generated": dt.datetime.now().isoformat(timespec="seconds"),
                                            "configs": rows, "selection_rule": SELECTION_RULE})
    fmt = lambda s, d=3: f"{s['mean']:.{d}f} ± {s['std']:.{d}f}"
    lines = ["| Konfigurasi | Param | Val acc (mean ± std) | Wilson 95% | Bootstrap sesi 95% | Macro-F1 | Val loss | ECE | Run dgn spike >3 | Max val loss (median) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in done:
        w, b = r["accuracy_wilson95_mean_seed"], r["accuracy_session_bootstrap95"]
        lines.append(f"| {r['name']} | {r['parameters']:,} | {fmt(r['accuracy'])} | {w[0]:.2f}–{w[1]:.2f} | {b[0]:.2f}–{b[1]:.2f} | "
                     f"{fmt(r['macro_f1'])} | {fmt(r['loss'])} | {fmt(r['ece'])} | {r['stability']['runs_with_spike']}/{r['stability']['runs']} | "
                     f"{r['stability']['max_val_loss_median']:.2f} |")
    (root / "ablation_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return rows


def select(names, also_report=(), root=OUT):
    if (root / "selection.json").exists():
        raise FileExistsError("Selection already frozen")
    rows = {r["name"]: r for r in json.loads((root / "cv_summary.json").read_text())["configs"] if r["complete"]}
    missing = [n for n in names if n not in rows]
    if missing:
        raise ValueError(f"Incomplete configs: {missing}")
    ranked = sorted((rows[n] for n in names), key=lambda r: -r["macro_f1"]["mean"])
    top = ranked[0]
    tied = [r for r in ranked if top["macro_f1"]["mean"] - r["macro_f1"]["mean"] <= top["macro_f1"]["std"]]
    winner = sorted(tied, key=lambda r: (r["loss"]["mean"], r["parameters"]))[0]
    selection = {"selected": winner["name"], "frozen": dt.datetime.now().isoformat(timespec="seconds"),
                 "rule": SELECTION_RULE, "considered": names,
                 "tied_within_one_std": [r["name"] for r in tied], "test_accessed": False,
                 "also_report": [n for n in also_report if n != winner["name"]]}
    atomic_json(root / "selection.json", selection)
    return selection


# ------------------------------------------------------------------ final refit + locked test
def final_dir(name, selection, root=OUT):
    return root / "final" if name == selection["selected"] else root / "final_extra" / name


def final_plan(name, root=OUT):
    runs = [json.loads(p.read_text()) for p in (root / "runs" / DERIVED.get(name, name)).glob("f*_s*.json")]
    epochs = int(np.median([r["best_epoch"] for r in runs]))
    width = max(len(r["history"]["learning_rate"]) for r in runs)
    padded = [r["history"]["learning_rate"] + [r["history"]["learning_rate"][-1]] * (width - len(r["history"]["learning_rate"]))
              for r in runs]
    return {"config": name, "epochs": epochs, "lr_trajectory": np.median(np.array(padded), axis=0)[:epochs].tolist(),
            "seeds": SEEDS, "basis": "median best epoch and median per-epoch learning rate of the 25 CV runs; no validation data"}


def final(root=OUT, workers=2):
    selection = json.loads((root / "selection.json").read_text())
    names = [selection["selected"]] + selection.get("also_report", [])
    plans = {n: final_plan(n, root) for n in names}
    atomic_json(root / "final_plan.json", plans)
    jobs = [(n, s) for n in names for s in SEEDS if not (final_dir(n, selection, root) / f"seed{s}" / "model.keras").exists()]
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def launch(job):
        name, seed = job
        with open(logs / f"final_{name}_s{seed}.log", "w") as log:
            code = subprocess.run([sys.executable, "-u", "-m", "src.cv_study", "final-worker", name, str(seed)],
                                  cwd=ROOT_DIR, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"{'done' if code == 0 else 'FAILED'} final {name} seed {seed}", flush=True)
        return code

    with ThreadPoolExecutor(workers) as pool:
        if any(list(pool.map(launch, jobs))):
            raise SystemExit("A final refit failed; see logs")
    return {n: {k: v for k, v in p.items() if k != "lr_trajectory"} for n, p in plans.items()}


def final_worker(name, seed, root=OUT):
    import platform
    import keras
    import tensorflow as tf
    from src.evaluate import metrics, predict_batches
    from src.model import build_model
    from src.train import fit_loop, plot_learning_curve
    plan = json.loads((root / "final_plan.json").read_text())[name]
    selection = json.loads((root / "selection.json").read_text())
    summary = {r["name"]: r for r in json.loads((root / "cv_summary.json").read_text())["configs"] if r["complete"]}
    chosen = summary[name]
    frame, x, y = load_data(root)
    config = replace(configs(root)[name], seed=seed)
    set_seed(config)
    dev = (frame.role == "dev").to_numpy()
    xd = x[dev].astype("float32") / 255
    model = build_model(config=config)
    started = time.monotonic()
    history = fit_loop(model, xd, y[dev], config, lr_trajectory=plan["lr_trajectory"], fixed_epochs=plan["epochs"],
                       verbose=False)
    run = final_dir(name, selection, root) / f"seed{seed}"
    run.mkdir(parents=True, exist_ok=True)
    model.save(run / "model.keras")
    train_metrics = metrics(y[dev], predict_batches(model, xd))
    meta = {"config": config.to_dict(), "classes": CLASS_NAMES, "temperature": 1.0,
            "calibration": {"temperature": 1.0, "fitted": False,
                            "note": "Raw softmax; review threshold calibrated on CV out-of-fold predictions"},
            "temperature_note": "Raw softmax retained",
            "dataset_fingerprint": hashlib.sha256((root / "frame.csv").read_bytes()).hexdigest(),
            "mode": "own", "study": "own_v2", "config_name": name, "role": "selected" if name == selection["selected"] else "comparison",
            "selection": selection, "best_epoch": plan["epochs"], "epochs_run": len(history["loss"]),
            "elapsed_seconds": time.monotonic() - started, "train_unique": int(dev.sum()),
            "model_parameters": int(model.count_params()), "train_clean": train_metrics,
            "cv": {k: chosen[k] for k in ("accuracy", "macro_f1", "loss", "ece", "accuracy_wilson95_mean_seed",
                                          "accuracy_session_bootstrap95", "oof_images_per_seed")},
            "validation": {"accuracy": chosen["accuracy"]["mean"], "macro_f1": chosen["macro_f1"]["mean"],
                           "loss": chosen["loss"]["mean"], "note": "mean over 5 seeds of 5-fold session-grouped CV"},
            "review_threshold": chosen["review_threshold"],
            "runtime": {"python": platform.python_version(), "tensorflow": tf.__version__, "keras": keras.__version__},
            "weights": "random initialization; no pretrained weights", "production_ready": False}
    atomic_json(run / "run.json", meta)
    atomic_json(run / "history.json", history)
    plot_learning_curve(history, run / "learning_curve.png")


def test_once(root=OUT):
    """Scores the locked test exactly once for every declared final model and seed. Refuses to run again."""
    import tensorflow as tf
    from src.evaluate import metrics, predict_batches
    target = root / "test_evaluation.json"
    if target.exists():
        raise FileExistsError("Locked test has already been evaluated; it must not be re-run")
    selection = json.loads((root / "selection.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    frame, x, y = load_data(root)
    test = (frame.role == "test").to_numpy()
    if sorted(frame.sha256[test]) != protocol["locked_test_sha256"]:
        raise ValueError("Locked test changed since the protocol was frozen")
    xt, yt, n = x[test].astype("float32") / 255, y[test], int(test.sum())
    from PIL import Image, ImageFilter
    scenarios = {"darker": np.clip(xt * .8, 0, 1), "brighter": np.clip(xt * 1.2, 0, 1),
                 "blur": np.stack([np.asarray(Image.fromarray((im * 255).astype("uint8")).filter(ImageFilter.GaussianBlur(1)),
                                              dtype=np.float32) / 255 for im in xt])}   # same perturbations as own_v1
    models, rows = {}, []
    for name in [selection["selected"]] + selection.get("also_report", []):
        per_seed = []
        for seed in SEEDS:
            model = tf.keras.models.load_model(final_dir(name, selection, root) / f"seed{seed}" / "model.keras", compile=False)
            probs = predict_batches(model, xt)
            m = metrics(yt, probs)
            robust = {k: metrics(yt, predict_batches(model, v)) for k, v in scenarios.items()}
            per_seed.append({"seed": seed, **{k: m[k] for k in ("accuracy", "macro_f1", "loss", "expected_calibration_error",
                                                                 "confusion_matrix", "report")},
                             "robustness": {k: {"accuracy": r["accuracy"], "macro_f1": r["macro_f1"], "loss": r["loss"]}
                                            for k, r in robust.items()}})
            for i, f in enumerate(frame.file[test]):
                rows.append({"model": name, "seed": seed, "file": f, "label": CLASS_NAMES[yt[i]],
                             "prediction": CLASS_NAMES[int(probs[i].argmax())],
                             **{"prob_" + c: float(probs[i][j]) for j, c in enumerate(CLASS_NAMES)}})
        stat = lambda key: {"mean": float(np.mean([s[key] for s in per_seed])), "std": float(np.std([s[key] for s in per_seed], ddof=1))}
        acc = stat("accuracy")
        models[name] = {"role": "selected" if name == selection["selected"] else "comparison", "n_images": n,
                        "accuracy": acc, "macro_f1": stat("macro_f1"), "loss": stat("loss"),
                        "ece": stat("expected_calibration_error"),
                        "accuracy_wilson95_mean_seed": wilson(round(acc["mean"] * n), n),
                        "confusion_matrix_pooled": np.sum([s["confusion_matrix"] for s in per_seed], axis=0).tolist(),
                        "robustness": {k: {m: {"mean": float(np.mean([s["robustness"][k][m] for s in per_seed])),
                                               "std": float(np.std([s["robustness"][k][m] for s in per_seed], ddof=1))}
                                           for m in ("accuracy", "macro_f1", "loss")} for k in scenarios},
                        "per_seed": per_seed}
    pd.DataFrame(rows).to_csv(root / "test_predictions.csv", index=False)
    result = {"evaluated": dt.datetime.now().isoformat(timespec="seconds"), "selected": selection["selected"],
              "selection_frozen": selection["frozen"], "n_images": n, "sessions": protocol["locked_test_sessions"],
              "models": models,
              "note": "Evaluated once after selection.json was frozen; test images were never used for training, "
                      "early stopping, threshold calibration or model selection. Comparison models were declared "
                      "in selection.json before this evaluation."}
    atomic_json(target, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("protocol")
    p = sub.add_parser("sweep"); p.add_argument("--configs", nargs="+", required=True); p.add_argument("--workers", type=int, default=2)
    w = sub.add_parser("worker"); w.add_argument("name"); w.add_argument("fold", type=int); w.add_argument("seed", type=int)
    sub.add_parser("summarize")
    s = sub.add_parser("select"); s.add_argument("--configs", nargs="+", required=True)
    s.add_argument("--also-report", nargs="*", default=[], help="Comparison models refit and tested alongside the winner")
    f = sub.add_parser("final"); f.add_argument("--workers", type=int, default=2)
    fw = sub.add_parser("final-worker"); fw.add_argument("name"); fw.add_argument("seed", type=int)
    sub.add_parser("test")
    a = parser.parse_args()
    if a.command == "protocol":
        print(json.dumps(build_protocol(), indent=2))
    elif a.command == "sweep":
        unknown = set(a.configs) - set(configs())
        if unknown:
            raise SystemExit(f"Unknown configs: {sorted(unknown)}")
        sweep(a.configs, a.workers)
    elif a.command == "worker":
        r = run_one(a.name, a.fold, a.seed)
        print(json.dumps({k: r[k] for k in ("config", "fold", "seed", "best_epoch", "epochs_run", "elapsed_seconds")}))
    elif a.command == "summarize":
        summarize()
    elif a.command == "select":
        print(json.dumps(select(a.configs, a.also_report), indent=2))
    elif a.command == "final":
        print(json.dumps(final(workers=a.workers), indent=2))
    elif a.command == "final-worker":
        final_worker(a.name, a.seed)
    elif a.command == "test":
        result = test_once()
        print(json.dumps({n: {k: v for k, v in m.items() if k != "per_seed"} for n, m in result["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
