"""Export trained CNNs to a static web bundle (GitHub Pages); inference then runs in plain JavaScript.

Any functional Keras graph built from SUPPORTED_LAYERS is exported (baseline and regularized CNNs).
BatchNorm directly after a convolution is folded into it. Every export is checked against Keras and
refused when the maximum absolute probability difference exceeds MAX_PARITY_ERROR.

  python -m src.export_web --study own_v2            # models + report for the selected own_v2 study
  python -m src.export_web --study own_v1            # legacy single-split study
  python -m src.export_web ... --no-examples         # publish without example photos

Besides the web bundle this writes site/model_info.json (numbers shown in the demo's "Tentang demo ini" box and
quoted by README/report; never typed by hand) and copies the SELECTED Keras model to models/release/<id>/ so that
`python -m src.predict` runs the very same network as the browser. `node tests/js_parity.mjs --write-info` adds the
measured JavaScript-vs-Keras parity to model_info.json.
"""
import argparse
import base64
import hashlib
import io
import datetime as dt
import json
import platform
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from src.artifacts import atomic_json
from src.config import CLASS_NAMES, ROOT_DIR, resolve_path
from src.preprocessing import load_image

SITE_DIR = ROOT_DIR / "site"
RELEASE_DIR = ROOT_DIR / "models" / "release"
FIXTURE_DIR = ROOT_DIR / "tests" / "fixtures"
MAX_PARITY_ERROR = 1e-4
FORMAT = "tomato-vision-graph/2"
IDENTITY_LAYERS = {"Dropout", "SpatialDropout2D", "GaussianNoise", "GaussianDropout"}
AUGMENTATION_LAYERS = {"InputLayer", "RandomFlip", "RandomRotation", "RandomTranslation", "RandomZoom",
                       "RandomContrast", "RandomBrightness"}
SUPPORTED_LAYERS = sorted({"InputLayer", "Conv2D", "SeparableConv2D", "BatchNormalization", "GroupNormalization",
                           "ReLU", "Activation(relu|linear|softmax)", "Add", "MaxPooling2D", "GlobalAveragePooling2D",
                           "Dense(relu|linear|softmax)", "Sequential(augmentation only)"} | IDENTITY_LAYERS)


# ------------------------------------------------------------------ graph export
def _sources(inbound):
    names = []

    def walk(value):
        if isinstance(value, dict):
            history = value.get("config", {}).get("keras_history") if value.get("class_name") == "__keras_tensor__" else None
            if history:
                names.append(history[0])
            else:
                for item in value.values():
                    walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    if len(inbound) > 1:
        raise ValueError("Shared layers are not supported")
    walk(inbound[0]["args"] if inbound else [])
    return names


def export_graph(model):
    """Return (nodes, flat float32 weights, output name). Raises ValueError listing unsupported layers."""
    layers = {layer.name: layer for layer in model.layers}
    config = model.get_config()
    unsupported = sorted({f"{l['class_name']} ({l['name']})" for l in config["layers"] if not _supported(l, layers)})
    if unsupported:
        raise ValueError("Unsupported layers for the web export: " + ", ".join(unsupported) +
                         ". Supported: " + ", ".join(SUPPORTED_LAYERS))
    consumers = {}
    for item in config["layers"]:
        for source in _sources(item["inbound_nodes"]):
            consumers[source] = consumers.get(source, 0) + 1
    nodes, arrays, alias = [], [], {}

    def ref(name):
        while name in alias:
            name = alias[name]
        return name

    def push(array):
        arrays.append(np.ascontiguousarray(array, dtype="float64").ravel())
        return len(arrays) - 1

    by_name = {}
    for item in config["layers"]:
        kind, name, cfg = item["class_name"], item["name"], item["config"]
        inputs = [ref(s) for s in _sources(item["inbound_nodes"])]
        layer = layers[name]
        if kind == "InputLayer":
            node = {"name": name, "op": "input"}
        elif kind == "Sequential" or kind in IDENTITY_LAYERS:
            alias[name] = inputs[0]
            continue
        elif kind == "ReLU":
            node = {"name": name, "op": "relu", "max": cfg.get("max_value")}
        elif kind == "Activation":
            if cfg["activation"] == "linear":
                alias[name] = inputs[0]
                continue
            node = {"name": name, "op": "relu", "max": None} if cfg["activation"] == "relu" else {"name": name, "op": "softmax"}
        elif kind in ("Conv2D", "SeparableConv2D"):
            weights = layer.get_weights()
            node = {"name": name, "op": "conv" if kind == "Conv2D" else "sepconv", "k": int(cfg["kernel_size"][0]),
                    "stride": int(cfg["strides"][0]), "padding": cfg["padding"], "act": cfg["activation"]}
            if kind == "Conv2D":
                node.update({"cin": int(weights[0].shape[2]), "cout": int(weights[0].shape[3]), "w": push(weights[0])})
            else:
                node.update({"cin": int(weights[0].shape[2]), "cout": int(weights[1].shape[3]),
                             "dw": push(weights[0][:, :, :, 0]), "w": push(weights[1][0, 0])})
            node["b"] = push(weights[-1]) if cfg["use_bias"] else None
        elif kind == "BatchNormalization":
            gamma, beta, mean, var = (w.astype("float64") for w in layer.get_weights())
            scale = gamma / np.sqrt(var + layer.epsilon)
            shift = beta - mean * scale
            source = by_name.get(inputs[0])
            if source and source["op"] == "conv" and source["act"] == "linear" and consumers.get(source["name"], 0) == 1:
                kernel = arrays[source["w"]].reshape(source["k"], source["k"], source["cin"], source["cout"])
                arrays[source["w"]] = (kernel * scale).ravel()
                bias = arrays[source["b"]] if source["b"] is not None else np.zeros(source["cout"])
                if source["b"] is None:
                    source["b"] = push(bias)
                arrays[source["b"]] = bias * scale + shift
                alias[name] = source["name"]
                continue
            node = {"name": name, "op": "affine", "scale": push(scale), "shift": push(shift)}
        elif kind == "GroupNormalization":
            weights = layer.get_weights()
            node = {"name": name, "op": "groupnorm", "groups": int(cfg["groups"]), "epsilon": float(cfg["epsilon"]),
                    "gamma": push(weights[0]), "beta": push(weights[1])}
        elif kind == "Add":
            node = {"name": name, "op": "add"}
        elif kind == "MaxPooling2D":
            node = {"name": name, "op": "maxpool", "k": int(cfg["pool_size"][0]), "stride": int(cfg["strides"][0])}
        elif kind == "GlobalAveragePooling2D":
            node = {"name": name, "op": "gap"}
        elif kind == "Dense":
            kernel, bias = layer.get_weights()
            node = {"name": name, "op": "dense", "in": int(kernel.shape[0]), "out": int(kernel.shape[1]),
                    "w": push(kernel), "b": push(bias), "act": cfg["activation"]}
        node["inputs"] = inputs
        nodes.append(node)
        by_name[name] = node
    outputs = config["output_layers"]
    if isinstance(outputs[0], str):          # single output serialized as [name, node, tensor]
        outputs = [outputs]
    if len(outputs) != 1:
        raise ValueError("Exactly one model output is required")
    output = ref(outputs[0][0])
    # Replace array indexes with float offsets into one flat buffer.
    offsets, total = [], 0
    for array in arrays:
        offsets.append(total)
        total += array.size
    for node in nodes:
        for key in ("w", "b", "dw", "scale", "shift", "gamma", "beta"):
            if node.get(key) is not None:
                node[key] = offsets[node[key]]
    return nodes, np.concatenate(arrays).astype("<f4"), output


def _supported(item, layers):
    kind, cfg = item["class_name"], item["config"]
    if kind in IDENTITY_LAYERS or kind in ("InputLayer", "Add", "GlobalAveragePooling2D", "BatchNormalization"):
        return kind != "GlobalAveragePooling2D" or not cfg.get("keepdims")
    if kind == "Sequential":
        return all(l["class_name"] in AUGMENTATION_LAYERS for l in cfg["layers"])
    if kind == "ReLU":
        return not cfg.get("negative_slope") and not cfg.get("threshold")
    if kind == "Activation":
        return cfg["activation"] in ("relu", "linear", "softmax")
    if kind in ("Conv2D", "SeparableConv2D"):
        square = cfg["kernel_size"][0] == cfg["kernel_size"][1] and cfg["strides"][0] == cfg["strides"][1]
        plain = tuple(cfg.get("dilation_rate", (1, 1))) == (1, 1) and cfg.get("groups", 1) in (1, None)
        return (square and plain and cfg["padding"] in ("same", "valid") and cfg["activation"] in ("linear", "relu")
                and cfg.get("depth_multiplier", 1) == 1)
    if kind == "GroupNormalization":
        return cfg.get("axis", -1) == -1 and cfg.get("center", True) and cfg.get("scale", True)
    if kind == "MaxPooling2D":
        return cfg["padding"] == "valid" and cfg["pool_size"][0] == cfg["pool_size"][1] and cfg["strides"][0] == cfg["strides"][1]
    if kind == "Dense":
        return cfg["activation"] in ("relu", "linear", "softmax")
    return False


# ------------------------------------------------------------------ numpy reference
def _pad(n, k, s, padding):
    if padding == "valid":
        return 0, 0, (n - k) // s + 1
    out = -(-n // s)
    total = max((out - 1) * s + k - n, 0)
    return total // 2, total - total // 2, out


def _windows(x, k, s, padding):
    _, h, w, _ = x.shape
    t, b, oh = _pad(h, k, s, padding)
    l, r, ow = _pad(w, k, s, padding)
    padded = np.pad(x, ((0, 0), (t, b), (l, r), (0, 0)))
    return [(ky, kx, padded[:, ky:ky + (oh - 1) * s + 1:s, kx:kx + (ow - 1) * s + 1:s, :])
            for ky in range(k) for kx in range(k)], oh, ow


def forward_graph(nodes, flat, x, output):
    """Float64 numpy reference of the exported graph. x: (N,H,W,3) in [0,1]."""
    flat = np.asarray(flat, dtype="float64")
    get = lambda offset, size: flat[offset:offset + size]
    values = {}
    for node in nodes:
        op = node["op"]
        a = values[node["inputs"][0]] if node.get("inputs") else None
        if op == "input":
            y = np.asarray(x, dtype="float64")
        elif op == "relu":
            y = np.maximum(a, 0) if node["max"] is None else np.clip(a, 0, node["max"])
        elif op == "softmax":
            e = np.exp(a - a.max(axis=-1, keepdims=True)); y = e / e.sum(axis=-1, keepdims=True)
        elif op in ("conv", "sepconv"):
            k, cin, cout = node["k"], node["cin"], node["cout"]
            windows, oh, ow = _windows(a, k, node["stride"], node["padding"])
            if op == "conv":
                kernel = get(node["w"], k * k * cin * cout).reshape(k, k, cin, cout)
                y = sum(win @ kernel[ky, kx] for ky, kx, win in windows)
            else:
                depth = get(node["dw"], k * k * cin).reshape(k, k, cin)
                y = sum(win * depth[ky, kx] for ky, kx, win in windows) @ get(node["w"], cin * cout).reshape(cin, cout)
            if node["b"] is not None:
                y = y + get(node["b"], cout)
            if node["act"] == "relu":
                y = np.maximum(y, 0)
        elif op == "affine":
            c = a.shape[-1]; y = a * get(node["scale"], c) + get(node["shift"], c)
        elif op == "groupnorm":
            n, h, w, c = a.shape; g = node["groups"]
            r = a.reshape(n, h, w, g, c // g)
            mean = r.mean(axis=(1, 2, 4), keepdims=True); var = r.var(axis=(1, 2, 4), keepdims=True)
            y = ((r - mean) / np.sqrt(var + node["epsilon"])).reshape(n, h, w, c) * get(node["gamma"], c) + get(node["beta"], c)
        elif op == "add":
            y = sum(values[i] for i in node["inputs"])
        elif op == "maxpool":
            k, s = node["k"], node["stride"]
            windows, _, _ = _windows(a, k, s, "valid")
            y = np.max([win for _, _, win in windows], axis=0)
        elif op == "gap":
            y = a.mean(axis=(1, 2))
        elif op == "dense":
            y = a @ get(node["w"], node["in"] * node["out"]).reshape(node["in"], node["out"]) + get(node["b"], node["out"])
            if node["act"] == "relu":
                y = np.maximum(y, 0)
            elif node["act"] == "softmax":
                e = np.exp(y - y.max(axis=-1, keepdims=True)); y = e / e.sum(axis=-1, keepdims=True)
        values[node["name"]] = y
    return values[output]


def verify_against_keras(model, nodes, flat, output, images):
    rng = np.random.default_rng(0)
    batch = np.concatenate([np.stack(images), rng.random((4, 128, 128, 3))]).astype("float32") if images else \
        rng.random((8, 128, 128, 3)).astype("float32")
    expected = model(batch, training=False).numpy()
    error = float(np.abs(expected - forward_graph(nodes, flat, batch, output)).max())
    if error > MAX_PARITY_ERROR:
        raise AssertionError(f"Exported network deviates from Keras by {error:.2e} (> {MAX_PARITY_ERROR})")
    return error


# ------------------------------------------------------------------ model bundle
def export_model(run, out_dir, model_id, info, check_images, release_dir=None):
    import tensorflow as tf
    run = resolve_path(run)
    meta = json.loads((run / "run.json").read_text())
    if meta["classes"] != CLASS_NAMES:
        raise ValueError("Unsupported class order")
    model = tf.keras.models.load_model(run / "model.keras", compile=False)
    size = meta["config"]["image_size"]
    if tuple(model.input_shape[1:]) != (size, size, 3) or model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError("Model does not match the saved configuration")
    nodes, flat, output = export_graph(model)
    error = verify_against_keras(model, nodes, flat, output, check_images)
    target = out_dir / model_id
    target.mkdir(parents=True, exist_ok=True)
    weights = flat.tobytes()
    (target / "weights.bin").write_bytes(weights)
    threshold = (meta.get("review_threshold") or {}).get("threshold", meta["config"]["confidence_threshold"])
    spec = {"format": FORMAT, "id": model_id, "classes": CLASS_NAMES, "input_size": size, "nodes": nodes, "output": output,
            "weights_file": "weights.bin", "weights_floats": int(flat.size),
            "weights_sha256": hashlib.sha256(weights).hexdigest(), "temperature": meta["temperature"],
            "confidence_threshold": float(threshold), "review_threshold": meta.get("review_threshold"),
            "mode": meta["mode"], "production_ready": bool(meta["production_ready"]), "calibration": meta.get("calibration"),
            "architecture": meta["config"]["architecture"], "parameters": int(meta["model_parameters"]),
            "pad_color": 127, "keras_parity_max_abs_error": error, "source_run": _display_path(run)}
    atomic_json(target / "model.json", spec)
    entry = {"id": model_id, "path": f"data/models/{model_id}/", "architecture": spec["architecture"],
             "parameters": spec["parameters"], "weights_bytes": len(weights), "weights_sha256": spec["weights_sha256"],
             "keras_parity_max_abs_error": error,
             "review_threshold": spec["review_threshold"], "confidence_threshold": spec["confidence_threshold"], **info}
    if release_dir is not None:
        released = export_release(run, release_dir, model_id)
        entry["keras_release_file"] = _display_path(released / "model.keras")
    return entry


def write_fixture(run, path, examples_dir=None):
    """Golden cases produced by the real Python predictor (tests/js_parity.mjs). >= 10 per model."""
    from src.predict import Predictor
    predictor = Predictor(run)
    sources = sorted(Path(p) for p in pd.read_csv(ROOT_DIR / "outputs/experiments/own_v1/baseline/dataset/manifest.csv").filepath)
    sizes = [(150, 200), (200, 150), (128, 128), (100, 100), (64, 48), (40, 160), (200, 100), (129, 127), (256, 256), (90, 150)]
    cases = []

    def add(name, array):
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="PNG")
        buffer.seek(0); expected = predictor.predict(buffer)
        buffer.seek(0); x = load_image(buffer, 128)
        views = np.stack([x, x[:, ::-1, :], np.clip(x * .9, 0, 1), np.clip(x * 1.1, 0, 1)])
        raw = predictor.model(views, training=False).numpy()
        cases.append({"name": name, "width": int(array.shape[1]), "height": int(array.shape[0]),
                      "rgb": base64.b64encode(array.tobytes()).decode(),
                      "expected_input": base64.b64encode(np.rint(x * 255).astype(np.uint8).tobytes()).decode(),
                      "expected_views": raw.astype(float).tolist(),
                      "expected": {k: expected[k] for k in ("label", "confidence", "probabilities", "quality", "needs_review",
                                                            "review_reasons", "probability_margin")} |
                                  {"agreement": expected["stability"]["agreement"]}})

    for number, (w, h) in enumerate(sizes):
        with Image.open(resolve_path(sources[(number * 7) % len(sources)])) as image:
            small = image.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
        add(f"photo_{w}x{h}", np.asarray(small, dtype=np.uint8))
    add("dark_flat", np.full((64, 64, 3), 5, dtype=np.uint8))
    add("bright_flat", np.full((60, 80, 3), 253, dtype=np.uint8))
    blue = np.zeros((120, 160, 3), dtype=np.uint8)
    blue[...] = (40, 90, 200)
    blue[30:90, 50:110] = (30, 160, 60)
    add("not_tomato_blue_green", blue)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"run": _display_path(resolve_path(run)), "cases": cases}))
    return len(cases)


# ------------------------------------------------------------------ release copy + model_info
RELEASE_META_KEYS = ("config", "classes", "temperature", "calibration", "review_threshold", "mode", "production_ready",
                     "model_parameters", "runtime", "dataset_fingerprint")


def _display_path(path):
    try:
        return str(Path(path).relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def export_release(run, release_dir, model_id):
    """Copy model.keras plus a trimmed run.json (no local paths, no file names) for src.predict."""
    run = resolve_path(run)
    meta = json.loads((run / "run.json").read_text())
    target = release_dir / model_id
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run / "model.keras", target / "model.keras")
    trimmed = {key: meta[key] for key in RELEASE_META_KEYS if key in meta}
    trimmed["source_run"] = _display_path(run)
    atomic_json(target / "run.json", trimmed)
    return target


def write_release_manifest(release_dir, default_id, model_ids):
    atomic_json(release_dir / "release.json", {"default": default_id, "models": sorted(model_ids),
                                               "note": "Same Keras network that the web demo exports (see site/model_info.json)."})


def build_model_info(models, default_id, previous=None):
    """Facts about the exported models, derived from the export itself. `js_parity` is filled by tests/js_parity.mjs."""
    import keras
    import tensorflow as tf
    previous = previous or {}
    entries = []
    for model in models:
        old = next((m for m in previous.get("models", []) if m["id"] == model["id"]), {})
        entries.append({"id": model["id"], "role": model.get("role"), "architecture": model["architecture"],
                        "parameters": model["parameters"], "weights_bytes": model["weights_bytes"],
                        "weights_mb": round(model["weights_bytes"] / 1e6, 2),
                        "weights_sha256": model["weights_sha256"],
                        "keras_parity_max_abs_error": model["keras_parity_max_abs_error"],
                        "review_threshold": model["confidence_threshold"],
                        "keras_release_file": model.get("keras_release_file"),
                        "js_parity": old.get("js_parity") if old.get("weights_sha256") == model["weights_sha256"] else None})
    return {"format": "tomato-vision-model-info/1", "default": default_id,
            "exported_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
            "export_environment": {"python": platform.python_version(), "tensorflow": tf.__version__, "keras": keras.__version__},
            "parity_gate": {"max_abs_error": MAX_PARITY_ERROR,
                            "meaning": "Exported float64 graph vs Keras probabilities, on test photos plus random images"},
            "models": entries}


def write_model_info(site, models, default_id):
    path = site / "model_info.json"
    previous = json.loads(path.read_text()) if path.exists() else None
    info = build_model_info(models, default_id, previous)
    atomic_json(path, info)
    return info


# ------------------------------------------------------------------ examples + reports
def write_examples(rows, out_dir):
    """Thumbnails without any metadata (no EXIF/GPS: images are rebuilt from pixels)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    written = []
    for number, row in enumerate(rows, 1):
        with Image.open(resolve_path(row["filepath"])) as image:
            pixels = ImageOps.exif_transpose(image).convert("RGB")
            pixels.thumbnail((480, 480), Image.Resampling.LANCZOS)
            clean = Image.new("RGB", pixels.size)
            clean.putdata(list(pixels.getdata()))
            name = f"contoh_{number}_{row['label']}.jpg"
            clean.save(out_dir / name, format="JPEG", quality=86, optimize=True)
        written.append({"file": f"examples/{name}", "label": row["label"], "split": row["split"],
                        "source": Path(row["filepath"]).name})
    return written


def site_config(site, examples_enabled):
    atomic_json(site / "data" / "site-config.json", {"examples": bool(examples_enabled)})


STUDY_DIR = ROOT_DIR / "outputs" / "experiments" / "own_v2"
KEEP_IMAGES = {"logo.png", "favicon.png", "apple-touch-icon.png"}


def remove_legacy_bundle(site):
    """Drop files of the old single-model format (tomato-vision-cnn/1) and stale figures, so they are not deployed."""
    legacy = site / "data" / "model.json"
    if legacy.exists() and json.loads(legacy.read_text()).get("format") != FORMAT:
        legacy.unlink()
        (site / "data" / "weights.bin").unlink(missing_ok=True)
    for image in (site / "img").glob("*.png"):
        if image.name not in KEEP_IMAGES:
            image.unlink()


def build_v2(site, examples_enabled=True, release_dir=RELEASE_DIR, study=STUDY_DIR, fixture_dir=FIXTURE_DIR):
    study = Path(study)
    selection = json.loads((study / "selection.json").read_text())
    summary = {r["name"]: r for r in json.loads((study / "cv_summary.json").read_text())["configs"] if r["complete"]}
    test = json.loads((study / "test_evaluation.json").read_text())
    protocol = json.loads((study / "protocol.json").read_text())
    frame = pd.read_csv(study / "frame.csv")
    test_rows = frame[frame.role == "test"].sort_values("file")
    examples = write_examples([{"filepath": r.filepath, "label": r.label, "split": "test (terkunci)"}
                               for r in test_rows.itertuples()], site / "examples") if examples_enabled else []
    if not examples_enabled:
        shutil.rmtree(site / "examples", ignore_errors=True)
    check = [load_image(resolve_path(p), 128) for p in test_rows.filepath]
    remove_legacy_bundle(site)
    out = site / "data" / "models"
    shutil.rmtree(out, ignore_errors=True)
    models, fixtures = [], []
    for model_id, run, role in [(selection["selected"], study / "final" / "seed42", "selected")] + \
            [(n, study / "final_extra" / n / "seed42", "comparison") for n in selection.get("also_report", [])]:
        cv = summary[model_id]
        tested = test["models"][model_id]
        info = {"role": role, "label": model_id, "cv": {k: cv[k] for k in ("accuracy", "macro_f1", "loss", "ece",
                "accuracy_wilson95_mean_seed", "accuracy_session_bootstrap95", "oof_images_per_seed")},
                "test": {k: tested[k] for k in ("accuracy", "macro_f1", "accuracy_wilson95_mean_seed", "n_images")}}
        models.append(export_model(run, out, model_id, info, check, release_dir if role == "selected" else None))
        fixtures.append((run, Path(fixture_dir) / f"web_parity_{model_id}.json"))
    atomic_json(site / "data" / "models.json", {"default": selection["selected"], "models": models})
    if release_dir is not None:
        write_release_manifest(release_dir, selection["selected"], [m["id"] for m in models if m.get("keras_release_file")])
    write_model_info(site, models, selection["selected"])
    rows = [summary[n] for n in summary]
    report = {"title": "Klasifikasi Tingkat Kesegaran Tomat Menggunakan Convolutional Neural Network (CNN) Berbasis TensorFlow",
              "study": "own_v2", "protocol": {k: protocol[k] for k in ("grouping", "locked_test_sessions", "folds", "seeds",
                                                                     "selection_rule", "known_bias", "amendments")},
              "selection": selection, "cv": rows, "test": test,
              "dataset": {"images": int(len(frame)), "sessions": int(frame.session_id.nunique()),
                          "class_counts": frame.label.value_counts().to_dict(),
                          "roles": {r: frame[frame.role == r].label.value_counts().to_dict() for r in ("dev", "test")},
                          "unlabeled": 22},
              "examples": examples}
    legacy = ROOT_DIR / "outputs" / "experiments" / "own_v1" / "baseline"
    if (legacy / "evaluation.json").exists():
        v1, v1_test = json.loads((legacy / "run.json").read_text()), json.loads((legacy / "evaluation.json").read_text())
        report["legacy_own_v1"] = {
            "validation_accuracy": v1["validation"]["accuracy"], "validation_images": v1["validation"]["n_images"],
            "test_accuracy": v1_test["accuracy"], "test_images": v1_test["n_images"],
            "note": "own_v1 memakai satu split per 'set'; sesi meja kayu (30 foto) tersebar di train/validasi/test, "
                    "jadi angka 100% own_v1 tidak sebanding dan kemungkinan terlalu optimistis. Sesi test own_v2 "
                    "(B06, B07) termasuk data training own_v1, sehingga model own_v1 tidak dinilai pada test baru."}
    atomic_json(site / "data" / "report.json", report)
    for name in ("logo.png", "favicon.png", "apple-touch-icon.png"):
        shutil.copyfile(ROOT_DIR / "web" / name, site / "img" / name)
    for old in (site / "img").glob("learning_curve_*.png"):
        old.unlink()
    for model in models:
        source = study / ("final" if model["role"] == "selected" else f"final_extra/{model['id']}") / "seed42" / "learning_curve.png"
        if source.exists():
            shutil.copyfile(source, site / "img" / f"learning_curve_{model['id']}.png")
    for figure in (study / "figures").glob("*.png"):
        shutil.copyfile(figure, site / "img" / figure.name)
    site_config(site, examples_enabled)
    return models, fixtures


SINGLE_SPLIT_STUDY_DIR = ROOT_DIR / "outputs" / "experiments" / "own_v4_ref22"
REPORT_TITLE = "Klasifikasi Tingkat Kesegaran Tomat Menggunakan Convolutional Neural Network (CNN) Berbasis TensorFlow"
METRIC_KEYS = ("accuracy", "macro_f1", "loss", "expected_calibration_error", "report", "confusion_matrix", "n_images")


def _legacy_own_v1():
    legacy = ROOT_DIR / "outputs" / "experiments" / "own_v1" / "baseline"
    if not (legacy / "evaluation.json").exists():
        return None
    v1, v1_test = json.loads((legacy / "run.json").read_text()), json.loads((legacy / "evaluation.json").read_text())
    return {"validation_accuracy": v1["validation"]["accuracy"], "validation_images": v1["validation"]["n_images"],
            "test_accuracy": v1_test["accuracy"], "test_images": v1_test["n_images"],
            "note": "own_v1 memakai satu split per 'set'; sesi meja kayu (30 foto) tersebar di train/validasi/test, "
                    "jadi angka 100% own_v1 tidak sebanding dan kemungkinan terlalu optimistis."}


def build_single_split(site, examples_enabled=True, release_dir=RELEASE_DIR, study=SINGLE_SPLIT_STUDY_DIR):
    """Single train/validation/test split study (own_v3, own_v4_ref22): selection.json lists the candidates.

    The locked test split is never evaluated here; example photos and the parity check use validation photos only."""
    study = Path(study)
    selection = json.loads((study / "selection.json").read_text())
    selected = Path(selection["selected"]).name
    candidates = sorted(selection["candidates"], key=lambda c: c["name"] != selected)
    manifest = pd.read_csv(resolve_path(candidates[0]["run"]) / "dataset" / "manifest.csv")
    val_rows = manifest[manifest.split == "val"].sort_values(["label_idx", "filepath"])
    examples = write_examples([{"filepath": r.filepath, "label": r.label, "split": f"validation ({r.session_id})"}
                               for r in val_rows.itertuples()], site / "examples") if examples_enabled else []
    if not examples_enabled:
        shutil.rmtree(site / "examples", ignore_errors=True)
    check = [load_image(resolve_path(p), 128) for p in val_rows.filepath]
    remove_legacy_bundle(site)
    out = site / "data" / "models"
    shutil.rmtree(out, ignore_errors=True)
    sessions = dict(zip(manifest.filepath, manifest.session_id))
    models, fixtures, rows = [], [], []
    for candidate in candidates:
        run = resolve_path(candidate["run"])
        meta = json.loads((run / "run.json").read_text())
        role = "selected" if candidate["name"] == selected else "comparison"
        validation = {k: meta["validation"][k] for k in METRIC_KEYS}
        info = {"role": role, "label": candidate["name"],
                "validation": {k: validation[k] for k in ("accuracy", "macro_f1", "loss", "n_images")}}
        models.append(export_model(run, out, candidate["name"], info, check, release_dir if role == "selected" else None))
        fixtures.append((run, Path(FIXTURE_DIR) / f"web_parity_{candidate['name']}.json"))
        predictions = pd.read_csv(run / "validation_predictions.csv")
        config = meta["config"]
        rows.append({"name": candidate["name"], "role": role, "architecture": config["architecture"],
                     "parameters": int(meta["model_parameters"]), "augment": bool(config.get("augment")),
                     "dropout": config.get("dropout"), "learning_rate": config.get("learning_rate"),
                     "lr_schedule": config.get("lr_schedule"), "epochs_max": config.get("epochs"),
                     "best_epoch": meta["best_epoch"], "epochs_run": meta["epochs_run"],
                     "elapsed_seconds": round(meta["elapsed_seconds"]),
                     "train": {k: meta["train_clean"][k] for k in ("accuracy", "macro_f1", "loss", "n_images")},
                     "validation": validation,
                     "predictions": [{"source": Path(p.filepath).name, "session": sessions.get(p.filepath, ""), "label": p.label,
                                      "probabilities": {c: float(getattr(p, "prob_" + c)) for c in CLASS_NAMES}}
                                     for p in predictions.itertuples()]})
        source = run / "learning_curve.png"
        if source.exists():
            shutil.copyfile(source, site / "img" / f"learning_curve_{candidate['name']}.png")
    atomic_json(site / "data" / "models.json", {"default": selected, "models": models})
    if release_dir is not None:
        write_release_manifest(release_dir, selected, [m["id"] for m in models if m.get("keras_release_file")])
    write_model_info(site, models, selected)
    splits = {s: {c: int(n) for c, n in manifest[manifest.split == s].label.value_counts().items()} for s in ("train", "val", "test")}
    report = {"title": REPORT_TITLE, "study": study.name, "study_type": "single_split",
              "selection": {"selected": selected, "rule": selection["selection_rule"],
                            "test_evaluated": bool(selection.get("test_evaluated")),
                            "production_ready": bool(selection.get("production_ready")), "limitation": selection.get("limitation")},
              "candidates": rows,
              "dataset": {"images": int(len(manifest)), "sessions": int(manifest.session_id.nunique()),
                          "class_counts": {c: int(n) for c, n in manifest.label.value_counts().items()},
                          "splits": splits,
                          "split_sessions": {s: sorted(manifest[manifest.split == s].session_id.unique().tolist()) for s in ("train", "val", "test")},
                          "sources": {s: int(n) for s, n in manifest.source.value_counts().items()},
                          "fingerprint": json.loads((resolve_path(candidates[0]["run"]) / "dataset" / "dataset.json").read_text()).get("fingerprint")},
              "examples": examples}
    legacy = _legacy_own_v1()
    if legacy:
        report["legacy_own_v1"] = legacy
    atomic_json(site / "data" / "report.json", report)
    for name in ("logo.png", "favicon.png", "apple-touch-icon.png"):
        shutil.copyfile(ROOT_DIR / "web" / name, site / "img" / name)
    site_config(site, examples_enabled)
    return models, fixtures


def run_export(site=SITE_DIR, examples=True, fixtures=True, release_dir=RELEASE_DIR, study=STUDY_DIR, fixture_dir=FIXTURE_DIR):
    """Full export: web bundle, report, model_info.json, Keras release copy and parity fixtures."""
    site, fixture_dir = Path(site), Path(fixture_dir)
    (site / "img").mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if (Path(study) / "cv_summary.json").exists():
        models, cases = build_v2(site, examples, release_dir, study, fixture_dir)
    else:
        for old in (site / "img").glob("learning_curve_*.png"):
            old.unlink()
        models, cases = build_single_split(site, examples, release_dir, study)
        cases = [(run, fixture_dir / path.name) for run, path in cases]
    for model in models:
        print(f"Exported {model['id']}: {model['weights_bytes']/1e6:.2f} MB, Keras parity error {model['keras_parity_max_abs_error']:.2e}")
    if fixtures:
        for old in fixture_dir.glob("web_parity*.json"):
            old.unlink()
        for run, path in cases:
            print(f"Wrote {write_fixture(run, path)} parity cases to {_display_path(path)}")
    print("Wrote", _display_path(site / "model_info.json"))
    print("Next: node tests/js_parity.mjs --write-info   # records the measured JavaScript parity in model_info.json")
    print(f"Done in {time.monotonic() - started:.0f} s")
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", choices=["own_v2", "own_v4_ref22"], default="own_v4_ref22",
                        help="own_v2: session CV study (needs cv_summary.json); own_v4_ref22: single split, validation only")
    parser.add_argument("--site", default=str(SITE_DIR))
    parser.add_argument("--no-examples", action="store_true", help="Publish without example photos")
    parser.add_argument("--no-fixture", action="store_true")
    parser.add_argument("--release-dir", default=str(RELEASE_DIR), help="Where the selected Keras model is copied for src.predict")
    parser.add_argument("--no-release", action="store_true", help="Do not copy the Keras model")
    args = parser.parse_args()
    study = ROOT_DIR / "outputs" / "experiments" / args.study
    run_export(args.site, not args.no_examples, not args.no_fixture, None if args.no_release else Path(args.release_dir), study)


if __name__ == "__main__":
    main()
