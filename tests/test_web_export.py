"""The static web bundle must reproduce the Python models; no TensorFlow needed for the core checks.

Checks the folder given by TV_SITE (default: site/) and fixtures in TV_FIXTURES (default: tests/fixtures/).
"""
import base64
import hashlib
import json
import os
import unittest
from pathlib import Path
import numpy as np
from src.config import CLASS_NAMES, ROOT_DIR
from src.export_web import SUPPORTED_LAYERS, export_graph, forward_graph

SITE = Path(os.environ.get("TV_SITE", ROOT_DIR / "site"))
FIXTURES = Path(os.environ.get("TV_FIXTURES", ROOT_DIR / "tests" / "fixtures"))


def load(entry):
    folder = SITE / entry["path"]
    spec = json.loads((folder / "model.json").read_text())
    raw = (folder / spec["weights_file"]).read_bytes()
    return spec, raw


@unittest.skipUnless((SITE / "data" / "models.json").exists(), "No exported web bundle")
class WebBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((SITE / "data" / "models.json").read_text())

    def test_models_listed_and_default_present(self):
        ids = [m["id"] for m in self.manifest["models"]]
        self.assertIn(self.manifest["default"], ids)
        self.assertEqual(len(ids), len(set(ids)))

    def test_weights_match_spec_and_gate(self):
        for entry in self.manifest["models"]:
            spec, raw = load(entry)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), spec["weights_sha256"])
            self.assertEqual(len(raw), entry["weights_bytes"])
            weights = np.frombuffer(raw, dtype="<f4")
            self.assertEqual(weights.size, spec["weights_floats"])
            self.assertTrue(np.isfinite(weights).all())
            self.assertEqual(spec["classes"], CLASS_NAMES)
            self.assertFalse(spec["production_ready"])
            self.assertLessEqual(spec["keras_parity_max_abs_error"], 1e-4)
            self.assertLess(len(raw), 5_000_000, "target: model under 5 MB")

    def test_offsets_stay_inside_weights(self):
        for entry in self.manifest["models"]:
            spec, raw = load(entry)
            size = len(raw) // 4
            for node in spec["nodes"]:
                for key in ("w", "b", "dw", "scale", "shift", "gamma", "beta"):
                    if node.get(key) is not None:
                        self.assertLess(node[key], size, f"{entry['id']}:{node['name']}:{key}")

    def test_numpy_graph_reproduces_python_predictor(self):
        for entry in self.manifest["models"]:
            spec, raw = load(entry)
            fixture = json.loads((FIXTURES / f"web_parity_{entry['id']}.json").read_text())
            self.assertGreaterEqual(len(fixture["cases"]), 10)
            flat = np.frombuffer(raw, dtype="<f4").astype("float64")
            for case in fixture["cases"]:
                tensor = np.frombuffer(base64.b64decode(case["expected_input"]), dtype=np.uint8)
                x = (tensor.reshape(128, 128, 3) / 255).astype("float32")
                views = np.stack([x, x[:, ::-1, :], np.clip(x * .9, 0, 1), np.clip(x * 1.1, 0, 1)])
                actual = forward_graph(spec["nodes"], flat, views, spec["output"])
                np.testing.assert_allclose(actual, np.asarray(case["expected_views"]), atol=1e-4,
                                           err_msg=f"{entry['id']}/{case['name']}")

    def test_published_images_carry_no_metadata(self):
        for path in list(SITE.rglob("*.jpg")) + list(SITE.rglob("*.jpeg")):
            data = path.read_bytes()
            self.assertNotIn(b"Exif\x00\x00", data, f"EXIF block in {path}")
            self.assertNotIn(b"GPS", data[:4096], f"GPS tag in {path}")

    def test_site_config_matches_examples(self):
        config_path = SITE / "data" / "site-config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {"examples": False}
        report_path = SITE / "data" / "report.json"
        examples = json.loads(report_path.read_text()).get("examples", []) if report_path.exists() else []
        if config["examples"]:
            self.assertTrue(examples)
            for example in examples:
                self.assertTrue((SITE / example["file"]).exists(), example["file"])
        else:
            self.assertEqual(list(SITE.glob("examples/*")), [])


class LegacyCleanupTests(unittest.TestCase):
    def test_old_single_model_bundle_and_stale_figures_are_removed(self):
        import tempfile
        from src.export_web import FORMAT, remove_legacy_bundle
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp)
            (site / "data").mkdir(); (site / "img").mkdir()
            (site / "data" / "model.json").write_text(json.dumps({"format": "tomato-vision-cnn/1"}))
            (site / "data" / "weights.bin").write_bytes(b"x")
            for name in ("logo.png", "favicon.png", "apple-touch-icon.png", "confusion_matrix.png", "learning_curve_old.png"):
                (site / "img" / name).write_bytes(b"png")
            remove_legacy_bundle(site)
            self.assertFalse((site / "data" / "model.json").exists())
            self.assertFalse((site / "data" / "weights.bin").exists())
            self.assertEqual(sorted(p.name for p in (site / "img").iterdir()), ["apple-touch-icon.png", "favicon.png", "logo.png"])
            (site / "data" / "model.json").write_text(json.dumps({"format": FORMAT}))   # current format is kept
            remove_legacy_bundle(site)
            self.assertTrue((site / "data" / "model.json").exists())


class GraphExportTests(unittest.TestCase):
    """Export gate on freshly built Keras models (both architectures, random weights)."""

    def test_both_architectures_export_within_tolerance(self):
        import tensorflow as tf
        from src.config import ExperimentConfig
        from src.model import build_model
        x = np.random.default_rng(3).random((3, 128, 128, 3)).astype("float32")
        for architecture in ("baseline", "regularized"):
            model = build_model(config=ExperimentConfig(architecture=architecture, augment=architecture == "regularized"))
            for layer in model.layers:            # non-trivial BatchNorm statistics
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    g, b, m, v = layer.get_weights()
                    layer.set_weights([g * 1.3, b + .1, m + .05, v * 2])
            nodes, flat, output = export_graph(model)
            np.testing.assert_allclose(model(x, training=False).numpy(), forward_graph(nodes, flat, x, output), atol=1e-4,
                                       err_msg=architecture)

    def test_unsupported_layer_is_reported(self):
        import tensorflow as tf
        inputs = tf.keras.Input((128, 128, 3))
        x = tf.keras.layers.Conv2D(4, 3, padding="same")(inputs)
        x = tf.keras.layers.LayerNormalization()(x)
        outputs = tf.keras.layers.Dense(3, activation="softmax")(tf.keras.layers.GlobalAveragePooling2D()(x))
        with self.assertRaisesRegex(ValueError, "LayerNormalization"):
            export_graph(tf.keras.Model(inputs, outputs))
        self.assertIn("SeparableConv2D", SUPPORTED_LAYERS)


if __name__ == "__main__":
    unittest.main()
