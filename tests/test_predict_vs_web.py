"""src.predict (TensorFlow/Keras) vs the browser inference core (site/tomato-core.js) on the same photos.

Both sides decode with Pillow, so this isolates what differs: preprocessing code, the network, and the review logic.
Compared per photo: label, needs_review, review reasons, and every class probability (tolerance from model_info.json,
else 0.01, the end-to-end tolerance of tests/js_parity.mjs).

Environment: TV_SITE (default site/), TV_RELEASE (default models/release/), TV_PREDICT_IMAGES (default <site>/examples).
Skipped when node, the web bundle or the released Keras model is missing.
"""
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image, ImageOps
from src.config import ROOT_DIR

SITE = Path(os.environ.get("TV_SITE", ROOT_DIR / "site"))
RELEASE = Path(os.environ.get("TV_RELEASE", ROOT_DIR / "models" / "release"))
IMAGES = Path(os.environ.get("TV_PREDICT_IMAGES", SITE / "examples"))
DEFAULT_TOLERANCE = 0.01


def available():
    if shutil.which("node") is None:
        return "node not installed"
    if not (SITE / "data" / "models.json").exists():
        return "no exported web bundle"
    if not (RELEASE / "release.json").exists():
        return "no released Keras model (python -m src.export_web)"
    if not IMAGES.exists() or not any(IMAGES.glob("*.jpg")):
        return f"no photos in {IMAGES}"
    return None


@unittest.skipIf(available(), available() or "")
class PredictMatchesWebTests(unittest.TestCase):
    def test_same_probabilities_labels_and_review_status(self):
        release = json.loads((RELEASE / "release.json").read_text())
        web = json.loads((SITE / "data" / "models.json").read_text())
        self.assertEqual(release["default"], web["default"], "CLI and web demo must default to the same model")
        model_id = release["default"]
        photos = sorted(IMAGES.glob("*.jpg"))
        tolerance = DEFAULT_TOLERANCE
        info = SITE / "model_info.json"
        if info.exists():
            entry = next(m for m in json.loads(info.read_text())["models"] if m["id"] == model_id)
            tolerance = (entry.get("js_parity") or {}).get("tolerance", {}).get("end_to_end_probability", tolerance)
        cli = subprocess.run([sys.executable, "-m", "src.predict", "--json", "--run", str(RELEASE / model_id), *map(str, photos)],
                             cwd=ROOT_DIR, capture_output=True, text=True, timeout=900)
        self.assertEqual(cli.returncode, 0, cli.stderr[-2000:])
        python_rows = {Path(r["image"]).name: r for r in json.loads(cli.stdout)}
        images = []
        for photo in photos:
            with Image.open(photo) as image:
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                images.append({"name": photo.name, "width": rgb.width, "height": rgb.height,
                               "rgb": base64.b64encode(rgb.tobytes()).decode()})
        with tempfile.TemporaryDirectory() as tmp:
            request = Path(tmp) / "request.json"
            request.write_text(json.dumps({"site": str(SITE), "model": model_id, "images": images}))
            node = subprocess.run(["node", "tests/js_predict.mjs", str(request)], cwd=ROOT_DIR, capture_output=True, text=True, timeout=900)
        self.assertEqual(node.returncode, 0, node.stderr[-2000:])
        worst = 0.0
        for row in json.loads(node.stdout):
            name, expected = row["name"], python_rows[row["name"]]
            self.assertNotIn("error", row, name)
            self.assertNotIn("error", expected, name)
            self.assertEqual(row["label"], expected["label"], name)
            self.assertEqual(row["needs_review"], expected["needs_review"], name)
            self.assertEqual(row["review_reasons"], expected["review_reasons"], name)
            for cls, value in expected["probabilities"].items():
                diff = abs(row["probabilities"][cls] - value)
                worst = max(worst, diff)
                self.assertLess(diff, tolerance, f"{name}/{cls}")
        print(f"\npredict.py vs web: {len(photos)} photos, model {model_id}, max probability difference {worst:.2e} (tolerance {tolerance})")


if __name__ == "__main__":
    unittest.main()
