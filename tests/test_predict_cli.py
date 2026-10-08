"""src.predict: files, many files, folders, clear errors for non-images. Uses a tiny random model (no training)."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from src import predict
from src.config import CLASS_NAMES, ExperimentConfig
from src.model import build_model


def make_run(directory, threshold=0.7):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(7)
    Path(directory).mkdir(parents=True, exist_ok=True)
    config = ExperimentConfig(image_size=64, confidence_threshold=threshold)
    build_model(config=config).save(Path(directory) / "model.keras")
    (Path(directory) / "run.json").write_text(json.dumps({
        "config": config.to_dict(), "classes": CLASS_NAMES, "temperature": 1.0, "mode": "own", "production_ready": False}))
    return Path(directory)


def photo(path, size=(80, 60), color=(200, 40, 30)):
    Image.new("RGB", size, color).save(path)
    return path


def run_cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            code = predict.main(list(argv))
        except SystemExit as stop:      # argparse errors
            code = stop.code
    return code, out.getvalue(), err.getvalue()


class ExpandInputsTests(unittest.TestCase):
    def test_folder_file_missing_and_empty_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            photo(root / "a.jpg"); photo(root / "b.PNG"); (root / "notes.txt").write_text("x"); (root / ".hidden").write_text("x")
            (root / "empty").mkdir()
            files, problems, skipped = predict.expand_inputs([root, root / "a.jpg", root / "missing.png", root / "empty"])
            self.assertEqual([f.name for f in files], ["a.jpg", "b.PNG", "a.jpg"])
            self.assertEqual(skipped, 1)       # notes.txt only; dotfiles ignored
            self.assertEqual(len(problems), 2)
            self.assertIn("tidak ditemukan", problems[0]["error"])
            self.assertIn("tidak berisi foto", problems[1]["error"])


class PredictCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.run_dir = make_run(root / "run")
        cls.images = root / "images"
        cls.images.mkdir()
        photo(cls.images / "one.jpg"); photo(cls.images / "two.png", (50, 90), (30, 160, 60))
        (cls.images / "teks.txt").write_text("bukan gambar")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_single_file_json(self):
        code, out, _ = run_cli(str(self.images / "one.jpg"), "--run", str(self.run_dir), "--json")
        self.assertEqual(code, 0)
        row = json.loads(out)[0]
        self.assertAlmostEqual(sum(row["probabilities"].values()), 1.0, places=5)
        self.assertEqual(list(row["probabilities"]), CLASS_NAMES)
        self.assertIn("needs_review", row)
        self.assertIn("review_reasons", row)

    def test_folder_and_legacy_image_flag(self):
        code, out, _ = run_cli(str(self.images), "--run", str(self.run_dir), "--json")
        self.assertEqual(code, 0)
        self.assertEqual([Path(r["image"]).name for r in json.loads(out)], ["one.jpg", "two.png"])
        code, out, _ = run_cli("--image", str(self.images / "one.jpg"), str(self.images / "two.png"), "--run", str(self.run_dir), "--json")
        self.assertEqual((code, len(json.loads(out))), (0, 2))

    def test_text_output_shows_probabilities_and_review_status(self):
        code, out, _ = run_cli(str(self.images / "one.jpg"), "--run", str(self.run_dir))
        self.assertEqual(code, 0)
        for name in CLASS_NAMES:
            self.assertIn(name, out)
        self.assertRegex(out, r"PERLU TINJAUAN|kandidat")
        self.assertIn("ambang perlu tinjauan: 70%", out)

    def test_non_image_file_gives_clear_error_and_nonzero_exit(self):
        code, out, _ = run_cli(str(self.images / "teks.txt"), str(self.images / "one.jpg"), "--run", str(self.run_dir), "--json")
        self.assertEqual(code, 1)
        rows = json.loads(out)
        self.assertIn("Bukan file gambar", rows[0]["error"])
        self.assertNotIn("error", rows[1])          # the valid photo is still predicted

    def test_missing_input_and_missing_model(self):
        code, _, err = run_cli(str(self.images / "nope.jpg"), "--run", str(self.run_dir))
        self.assertEqual(code, 2)
        self.assertIn("tidak ditemukan", err)
        code, _, err = run_cli(str(self.images / "one.jpg"), "--run", str(self.images / "no_run"))
        self.assertEqual(code, 2)
        self.assertIn("Model tidak dapat dimuat", err)
        self.assertEqual(run_cli()[0], 2)         # argparse: needs at least one input

    def test_review_threshold_is_the_saved_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = make_run(Path(tmp), threshold=0.999)
            meta = json.loads((run / "run.json").read_text())
            meta["review_threshold"] = {"threshold": 0.55}      # calibrated value wins, as in the web export
            (run / "run.json").write_text(json.dumps(meta))
            self.assertAlmostEqual(predict.Predictor(run).review_threshold, 0.55)

    def test_default_run_prefers_release_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            release = Path(tmp)
            (release / "x").mkdir()
            (release / "x" / "model.keras").write_bytes(b"x"); (release / "x" / "run.json").write_text("{}")
            (release / "release.json").write_text(json.dumps({"default": "x"}))
            with patch.object(predict, "RELEASE_DIR", release):
                self.assertEqual(predict.default_run(), release / "x")
            with patch.object(predict, "RELEASE_DIR", release / "none"), patch.object(predict, "selected_run", side_effect=FileNotFoundError):
                with self.assertRaisesRegex(FileNotFoundError, "Tidak ada model"):
                    predict.default_run()


if __name__ == "__main__":
    unittest.main()
