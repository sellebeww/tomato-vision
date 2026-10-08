"""Regression checks for session leakage and generated training provenance."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from src.config import CLASS_NAMES
from src.dataset_split import assign_group_splits, assign_locked_splits, validate_splits
from src.manifest import apply_session_groups, enrich, load_bundle, save_bundle, sha256
from tests.test_pipeline import example_manifest


class DatasetExpansionTests(unittest.TestCase):
    def test_session_links_distinct_fruit_ids_and_cross_session_views(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            pd.DataFrame({"file": ["a", "b", "c"], "session_id": ["day1", "day1", "day2"]}).to_csv(root/"sessions.csv", index=False)
            frame = pd.DataFrame({"filepath": ["a.jpg", "b.jpg", "c.jpg"], "group_id": ["fruit1", "fruit2", "fruit2"]})
            with patch("src.manifest.DATA_DIR", root):
                result = apply_session_groups(frame)
                self.assertEqual(result.group_id.nunique(), 1)
                self.assertEqual(result.annotated_group_id.tolist(), frame.group_id.tolist())
                with self.assertRaisesRegex(ValueError, "Every approved"):
                    apply_session_groups(pd.DataFrame({"filepath": ["unknown.jpg"], "group_id": ["fruit3"]}))

    def test_session_leakage_rejected_even_with_distinct_fruit_groups(self):
        frame = assign_group_splits(example_manifest())
        frame["session_id"] = "same_session"
        with self.assertRaisesRegex(ValueError, "session_id"):
            validate_splits(frame)

    def test_synthetic_images_only_allowed_in_train(self):
        frame = assign_group_splits(example_manifest())
        synthetic = frame[frame.split == "train"].iloc[:1].assign(source="synthetic_ai", split="train")
        validate_splits(pd.concat([frame, synthetic.assign(sha256="s", pixel_sha256="ps", group_id="synthetic:scene01")]))
        for split in ("val", "test"):
            leaked = frame[frame.split == split].iloc[:1].assign(source="synthetic_ai")
            with self.assertRaisesRegex(ValueError, "AI-generated"):
                validate_splits(pd.concat([frame, leaked.assign(sha256="s", pixel_sha256="ps")]))

    def test_locked_holdout_is_preserved_for_every_seed(self):
        frame = example_manifest()
        locked = frame[frame.group_id == "fruit0"].sha256.tolist()
        for seed in (42, 43, 44):
            result = assign_locked_splits(frame, locked, seed)
            self.assertEqual(set(result[result.split == "test"].sha256), set(locked))
            validate_splits(result)
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            assign_locked_splits(frame, ["missing"])
        with self.assertRaisesRegex(ValueError, "shares"):
            assign_locked_splits(frame, locked[:1])

    def test_bundle_cannot_claim_a_different_locked_test(self):
        frame = assign_group_splits(example_manifest())
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)/"bundle"
            with self.assertRaisesRegex(ValueError, "locked test"):
                save_bundle(frame, {"mode": "own", "locked_test_sha256": ["wrong"]}, folder)
            self.assertFalse(folder.exists())

    def test_export_balanced_original_holdout_and_parent_integrity(self):
        from src.export_augmentation import export
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            rows = []
            for g in range(4):
                for i, label in enumerate(CLASS_NAMES):
                    path = root/f"{g}_{label}.png"
                    Image.fromarray(np.random.default_rng(g*3+i).integers(0, 256, (40, 40, 3), dtype=np.uint8)).save(path)
                    rows.append(dict(filepath=str(path), group_id=f"fruit{g}", label=label, sha256=sha256(path), source="own"))
            frame = assign_group_splits(enrich(pd.DataFrame(rows)))
            dataset = save_bundle(frame, {"mode": "own"}, root/"bundle")
            with patch("src.manifest.ROOT_DIR", root):
                folder = export(dataset, root/"generated", per_class=4, size=32)
                generated = pd.read_csv(folder/"manifest.csv")
                combined = pd.read_csv(folder/"annotations.csv")
                self.assertEqual(generated.label.value_counts().to_dict(), dict.fromkeys(CLASS_NAMES, 4))
                self.assertTrue(generated.source.eq("augmented").all())
                self.assertTrue(generated.split.eq("train").all())
                self.assertFalse(generated.pixel_sha256.duplicated().any())
                self.assertTrue(set(generated.parent_sha256).issubset(set(frame[frame.split == "train"].sha256)))
                for split in ("val", "test"):
                    self.assertEqual(set(combined[combined.split == split].sha256), set(frame[frame.split == split].sha256))
                for row in generated.itertuples():
                    self.assertEqual(sha256(root/row.filepath), row.sha256)
                load_bundle(folder/"parent_dataset")
                metadata = json.loads((folder/"dataset.json").read_text())
                self.assertEqual(metadata["combined_images"], 24)
                with self.assertRaises(FileExistsError):
                    export(dataset, folder, per_class=4, size=32)
                repeated = export(dataset, root/"repeated", per_class=4, size=32)
                self.assertEqual(generated.sha256.tolist(), pd.read_csv(repeated/"manifest.csv").sha256.tolist())
                with patch("PIL.Image.Image.save", side_effect=RuntimeError("interrupted")):
                    with self.assertRaises(RuntimeError):
                        export(dataset, root/"failed", per_class=4, size=32)
                self.assertFalse((root/"failed").exists())
                self.assertEqual(list(root.glob(".augmentation-*")), [])


if __name__ == "__main__":
    unittest.main()
