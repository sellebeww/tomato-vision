import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.consolidate_dataset import COLUMNS
from scripts.validate_dataset import ROOT, check


def make_package(root, rows):
    """rows: (file, label, source, split, group, approved) -> package with one file per row."""
    records = []
    for number, (file, label, source, split, group, approved) in enumerate(rows, start=1):
        path = root / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"image {number}".encode())
        record = dict.fromkeys(COLUMNS, "")
        record.update(id=Path(file).stem.rsplit("_" + label, 1)[0] if label else Path(file).stem, file=file, label=label,
                      source=source, split=split, group_id=group, approved=str(approved),
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        records.append(record)
    with (root / "labels.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(records)


GOOD = [("images/own_001_segar.png", "segar", "own", "train", "own:a", True),
        ("images/own_002_busuk.png", "busuk", "own", "val", "own:b", True),
        ("images/syn_001_segar.png", "segar", "synthetic_ai", "train", "synthetic:scene01", False),
        ("images/ref_001.jpeg", "", "own", "unassigned", "own:r", False)]


class DatasetPackageTests(unittest.TestCase):
    def problems(self, rows):
        with tempfile.TemporaryDirectory() as folder:
            make_package(Path(folder), rows)
            return check(folder)[0]

    def test_consistent_package_passes(self):
        self.assertEqual(self.problems(GOOD), [])

    def test_inconsistent_name_is_reported(self):
        bad = [("images/TOM001_segar.png", "segar", "own", "train", "own:a", True)] + GOOD[1:]
        self.assertTrue(any("pola" in p for p in self.problems(bad)))

    def test_synthetic_in_validation_is_reported(self):
        bad = GOOD[:2] + [("images/syn_001_segar.png", "segar", "synthetic_ai", "val", "synthetic:scene01", False)]
        self.assertTrue(any("val/test" in p for p in self.problems(bad)))

    def test_group_split_across_sets_is_reported(self):
        bad = [GOOD[0], ("images/own_002_busuk.png", "busuk", "own", "val", "own:a", True)]
        self.assertTrue(any("terpecah" in p for p in self.problems(bad)))

    def test_unlisted_and_tampered_files_are_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_package(root, GOOD)
            (root / "images/own_009_segar.png").write_bytes(b"x")
            (root / GOOD[0][0]).write_bytes(b"changed")
            messages = " ".join(check(root)[0])
            self.assertIn("tidak tercatat", messages)
            self.assertIn("hash berbeda", messages)

    def test_subfolder_is_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_package(root, GOOD)
            (root / "images/own").mkdir()
            self.assertTrue(any("subfolder" in p for p in check(root)[0]))

    @unittest.skipUnless((ROOT / "submission/dataset/images").is_dir(), "gambar dataset tidak ada di checkout ini")
    def test_shipped_package_is_consistent(self):
        problems, rows = check(ROOT / "submission/dataset")
        self.assertEqual(problems, [])
        self.assertEqual(len(rows), 229)


if __name__ == "__main__":
    unittest.main()
