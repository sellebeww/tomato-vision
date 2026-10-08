"""README/report numbers come from model_info.json (scripts/sync_docs.py); nothing is typed by hand."""
import json
import re
import tempfile
import unittest
import xml.dom.minidom
import zipfile
from pathlib import Path
from scripts import sync_docs

INFO = {
    "default": "m1", "exported_at": "2026-10-06T12:41:14+00:00",
    "export_environment": {"python": "3.11.9", "tensorflow": "2.16.1", "keras": "3.15.1"},
    "parity_gate": {"max_abs_error": 1e-4},
    "models": [{"id": "m1", "architecture": "baseline", "parameters": 423619, "weights_bytes": 1686796,
                "keras_parity_max_abs_error": 6.2e-8,
                "js_parity": {"max_network_error_vs_keras": 2.8e-6, "max_end_to_end_probability_error": 2.3e-6, "cases": 13,
                              "tolerance": {"network": 1e-4, "end_to_end_probability": 0.01}}}]}


def stat(mean, std=0.05):
    return {"mean": mean, "std": std}


def synthetic_report():
    """Shape of site/data/report.json (src/export_web.build_v2); values are made up for the test."""
    cv_row = lambda name, acc: {"name": name, "complete": True, "parameters": 423619, "oof_images_per_seed": 54,
                                "accuracy": stat(acc), "macro_f1": stat(acc - .01), "loss": stat(.5), "ece": stat(.1),
                                "accuracy_wilson95_mean_seed": [acc - .1, acc + .05], "accuracy_session_bootstrap95": [acc - .15, acc + .08],
                                "pooled_over_seeds": {"report": {c: {"precision": .8, "recall": .7, "f1-score": .75, "support": 90}
                                                                 for c in ("segar", "tidak_segar", "busuk")}},
                                "review_threshold": {"threshold": .62, "target_accuracy": .9, "coverage": .7}}
    test_row = lambda role: {"role": role, "n_images": 6, "accuracy": stat(.5), "macro_f1": stat(.45), "ece": stat(.2),
                             "accuracy_wilson95_mean_seed": [.19, .81], "robustness": {"blur": {"accuracy": stat(.3)}}}
    return {"study": "own_v2", "cv": [cv_row("bn09", .61), cv_row("no_restore", .55), {"name": "cosine", "complete": False}],
            "selection": {"selected": "bn09", "frozen": "2026-10-08T10:00:00"},
            "test": {"models": {"bn09": test_row("selected"), "ref": test_row("comparison")}},
            "protocol": {"folds": {str(k): [] for k in range(5)}, "seeds": [42, 43, 44, 45, 46], "locked_test_sessions": ["B06_talenan", "B07_baja"],
                         "selection_rule": sync_docs.cv_study_rule(), "known_bias": "Some bias"},
            "dataset": {"images": 60, "sessions": 11}}


class SyncDocsTests(unittest.TestCase):
    def setUp(self):
        self.entry = INFO["models"][0]

    def test_readme_block_contains_the_measured_numbers(self):
        block = sync_docs.readme_block(INFO, self.entry)
        for expected in ("2.8e-06", "2.3e-06", "13 kasus uji", "423.619", "1,69 MB", "6 Oktober 2026", "TensorFlow 2.16.1", "6.2e-08"):
            self.assertIn(expected, block)
        self.assertTrue(block.startswith(sync_docs.START) and block.endswith(sync_docs.END))

    def test_docx_paragraph_is_valid_xml_and_marks_colab_untested(self):
        for tested in (False, True):
            xml_text = sync_docs.docx_paragraph(sync_docs.docx_segments(INFO, self.entry, tested))
            xml.dom.minidom.parseString(f'<w:document xmlns:w="w">{xml_text}</w:document>')
            self.assertEqual("belum diuji di Colab" in xml_text, not tested)
            self.assertIn("Model TensorFlow diekspor untuk inferensi di browser", xml_text)
            self.assertIn("2.8e-06", xml_text)

    def test_docx_update_replaces_only_the_deployment_paragraph(self):
        body = ('<w:body><w:p><w:r><w:t>Awal</w:t></w:r></w:p><w:p><w:pPr/><w:r><w:t>Demo online (GitHub Pages).</w:t></w:r>'
                '<w:r><w:t> lama 6e-8</w:t></w:r></w:p><w:p><w:r><w:t>Akhir</w:t></w:r></w:p></w:body>')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.docx"
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("word/document.xml", f'<w:document xmlns:w="w">{body}</w:document>')
                z.writestr("word/media/image1.png", b"png")
            sync_docs.update_docx(path, INFO, self.entry, False)
            with zipfile.ZipFile(path) as z:
                xml_text = z.read("word/document.xml").decode()
                self.assertEqual(z.read("word/media/image1.png"), b"png")
            self.assertIn("Awal", xml_text); self.assertIn("Akhir", xml_text)
            self.assertNotIn("lama 6e-8", xml_text)
            self.assertEqual(xml_text.count("Demo online (GitHub Pages)."), 1)
            self.assertTrue(path.with_suffix(".docx.bak").exists())

    def test_results_block_and_evaluation_paragraphs_use_report_values(self):
        report = synthetic_report()
        block = sync_docs.results_block(report)
        for expected in ("**bn09** (terpilih)", "61,0% ± 5,0", "51,0%–66,0%", "46,0%–69,0%", "50,0% ± 5,0", "19,0%–81,0%",
                         "6 foto", "B06_talenan, B07_baja", "16,7%", "Gaussian blur (radius 1) 30,0% ± 5,0", "macro-F1 out-of-fold rata-rata"):
            self.assertIn(expected, block)
        self.assertNotIn("cosine", block)                  # incomplete configs are not reported
        self.assertIn("bobot epoch terakhir", block)      # no_restore explained
        paragraphs = sync_docs.evaluation_paragraphs(report)
        self.assertEqual(set(paragraphs), {"Protokol.", "Hasil utama.", "Per kelas", "Kalibrasi.", "Robustness.", "Keterbatasan evaluasi."})
        main = "".join(text for text, _ in paragraphs["Hasil utama."])
        self.assertIn("akurasi validasi silang 61,0% ± 5,0", main)
        self.assertIn("ref: akurasi test 50,0% ± 5,0", main)
        self.assertIn("62,0%", "".join(text for text, _ in paragraphs["Kalibrasi."]))

    def test_translations_match_the_frozen_protocol(self):
        try:
            from src import cv_study
        except (ImportError, TypeError) as error:   # CI job without numpy/pandas, or Python < 3.10
            self.skipTest(f"needs project dependencies: {error}")
        self.assertNotEqual(sync_docs.tr(cv_study.SELECTION_RULE), cv_study.SELECTION_RULE, "update TRANSLATIONS")
        protocol = cv_study.OUT / "protocol.json"
        if protocol.exists():
            frozen = json.loads(protocol.read_text())
            for key in ("selection_rule", "known_bias"):
                self.assertNotEqual(sync_docs.tr(frozen[key]), frozen[key], key)

    def test_readme_markers_present(self):
        text = sync_docs.README.read_text()
        for marker in (sync_docs.START, sync_docs.END, sync_docs.RSTART, sync_docs.REND):
            self.assertEqual(len(re.findall(re.escape(marker), text)), 1, marker)

    @unittest.skipUnless(sync_docs.INFO.exists() and any(m.get("js_parity") for m in json.loads(sync_docs.INFO.read_text())["models"]),
                         "site/model_info.json with js_parity not generated yet")
    def test_readme_matches_model_info(self):
        info, entry = sync_docs.load()
        self.assertIn(sync_docs.readme_block(info, entry), sync_docs.README.read_text())


if __name__ == "__main__":
    unittest.main()
