"""Keep README.md and the report (.docx) in step with site/model_info.json, so no parity number is typed by hand.

  python -m scripts.sync_docs                 # rewrite the generated block in README.md
  python -m scripts.sync_docs --check         # fail if README.md differs from model_info.json (used by tests)
  python -m scripts.sync_docs --docx submission/laporan/IS794_Laporan_TomatoVision.docx [--docx-out draft.docx] [--colab-tested]
                                              # rewrite the Deployment paragraph and the Evaluation paragraphs of the report

README blocks: <!-- model_info:start/end --> (from site/model_info.json) and <!-- results:start/end -->
(from site/data/report.json). model_info.json is produced by `python -m src.export_web` +
`node tests/js_parity.mjs --write-info`; report.json by `python -m src.export_web` after the own_v2 study.
"""
import argparse
import datetime as dt
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
INFO = ROOT / "site" / "model_info.json"
REPORT = ROOT / "site" / "data" / "report.json"
README = ROOT / "README.md"
START, END = "<!-- model_info:start -->", "<!-- model_info:end -->"
RSTART, REND = "<!-- results:start -->", "<!-- results:end -->"
LABELS = {"segar": "segar", "tidak_segar": "tidak segar", "busuk": "busuk"}
# Indonesian wording of frozen protocol sentences; used only when the protocol text matches exactly.
TRANSLATIONS = {
    "Highest mean OOF macro-F1 over seeds; if the runner-up is within one standard deviation, prefer lower mean OOF log loss, "
    "then fewer parameters. Locked test is never read.":
        "macro-F1 out-of-fold rata-rata tertinggi antar seed; bila kandidat lain berada dalam satu simpangan baku, pilih log loss "
        "out-of-fold rata-rata terendah, lalu jumlah parameter paling sedikit. Test terkunci tidak dibaca.",
    "Each fold's validation sessions also drive early stopping, so CV scores are optimistic; the locked test is the unbiased check":
        "sesi validasi tiap fold juga dipakai untuk early stopping, sehingga skor validasi silang cenderung optimistis; "
        "test terkunci adalah pemeriksaan yang tidak bias",
}
NOTES = {"no_restore": "run ref yang sama, tetapi memakai bobot epoch terakhir alih-alih bobot validasi terbaik"}
SCENARIOS = {"darker": "lebih gelap (×0,8)", "brighter": "lebih terang (×1,2)", "blur": "Gaussian blur (radius 1)"}
MONTHS = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"]


def sci(value):
    return f"{value:.1e}"


def date_id(iso):
    d = dt.datetime.fromisoformat(iso)
    return f"{d.day} {MONTHS[d.month - 1]} {d.year}"


def load(path=INFO):
    info = json.loads(Path(path).read_text())
    entry = next(m for m in info["models"] if m["id"] == info["default"])
    if not entry.get("js_parity"):
        raise SystemExit(f"{path}: js_parity is empty. Run: node tests/js_parity.mjs --write-info")
    return info, entry


def facts(info, entry):
    js, env = entry["js_parity"], info["export_environment"]
    return {"id": entry["id"], "arch": entry["architecture"], "params": thousands(entry["parameters"]),
            "mb": f"{entry['weights_bytes'] / 1e6:.2f}".replace(".", ","), "date": date_id(info["exported_at"]),
            "tf": env["tensorflow"], "keras": env["keras"], "export_err": sci(entry["keras_parity_max_abs_error"]),
            "export_gate": sci(info["parity_gate"]["max_abs_error"]), "net": sci(js["max_network_error_vs_keras"]),
            "net_tol": sci(js["tolerance"]["network"]), "e2e": sci(js["max_end_to_end_probability_error"]),
            "e2e_tol": sci(js["tolerance"]["end_to_end_probability"]), "cases": js["cases"]}


def readme_block(info, entry):
    f = facts(info, entry)
    return "\n".join([
        START,
        f"Diambil otomatis dari [`site/model_info.json`](site/model_info.json) (model `{f['id']}`, diekspor {f['date']}):",
        "",
        f"- Model: CNN `{f['arch']}`, {f['params']} parameter, bobot web {f['mb']} MB. Dilatih dengan TensorFlow {f['tf']} / Keras {f['keras']}.",
        f"- Ekspor vs Keras: selisih maksimum probabilitas {f['export_err']} (gerbang ekspor: {f['export_gate']}).",
        f"- JavaScript vs Keras ({f['cases']} kasus uji, `tests/js_parity.mjs`): selisih maksimum keluaran jaringan **{f['net']}** (batas uji {f['net_tol']}); "
        f"termasuk pengubahan ukuran foto, selisih probabilitas maksimum **{f['e2e']}** (batas {f['e2e_tol']}).",
        END])


def docx_segments(info, entry, colab_tested):
    f = facts(info, entry)
    colab = "" if colab_tested else " (belum diuji di Colab)"
    return [
        ("Demo online (GitHub Pages).", "bold"),
        (" Model TensorFlow diekspor untuk inferensi di browser. Situs statis di https://sellebeww.github.io/tomato-vision/ (aktif setelah GitHub Pages diaktifkan pada repositori) "
         f"menjalankan CNN {f['arch']} ({f['params']} parameter, bobot {f['mb']} MB; dilatih dengan TensorFlow {f['tf']}, diekspor {f['date']}) dengan JavaScript murni, karena GitHub Pages tidak dapat "
         "menjalankan Python/TensorFlow. Bobot model Keras diekspor (", "text"),
        ("python -m src.export_web", "code"),
        ("; BatchNorm dilipat ke konvolusi) dan foto tidak diunggah ke server. Paritas dengan TensorFlow (dicatat otomatis di ", "text"),
        ("site/model_info.json", "code"),
        (f"): selisih maksimum keluaran jaringan JavaScript terhadap Keras {f['net']} pada {f['cases']} kasus uji (batas {f['net_tol']}); termasuk pengubahan ukuran foto, "
         f"selisih probabilitas maksimum {f['e2e']} (batas {f['e2e_tol']}), diuji dengan ", "text"),
        ("tests/js_parity.mjs", "code"),
        (". Versi TensorFlow asli dapat dijalankan lewat notebook Colab" + colab + " atau lokal dengan ", "text"),
        ("python -m src.predict", "code"),
        (" (README, bagian “Jalankan versi TensorFlow”); ", "text"),
        ("tests/test_predict_vs_web.py", "code"),
        (" membandingkan keduanya pada foto contoh, dan tes browser ", "text"),
        ("tests/site_smoke.mjs", "code"),
        (" serta ", "text"),
        ("tests/site_about_smoke.mjs", "code"),
        (" memeriksa situsnya.", "text")]


# ------------------------------------------------------------------ evaluation results (report.json)
def thousands(value):
    return f"{value:,}".replace(",", ".")


def pct(value):
    return f"{value * 100:.1f}".replace(".", ",") + "%"


def pm(stat, as_pct=True):
    if as_pct:
        return f"{pct(stat['mean'])} ± {stat['std'] * 100:.1f}".replace(".", ",")
    return f"{stat['mean']:.3f} ± {stat['std']:.3f}".replace(".", ",")


def ci(bounds):
    return f"{pct(bounds[0])}–{pct(bounds[1])}"


def load_report(path=REPORT):
    report = json.loads(Path(path).read_text())
    for key in ("cv", "selection", "test", "protocol", "dataset"):
        if key not in report:
            raise SystemExit(f"{path}: missing '{key}' (export the finished own_v2 study first)")
    return report


def cv_study_rule():
    """The selection rule as frozen in src/cv_study.py (kept here as text so this module stays stdlib-only)."""
    return next(iter(TRANSLATIONS))


def tr(text):
    return TRANSLATIONS.get(text.rstrip(), text)


def evaluation_facts(report):
    selected = report["selection"]["selected"]
    cv = {row["name"]: row for row in report["cv"] if row.get("complete")}
    test = report["test"]["models"]
    return selected, cv, test, cv[selected], test[selected]


def results_block(report):
    selected, cv, test, scv, stest = evaluation_facts(report)
    protocol, dataset = report["protocol"], report["dataset"]
    lines = [RSTART,
             f"Diambil otomatis dari [`site/data/report.json`](site/data/report.json) (studi `{report.get('study', 'own_v2')}`). "
             f"Data: {dataset['images']} foto berlabel dari {dataset['sessions']} sesi. Validasi silang {len(protocol['folds'])} fold berbasis sesi × "
             f"{len(protocol['seeds'])} seed ({scv['oof_images_per_seed']} prediksi out-of-fold per seed); test terkunci {stest['n_images']} foto "
             f"dari sesi {', '.join(protocol['locked_test_sessions'])}, dinilai satu kali setelah model dipilih.",
             "",
             "**Validasi silang (out-of-fold, rata-rata ± simpangan baku antar seed):**",
             "",
             "| Konfigurasi | Parameter | Akurasi | Wilson 95% | Bootstrap sesi 95% | Macro-F1 | Log loss |",
             "|---|---|---|---|---|---|---|"]
    for name, row in cv.items():
        mark = f"**{name}** (terpilih)" if name == selected else name
        lines.append(f"| {mark} | {thousands(row['parameters'])} | {pm(row['accuracy'])} | {ci(row['accuracy_wilson95_mean_seed'])} | "
                     f"{ci(row['accuracy_session_bootstrap95'])} | {pm(row['macro_f1'], False)} | {pm(row['loss'], False)} |")
    lines += ["", f"**Test terkunci ({stest['n_images']} foto, 5 seed model final):**", "",
              "| Model | Peran | Akurasi | Wilson 95% | Macro-F1 |", "|---|---|---|---|---|"]
    for name, row in test.items():
        role = "terpilih" if row["role"] == "selected" else "pembanding"
        lines.append(f"| {name} | {role} | {pm(row['accuracy'])} | {ci(row['accuracy_wilson95_mean_seed'])} | {pm(row['macro_f1'], False)} |")
    robust = "; ".join(f"{SCENARIOS.get(k, k)} {pm(v['accuracy'])}" for k, v in stest["robustness"].items())
    lines += ["", f"Robustness model terpilih pada test terkunci: asli {pm(stest['accuracy'])}; {robust}.",
              f"Aturan seleksi (dibekukan {report['selection']['frozen'].replace('T', ' ')}, sebelum test dibuka): {tr(protocol['selection_rule'])}",
              f"Bias yang diketahui: {tr(protocol['known_bias'])}.",
              *[f"`{name}`: {note}." for name, note in NOTES.items() if name in cv],
              "Wilson 95% dihitung dari akurasi rata-rata seed terhadap jumlah foto; bootstrap mengambil ulang sesi pemotretan. "
              f"Dengan hanya {stest['n_images']} foto test, satu foto salah mengubah akurasi test {pct(1 / stest['n_images'])}.",
              REND]
    return "\n".join(lines)


def evaluation_paragraphs(report):
    """Lead text of a report paragraph -> new segments. Only facts from report.json."""
    selected, cv, test, scv, stest = evaluation_facts(report)
    protocol = report["protocol"]
    others = [n for n in cv if n != selected]
    classes = scv["pooled_over_seeds"]["report"]
    per_class = "; ".join(f"{LABELS[c]} P {classes[c]['precision']:.2f}/R {classes[c]['recall']:.2f}".replace(".", ",") for c in LABELS)
    comparisons = "; ".join(f"{n}: akurasi test {pm(t['accuracy'])}" for n, t in test.items() if n != selected)
    robust = "; ".join(f"{SCENARIOS.get(k, k)} {pm(v['accuracy'])}" for k, v in stest["robustness"].items())
    rt = scv.get("review_threshold")
    return {
        "Protokol.": [("Protokol.", "bold"), (
            f" Sesi pemotretan dipakai sebagai grup. Validasi silang {len(protocol['folds'])} fold berbasis sesi × {len(protocol['seeds'])} seed "
            f"({', '.join(map(str, protocol['seeds']))}) membandingkan {len(cv)} konfigurasi; aturan seleksi dibekukan sebelum test dibuka: "
            f"{tr(protocol['selection_rule'])} Test terkunci ({stest['n_images']} foto dari sesi {', '.join(protocol['locked_test_sessions'])}) "
            "dinilai satu kali untuk model terpilih dan model pembanding yang dideklarasikan sebelumnya.", "text")],
        "Hasil utama.": [("Hasil utama.", "bold"), (
            f" Model terpilih {selected} ({thousands(scv['parameters'])} parameter): akurasi validasi silang {pm(scv['accuracy'])} "
            f"(Wilson 95% {ci(scv['accuracy_wilson95_mean_seed'])}; bootstrap sesi {ci(scv['accuracy_session_bootstrap95'])}), macro-F1 "
            f"{pm(scv['macro_f1'], False)}. Test terkunci: akurasi {pm(stest['accuracy'])} (Wilson 95% {ci(stest['accuracy_wilson95_mean_seed'])}), "
            f"macro-F1 {pm(stest['macro_f1'], False)}." + (f" Pembanding: {comparisons}." if comparisons else "") +
            (f" Konfigurasi lain yang diuji: {', '.join(others)} (tabel lengkap di README dan tab Evaluasi demo)." if others else "") +
            "".join(f" {name}: {note}." for name, note in NOTES.items() if name in others), "text")],
        "Per kelas": [("Per kelas (out-of-fold, model terpilih).", "bold"), (f" {per_class}.", "text")],
        "Kalibrasi.": [("Kalibrasi.", "bold"), (
            f" ECE out-of-fold {pm(scv['ece'], False)}; test {pm(stest['ece'], False) if 'ece' in stest else 'tidak dihitung'}. Softmax mentah dipakai (tanpa temperature scaling)." +
            (f" Ambang \u201cperlu tinjauan\u201d {pct(rt['threshold'])} dikalibrasi dari prediksi out-of-fold (target akurasi {pct(rt['target_accuracy'])}, "
             f"cakupan {pct(rt['coverage'])})." if rt else ""), "text")],
        "Robustness.": [("Robustness.", "bold"), (f" Pada test terkunci: asli {pm(stest['accuracy'])}; {robust}.", "text")],
        "Keterbatasan evaluasi.": [("Keterbatasan evaluasi.", "bold"), (
            f" Test hanya {stest['n_images']} foto dari {len(protocol['locked_test_sessions'])} sesi; satu foto salah mengubah akurasi "
            f"{pct(1 / stest['n_images'])}. Selain itu, {tr(protocol['known_bias'])}. Angka ini indikasi awal, bukan bukti generalisasi.", "text")],
    }


CODE_RPR = '<w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:cs="Consolas"/><w:sz w:val="19"/><w:szCs w:val="19"/></w:rPr>'
PPR = '<w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr><w:spacing w:before="0" w:after="100"/></w:pPr>'


def docx_paragraph(segments, ppr=PPR):
    runs = []
    for text, kind in segments:
        rpr = {"bold": "<w:rPr><w:b/></w:rPr>", "code": CODE_RPR}.get(kind, "")
        runs.append(f'<w:r>{rpr}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>')
    return "<w:p>" + ppr + "".join(runs) + "</w:p>"


PARAGRAPH = re.compile(r"<w:p>(?:(?!</w:p>).)*</w:p>", re.S)


def paragraph_text(xml):
    return "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", xml, flags=re.S))


def replace_paragraphs(xml, replacements):
    """replacements: lead text -> segments. Each lead must start exactly one paragraph; its <w:pPr> is kept."""
    found = {lead: 0 for lead in replacements}

    def swap(match):
        text = paragraph_text(match.group(0))
        for lead, segments in replacements.items():
            if text.startswith(escape(lead)):
                found[lead] += 1
                ppr = re.search(r"<w:pPr>.*?</w:pPr>", match.group(0), re.S)
                return docx_paragraph(segments, ppr.group(0) if ppr else "")
        return match.group(0)

    xml = PARAGRAPH.sub(swap, xml)
    wrong = {lead: n for lead, n in found.items() if n != 1}
    if wrong:
        raise SystemExit(f"Report paragraphs not found exactly once: {wrong}")
    return xml


def update_docx(path, info, entry, colab_tested, backup=True, report=None, out=None):
    """Rewrite the Deployment paragraph (and, with report, the Evaluation paragraphs). out: write a copy instead."""
    path = Path(path)
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    replacements = {"Demo online (GitHub Pages).": docx_segments(info, entry, colab_tested)}
    if report is not None:
        replacements.update(evaluation_paragraphs(report))
    xml = replace_paragraphs(xml, replacements)
    target = Path(out) if out else path
    if backup and not out:
        shutil.copyfile(path, path.with_suffix(".docx.bak"))
    temporary = target.with_suffix(".docx.tmp")
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as zipped:
        for item in source.infolist():
            data = xml.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename)
            zipped.writestr(item, data)
    temporary.replace(target)


def replace_block(text, start, end, block):
    if text.count(start) != 1 or text.count(end) != 1:
        raise SystemExit(f"README.md needs exactly one {start} ... {end}")
    return re.sub(re.escape(start) + ".*?" + re.escape(end), lambda m: block, text, flags=re.S)


def update_readme(info, entry, check, report=None, readme=None):
    readme = Path(readme or README)
    text = readme.read_text()
    new = replace_block(text, START, END, readme_block(info, entry))
    if report is not None:
        new = replace_block(new, RSTART, REND, results_block(report))
    if check:
        if new != text:
            raise SystemExit("README.md differs from site/model_info.json / report.json. Run: python -m scripts.sync_docs")
        print("README.md matches the generated blocks")
    else:
        readme.write_text(new)
        print("README.md updated" + (" (parity + results)" if report is not None else " (parity)"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--info", default=str(INFO))
    parser.add_argument("--report", default=str(REPORT), help="report.json; its results are used when it holds a finished study")
    parser.add_argument("--readme", default=str(README))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--docx")
    parser.add_argument("--docx-out", help="write the updated report to this file instead of in place")
    parser.add_argument("--colab-tested", action="store_true", help="drop the 'belum diuji di Colab' note (only after really running it in Colab)")
    args = parser.parse_args(argv)
    info, entry = load(args.info)
    report = None
    if Path(args.report).exists() and "selection" in json.loads(Path(args.report).read_text()):
        report = load_report(args.report)
    if args.docx:
        update_docx(args.docx, info, entry, args.colab_tested, report=report, out=args.docx_out)
        print("Report updated:", args.docx_out or f"{args.docx} (backup: .docx.bak)")
    else:
        update_readme(info, entry, args.check, report, args.readme)


if __name__ == "__main__":
    sys.exit(main())
