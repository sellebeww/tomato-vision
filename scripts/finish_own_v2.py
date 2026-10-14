"""Everything after the own_v2 sweep, in order; each step is skipped when its artifact already exists.

  python -m scripts.finish_own_v2            # normally started by scripts/run_own_v2.sh after the sweep

1. summarize + select (pre-declared rule, locked test untouched)   -> cv_summary.json, selection.json
2. final refit of the selected + declared comparison configs        -> final/, final_extra/
3. locked test, exactly once                                        -> test_evaluation.json
4. web export + Keras release copy + parity fixtures                -> site/, models/release/, tests/fixtures/
5. measured JavaScript parity                                       -> site/model_info.json (js_parity)
6. README blocks + a DRAFT copy of the report (original .docx untouched)
7. all tests (Python, JS parity, predict.py vs web, browser)        -> <study>/release_status.json + .md

Nothing is committed or pushed. Paths are configurable so the whole chain can be dry-run on a scratch copy
(--skip-final then expects final models to exist already, because cv_study workers always write to own_v2).
"""
import argparse
import datetime as dt
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import cv_study  # noqa: E402

# Pre-declared before any CV result was summarized (2026-10-06): every swept stage-1 config plus the derived
# no_restore; ref (own_v1 settings) and regularized are refit and tested as comparisons. cosine was not swept.
CONSIDER = ["ref", "no_restore", "lr0003", "bn09", "aug", "regularized"]
ALSO_REPORT = ["ref", "regularized"]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def log(message):
    print(f"[{dt.datetime.now():%F %T}] {message}", flush=True)


def run(cmd, env=None, timeout=None):
    log("$ " + " ".join(map(str, cmd)))
    result = subprocess.run(list(map(str, cmd)), cwd=ROOT, env={**os.environ, **(env or {})}, capture_output=True,
                            text=True, timeout=timeout)
    tail = (result.stdout + result.stderr)[-3000:]
    print(tail, flush=True)
    return result.returncode, tail


def study_steps(root, skip_final):
    if not (root / "selection.json").exists():
        rows = {r["name"]: r for r in cv_study.summarize(root)}
        incomplete = [n for n in CONSIDER if not rows.get(n, {}).get("complete")]
        if incomplete:
            raise SystemExit(f"Sweep incomplete for {incomplete}; selection not frozen")
        log("selection: " + json.dumps(cv_study.select(CONSIDER, ALSO_REPORT, root)))
    if not skip_final:
        log("final refit: " + json.dumps(cv_study.final(root, workers=2)))
    selection = json.loads((root / "selection.json").read_text())
    for name in [selection["selected"]] + selection.get("also_report", []):
        for seed in cv_study.SEEDS:
            path = cv_study.final_dir(name, selection, root) / f"seed{seed}" / "model.keras"
            if not path.exists():
                raise SystemExit(f"Missing final model {path}; locked test NOT evaluated")
    if not (root / "test_evaluation.json").exists():
        result = cv_study.test_once(root)
        log("locked test evaluated once: " + json.dumps({n: m["accuracy"] for n, m in result["models"].items()}))


def browser_tests(site):
    """Serve the site and run both browser tests in a throw-away headless Chrome."""
    if not Path(CHROME).exists():
        return {"site_smoke": "skipped (Chrome not found)", "site_about_smoke": "skipped (Chrome not found)"}
    profile = Path(site).parent / ".chrome-profile-tests"
    server = subprocess.Popen([sys.executable, "-m", "http.server", "8765", "-d", str(site)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    chrome = subprocess.Popen([CHROME, "--headless=new", "--remote-debugging-port=9235", f"--user-data-dir={profile}", "--no-first-run",
                               "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    results = {}
    try:
        for _ in range(60):
            try:
                socket.create_connection(("127.0.0.1", 9235), 1).close()
                socket.create_connection(("127.0.0.1", 8765), 1).close()
                break
            except OSError:
                time.sleep(1)
        for test, extra in (("site_smoke", ["--site", site]), ("site_about_smoke", [])):
            code, tail = run(["node", f"tests/{test}.mjs", *extra], timeout=1800)
            results[test] = "passed" if code == 0 else "FAILED: " + tail[-800:]
    finally:
        for process in (chrome, server):
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=30)
        shutil.rmtree(profile, ignore_errors=True)
    return results


def main(argv=None):
    from src import export_web
    from scripts import sync_docs
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(cv_study.OUT))
    parser.add_argument("--site", default=str(export_web.SITE_DIR))
    parser.add_argument("--release", default=str(export_web.RELEASE_DIR))
    parser.add_argument("--fixtures", default=str(export_web.FIXTURE_DIR))
    parser.add_argument("--readme", default=str(sync_docs.README))
    parser.add_argument("--docx", default=str(ROOT / "submission/laporan/IS794_Laporan_TomatoVision.docx"))
    parser.add_argument("--docx-out", default=str(ROOT / "submission/laporan/IS794_Laporan_TomatoVision_own_v2_DRAFT.docx"))
    parser.add_argument("--skip-final", action="store_true", help="dry runs only: final models must already exist")
    parser.add_argument("--skip-browser", action="store_true")
    args = parser.parse_args(argv)
    root, site, release, fixtures = map(Path, (args.root, args.site, args.release, args.fixtures))
    status = {"started": dt.datetime.now().isoformat(timespec="seconds"), "root": str(root), "steps": {}, "tests": {}}

    study_steps(root, args.skip_final)
    status["steps"]["study"] = "selection frozen, final models present, locked test evaluated once"
    export_web.run_export(site, True, True, release, root, fixtures)
    status["steps"]["export"] = "done"
    code, tail = run(["node", "tests/js_parity.mjs", "--site", site, "--fixtures", fixtures, "--write-info"], timeout=3600)
    if code:
        raise SystemExit("JavaScript parity failed; docs not updated:\n" + tail)
    status["steps"]["js_parity"] = "written to model_info.json"
    info, entry = sync_docs.load(site / "model_info.json")
    report = sync_docs.load_report(site / "data" / "report.json")
    sync_docs.update_readme(info, entry, False, report, args.readme)
    sync_docs.update_docx(args.docx, info, entry, False, report=report, out=args.docx_out)
    status["steps"]["docs"] = f"README blocks updated; report draft {args.docx_out} (original untouched)"

    env = {"TV_SITE": str(site), "TV_FIXTURES": str(fixtures), "TV_RELEASE": str(release), "TV_PREDICT_IMAGES": str(site / "examples")}
    code, tail = run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], env=env, timeout=7200)
    status["tests"]["python_unittest"] = "passed" if code == 0 else "FAILED: " + tail[-1500:]
    code, tail = run(["node", "tests/js_parity.mjs", "--site", site, "--fixtures", fixtures], timeout=3600)
    status["tests"]["js_parity"] = "passed" if code == 0 else "FAILED: " + tail[-800:]
    code, tail = run([sys.executable, "-m", "scripts.sync_docs", "--check", "--info", site / "model_info.json",
                      "--report", site / "data" / "report.json", "--readme", args.readme])
    status["tests"]["readme_in_sync"] = "passed" if code == 0 else "FAILED: " + tail
    if not args.skip_browser:
        status["tests"].update(browser_tests(site))
    status["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    status["all_tests_passed"] = all(v == "passed" or v.startswith("skipped") for v in status["tests"].values())
    (root / "release_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False))
    lines = [f"# own_v2 release status ({status['finished']})", "", *[f"- {k}: {v}" for k, v in status["steps"].items()], "",
             "## Tests", *[f"- {k}: {v.splitlines()[0] if v else v}" for k, v in status["tests"].items()], "",
             "Nothing was committed. Still manual: Analysis/Conclusion text and own_v1 tables/figures in the report draft."]
    (root / "release_status.md").write_text("\n".join(lines) + "\n")
    log("\n".join(lines))
    return 0 if status["all_tests_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
