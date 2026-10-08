"""Cross-browser end-to-end test of the static demo (Playwright: Chromium, Firefox, WebKit) + axe-core audit.

Not part of `unittest discover` (needs Playwright). Setup once:
  python3 -m venv .venv-e2e && .venv-e2e/bin/pip install playwright axe-playwright-python pillow numpy
  .venv-e2e/bin/python -m playwright install chromium firefox webkit
Run:
  .venv-e2e/bin/python tests/site_e2e.py --site site [--json outputs/browser/e2e.json]
The script serves the folder itself on 127.0.0.1, so no other server is needed.
"""
import argparse
import functools
import http.server
import json
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
VIEWPORTS = {"desktop": (1280, 900), "phone": (390, 844), "small_phone": (360, 640)}


def until(page, expression, timeout=60000):
    """Poll with page.evaluate (wait_for_function needs eval, which the site's CSP correctly forbids)."""
    deadline = time.monotonic() + timeout / 1000
    while time.monotonic() < deadline:
        if page.evaluate(expression):
            return
        time.sleep(0.2)
    raise TimeoutError("Timed out waiting for: " + expression)


def serve(folder):
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass
    handler = functools.partial(Quiet, directory=str(folder))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/"


def make_assets(folder):
    folder.mkdir(parents=True, exist_ok=True)
    tomato = Image.open(ROOT / "data/raw/TOM031_segar.png").convert("RGB")
    assets = {"tomato": folder / "tomat.jpg", "dark": folder / "gelap.jpg", "blur": folder / "buram.jpg",
              "blue": folder / "bukan_tomat.png", "big_ok": folder / "besar_20mp.jpg", "too_big": folder / "terlalu_besar_27mp.png",
              "heic": folder / "iphone.heic", "text": folder / "catatan.txt"}
    tomato.save(assets["tomato"], quality=90)
    Image.fromarray((np.asarray(tomato, dtype=np.float32) * 0.03).astype(np.uint8)).save(assets["dark"], quality=90)
    tomato.filter(ImageFilter.GaussianBlur(25)).save(assets["blur"], quality=90)
    blue = np.zeros((300, 400, 3), np.uint8); blue[...] = (40, 90, 200); blue[80:220, 120:280] = (30, 160, 60)
    Image.fromarray(blue).save(assets["blue"])
    tomato.resize((5000, 4000)).save(assets["big_ok"], quality=70)
    Image.new("RGB", (6000, 4500), (200, 60, 40)).save(assets["too_big"])
    assets["heic"].write_bytes(b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic" + bytes(64))
    assets["text"].write_text("bukan gambar")
    many = []
    for i, path in enumerate(sorted((ROOT / "data/raw").glob("*.png"))[:12]):
        target = folder / f"banyak_{i:02d}.jpg"
        Image.open(path).convert("RGB").resize((640, 640)).save(target, quality=85)
        many.append(target)
    return assets, many


def run_browser(p, name, base, assets, many, results, axe_runner):
    browser = getattr(p, name).launch()
    outcome = {"browser": name, "checks": [], "errors": [], "external_requests": [], "failed_requests": []}
    try:
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        page = context.new_page()
        origin = base.rstrip("/")
        page.on("console", lambda m: m.type == "error" and outcome["errors"].append("console: " + m.text))
        page.on("pageerror", lambda e: outcome["errors"].append("pageerror: " + str(e)))
        page.on("request", lambda r: (not r.url.startswith(origin) and not r.url.startswith(("blob:", "data:")))
                and outcome["external_requests"].append(r.url))
        page.on("requestfailed", lambda r: outcome["failed_requests"].append(r.url))
        page.on("response", lambda r: r.status >= 400 and outcome["failed_requests"].append(f"{r.status} {r.url}"))
        started = time.monotonic()
        page.goto(base)
        until(page, "document.querySelector('#status').textContent.includes('dimuat')", timeout=60000)
        outcome["model_load_ms_reported"] = int(page.evaluate("document.documentElement.dataset.modelLoadMs || -1"))
        outcome["page_ready_ms"] = int((time.monotonic() - started) * 1000)
        check = lambda label, ok: outcome["checks"].append({"check": label, "ok": bool(ok)})
        report = page.evaluate("fetch('data/report.json').then(r=>r.ok?r.json():null)")
        config = page.evaluate("fetch('data/site-config.json').then(r=>r.ok?r.json():{examples:false})")
        models = page.evaluate("fetch('data/models.json').then(r=>r.json())")

        def predict(files, expect_cards):
            page.set_input_files("#upload", [str(f) for f in files])
            page.click("#predict")
            until(page, f"document.querySelectorAll('.prediction-card').length >= {expect_cards} && "
                                   "!document.querySelector('.prediction-card .pending') && !document.querySelector('#predict').disabled",
                                   timeout=180000)
            return page.locator("#result").inner_text()

        text = predict([assets["tomato"]], 1)
        check("upload JPG -> prediction with 3 probability bars", "Dugaan:" in text and page.locator(".prediction-card progress").count() == 3)
        text = predict([assets["dark"]], 1)
        check("very dark photo -> darkness warning + review", "sangat gelap" in text and "Perlu tinjauan" in text)
        text = predict([assets["blur"]], 1)
        check("blurred photo -> needs review", "Perlu tinjauan" in text)
        outcome["blur_text"] = text[:400]
        text = predict([assets["blue"]], 1)
        check("non-tomato (blue/green) -> colour warning", "hampir tidak terdeteksi" in text)
        text = predict([assets["big_ok"]], 1)
        check("20 MP photo -> processed with downscale note", "diperkecil di browser" in text and "Dugaan:" in text)
        text = predict([assets["too_big"]], 1)
        check("27 MP photo -> rejected (25 MP limit)", "25 megapiksel" in text)
        text = predict([assets["heic"]], 1)
        check("HEIC -> clear conversion message", "HEIC" in text and "JPG" in text)
        text = predict([assets["text"]], 1)
        check("non-image file -> format message", "Format tidak didukung" in text)
        t0 = time.monotonic()
        text = predict(many, len(many))
        outcome["batch_12_seconds"] = round(time.monotonic() - t0, 1)
        check("12 files at once -> 12 results, no errors", page.locator(".prediction-card").count() == 12 and page.locator(".prediction-card .error").count() == 0)
        check("download JSON enabled after predictions", page.is_enabled("#download-predictions"))

        if config.get("examples") and report and report.get("examples"):
            matches = []
            for i in range(page.locator(".example-button").count()):
                page.locator(".example-button").nth(i).click()
                until(page, "document.querySelector('.prediction-card .truth') || document.querySelector('.prediction-card .error')", timeout=60000)
                matches.append("sesuai" in page.locator(".prediction-card").inner_text())
            outcome["example_matches"] = f"{sum(matches)}/{len(matches)}"
            check("all example buttons produce a result", len(matches) == len(report["examples"]))
        else:
            check("no-example mode: block hidden, no buttons", page.locator("#examples-block").is_hidden() and page.locator(".example-button").count() == 0)

        if len(models["models"]) > 1:
            other = [m["id"] for m in models["models"] if m["id"] != models["default"]][0]
            page.select_option("#model-select", other)
            until(page, f"document.querySelector('#status').textContent.includes('Model {other}')", timeout=60000)
            text = predict([assets["tomato"]], 1)
            check(f"switch model -> {other} predicts", "Dugaan:" in text and other in page.locator("#model-card").inner_text() or "Dugaan:" in text)
            page.select_option("#model-select", models["default"])
            until(page, f"document.querySelector('#status').textContent.includes('Model {models['default']}')", timeout=60000)

        for tab in ("dataset", "training", "evaluation"):
            page.click(f"[data-panel={tab}]")
            page.wait_for_selector(f"#{tab}:not([hidden])")
        if report:
            check("ablation table rendered", page.locator("#ablation tbody tr").count() >= 1)
            check("per-class table rendered", page.locator("#class-metrics tbody tr").count() == 3)

        overflow = {}
        for label, (w, h) in VIEWPORTS.items():
            page.set_viewport_size({"width": w, "height": h})
            for tab in ("prediction", "dataset", "training", "evaluation"):
                page.click(f"[data-panel={tab}]")
                if page.evaluate("document.documentElement.scrollWidth > window.innerWidth + 1"):
                    overflow.setdefault(label, []).append(tab)
            if name == "chromium":
                page.click("[data-panel=prediction]")
                page.screenshot(path=str(ROOT / f"outputs/browser/e2e-{name}-{label}.png"), full_page=True)
        check("no horizontal overflow at 1280/390/360 px", not overflow)
        outcome["overflow"] = overflow

        page.set_viewport_size({"width": 1280, "height": 900})
        page.goto(base)                      # fresh page: focus starts at the document
        until(page, "document.querySelector('#status').textContent.includes('dimuat')")
        page.keyboard.press("Tab")
        check("keyboard: first Tab focuses the skip link", page.evaluate("document.activeElement.className") == "skip")

        if axe_runner and name == "chromium":
            axe = {}
            for tab in ("prediction", "dataset", "training", "evaluation"):
                page.click(f"[data-panel={tab}]")
                result = axe_runner.run(page)
                axe[tab] = [{"id": v["id"], "impact": v["impact"], "nodes": len(v["nodes"]), "help": v["help"]}
                            for v in result.response["violations"]]
            outcome["axe"] = axe
            serious = [v for vs in axe.values() for v in vs if v["impact"] in ("serious", "critical")]
            check("axe-core: no serious/critical violations", not serious)

        check("no external requests", not outcome["external_requests"])
        check("no failed requests", not outcome["failed_requests"])
        check("no console/page errors", not outcome["errors"])
        context.close()
    except Exception as error:
        outcome["fatal"] = f"{type(error).__name__}: {error}"
    finally:
        browser.close()
    results.append(outcome)


def throttled_load(p, base):
    """Chromium only (CDP network emulation)."""
    out = {}
    for label, (down_kbps, latency) in {"4G_9Mbps": (9000, 40), "3G_1.6Mbps": (1600, 150)}.items():
        browser = p.chromium.launch()
        page = browser.new_page()
        cdp = page.context.new_cdp_session(page)
        cdp.send("Network.enable")
        cdp.send("Network.setCacheDisabled", {"cacheDisabled": True})
        cdp.send("Network.emulateNetworkConditions", {"offline": False, "latency": latency,
                 "downloadThroughput": down_kbps * 1000 / 8, "uploadThroughput": 750 * 1000 / 8})
        started = time.monotonic()
        page.goto(base)
        until(page, "document.querySelector('#status').textContent.includes('dimuat')", timeout=120000)
        out[label] = {"page_and_model_ready_ms": int((time.monotonic() - started) * 1000),
                      "model_load_ms": int(page.evaluate("document.documentElement.dataset.modelLoadMs"))}
        browser.close()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site", default=str(ROOT / "site"))
    parser.add_argument("--browsers", nargs="+", default=["chromium", "firefox", "webkit"])
    parser.add_argument("--json", default=str(ROOT / "outputs/browser/e2e.json"))
    parser.add_argument("--skip-strip-check", action="store_true")
    args = parser.parse_args()
    from playwright.sync_api import sync_playwright
    try:
        from axe_playwright_python.sync_playwright import Axe
        axe_runner = Axe()
    except ImportError:
        axe_runner = None
    (ROOT / "outputs/browser").mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="tv-e2e-"))
    assets, many = make_assets(work / "assets")
    server, base = serve(Path(args.site))
    results = []
    summary = {"site": args.site}
    with sync_playwright() as p:
        for name in args.browsers:
            run_browser(p, name, base, assets, many, results, axe_runner)
        summary["throttled_load_chromium"] = throttled_load(p, base)
        if not args.skip_strip_check:
            stripped = work / "site_stripped"
            shutil.copytree(args.site, stripped)
            sys.path.insert(0, str(ROOT / "scripts"))
            from strip_examples import strip
            strip(stripped)
            server2, base2 = serve(stripped)
            stripped_results = []
            run_browser(p, "chromium", base2, assets, many[:2], stripped_results, None)
            summary["strip_examples_mode"] = stripped_results[0]
            server2.shutdown()
    server.shutdown()
    summary["browsers"] = results
    Path(args.json).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    failed = []
    for r in results + ([summary["strip_examples_mode"]] if "strip_examples_mode" in summary else []):
        if r.get("fatal"):
            failed.append(f"{r['browser']}: {r['fatal']}")
        failed += [f"{r['browser']}: {c['check']}" for c in r["checks"] if not c["ok"]]
    for r in results:
        ok = sum(c["ok"] for c in r["checks"])
        print(f"{r['browser']:9s} {ok}/{len(r['checks'])} checks ok · model load {r.get('model_load_ms_reported')} ms"
              f" · 12 files {r.get('batch_12_seconds')} s · examples {r.get('example_matches', '-')}" + (f" · FATAL {r['fatal']}" if r.get("fatal") else ""))
    print("throttled:", json.dumps(summary["throttled_load_chromium"]))
    if "strip_examples_mode" in summary:
        s = summary["strip_examples_mode"]
        print(f"strip-examples mode: {sum(c['ok'] for c in s['checks'])}/{len(s['checks'])} checks ok")
    print("FAILED:\n  " + "\n  ".join(failed) if failed else "ALL CHECKS PASSED")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
