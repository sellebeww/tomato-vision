#!/usr/bin/env python3
"""Remove example photos from the static demo and switch the UI to no-example mode.

Usage: python scripts/strip_examples.py [--site site]
Idempotent. Afterwards the page shows no example buttons or gallery and runs without errors.
Re-enable with: python -m src.export_web (examples are on by default).
"""
import argparse
import json
import shutil
from pathlib import Path


def strip(site: Path):
    removed = sorted(p.name for p in (site / "examples").glob("*")) if (site / "examples").exists() else []
    shutil.rmtree(site / "examples", ignore_errors=True)
    data = site / "data"
    (data / "site-config.json").write_text(json.dumps({"examples": False}, indent=2) + "\n")
    report_path = data / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        report["examples"] = []
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    leftovers = [str(p.relative_to(site)) for p in site.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg"}]
    if leftovers:
        raise SystemExit(f"Photos still present: {leftovers}")
    return removed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site", default=str(Path(__file__).resolve().parent.parent / "site"))
    removed = strip(Path(parser.parse_args().site))
    print(f"Removed {len(removed)} example photo(s); site-config.json now has examples=false.")
