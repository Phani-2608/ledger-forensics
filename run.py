#!/usr/bin/env python3
"""Ledger Forensics entry point.

    python run.py smoke     tiny run, ~1 minute, used by CI
    python run.py demo      default, a few minutes
    python run.py full      the numbers reported in the README

Stages can be run separately on a slow machine; the dataset is cached and
results.json is merged, not overwritten:

    python run.py demo core
    python run.py demo holdout
    python run.py demo curves decide
    python run.py api       serve the FastAPI app on :8000
    python run.py dashboard launch the Streamlit dashboard
"""
import sys


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"

    if cmd in ("smoke", "demo", "full"):
        from ledger_forensics.experiments.runner import run
        stages = sys.argv[2:] or None
        run(profile=cmd, stages=stages)
        return 0

    if cmd == "api":
        import uvicorn
        uvicorn.run("ledger_forensics.api.app:app", host="0.0.0.0", port=8000)
        return 0

    if cmd == "dashboard":
        import subprocess
        return subprocess.call(
            [sys.executable, "-m", "streamlit", "run", "dashboard/app.py"])

    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
