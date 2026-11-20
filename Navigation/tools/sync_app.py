"""Assemble the UNO Q app in ArduinoUNOq/navigation/ and optionally deploy it.

The app's own files (app.yaml, sketch/, python/main.py) live in Navigation/app/; this copies
them plus nav/, gaitid/, web/, config.py and models/ into ArduinoUNOq/navigation/, which is the
folder ArduinoUNOq/tools/deploy.py pushes to the board.

    python3 tools/sync_app.py                    # copy only
    python3 tools/sync_app.py --deploy           # ...and deploy (rebuilds + reflashes the STM32)
    python3 tools/sync_app.py --restart-python   # ...copy and restart only the Python side

Navigation/ stays the one place to edit and train; ArduinoUNOq/navigation/ only holds copies.
Re-run after every retrain. --restart-python is enough when only Python
files or models changed.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
APP = HERE.parent / "ArduinoUNOq" / "navigation"
DEST = APP / "python"
DIRS = ["nav", "gaitid", "web"]
FILES = ["config.py"]
MODELS = ["stepnet_board.npz", "activity.json", "stairs.json", "policy.npz"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy", action="store_true")
    ap.add_argument("--restart-python", action="store_true")
    a = ap.parse_args()

    shutil.copytree(HERE / "app", APP, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for d in DIRS:
        shutil.rmtree(DEST / d, ignore_errors=True)
        shutil.copytree(HERE / d, DEST / d,                    # training.py needs torch: laptop only
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "training.py"))
    for f in FILES:
        shutil.copy2(HERE / f, DEST / f)
    (DEST / "models").mkdir(exist_ok=True)
    copied = []
    for m in MODELS:
        src = HERE / "models" / m
        if src.exists():
            shutil.copy2(src, DEST / "models" / m)
            copied.append(m)
        elif (DEST / "models" / m).exists():
            (DEST / "models" / m).unlink()             # e.g. stairs.json after retraining without stairs
    print(f"Copied {', '.join(DIRS + FILES)} and models {copied} -> {DEST}")
    if "policy.npz" not in copied:
        print("Note: no policy.npz yet, so the board can record demos but can't navigate.")

    deploy = APP.parent / "tools" / "deploy.py"
    if a.restart_python:
        r = subprocess.call([sys.executable, str(deploy), "navigation", "--copy"], cwd=APP.parent)
        if r == 0:
            r = subprocess.call([sys.executable, str(deploy), "--cmd", "docker restart navigation-main-1"],
                                cwd=APP.parent)
        sys.exit(r)
    if a.deploy:
        sys.exit(subprocess.call([sys.executable, str(deploy), "navigation"], cwd=APP.parent))


if __name__ == "__main__":
    main()
