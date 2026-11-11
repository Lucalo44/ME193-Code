"""Copy the safe's code, trained models and code file into the UNO Q app, then deploy.

    python3 sync_to_unoq.py            # copy into ArduinoUNOq/safe/python/
    python3 sync_to_unoq.py --deploy   # ...and run tools/deploy.py safe

Safe/ stays the one place to edit and train; ArduinoUNOq/safe/python/ only holds copies
(they're git-ignored there). Re-run this after every retrain or set_password.py.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE.parent / "ArduinoUNOq" / "safe"
DEST = APP / "python"

# Everything run_safe.py imports (no collection/training scripts: those stay on the laptop)
MODULES = ["run_safe.py", "config.py", "preprocess.py", "models.py", "audio.py", "ui.py",
           "safe_io.py", "set_password.py", "camera_picker.py", "mic_picker.py", "stream.py"]
OPTIONAL = ["password.json", "mic.json"]     # mic.json: its name is a hint for --mic auto


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy", action="store_true", help="also run ArduinoUNOq/tools/deploy.py safe")
    a = ap.parse_args()

    if not (HERE / "password.json").exists():
        raise SystemExit("No password.json: run set_password.py first.")
    (DEST / "models").mkdir(parents=True, exist_ok=True)
    for name in MODULES:
        shutil.copy2(HERE / name, DEST / name)
    for name in OPTIONAL:
        if (HERE / name).exists():
            shutil.copy2(HERE / name, DEST / name)
    nets = sorted((HERE / "models").glob("*.pt"))
    for pt in nets:
        shutil.copy2(pt, DEST / "models" / pt.name)
    print(f"Copied {len(MODULES)} modules, {[p.name for p in nets]} and password.json -> {DEST}")
    if not any(p.name == "gesture.pt" for p in nets):
        print("Warning: no gesture.pt, so the safe won't start on the board.")

    if a.deploy:
        tools = APP.parent / "tools" / "deploy.py"
        sys.exit(subprocess.call([sys.executable, str(tools), "safe"], cwd=APP.parent))
    print("Next: cd ../ArduinoUNOq && python3 tools/deploy.py safe")


if __name__ == "__main__":
    main()
