"""Download recorded walks from the board into data/walks/<person>/ (skips ones already here).

    python3 tools/pull_walks.py

Uses ArduinoUNOq/tools/deploy.py's board finder (USB first, then the HOSTS addresses), so it
works wherever deploy.py works.
"""
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
TOOLS = HERE.parent / "ArduinoUNOq" / "tools"
sys.path.insert(0, str(TOOLS))

import config as C  # noqa: E402
import deploy  # noqa: E402

REMOTE = "/home/arduino/ArduinoApps/navigation/python/data/walks"


def main():
    board = deploy.find_board()
    kind, target = board
    with tempfile.TemporaryDirectory() as tmp:
        if kind == "adb":
            r = deploy.run(["adb", "pull", REMOTE, tmp], check=False)
            src = Path(tmp) / "walks"
        else:
            r = deploy.run(["scp", *deploy.SSH_OPTS, "-r", "-q", f"{target}:{REMOTE}", tmp], check=False)
            src = Path(tmp) / "walks"
        if r is None or r.returncode != 0 or not src.exists():
            raise SystemExit("No walks on the board yet (or the navigation app was never run).")
        new = 0
        for person in sorted(p for p in src.iterdir() if p.is_dir()):
            for walk in sorted(p for p in person.iterdir() if (p / "meta.json").exists()):
                dest = C.DATA_DIR / person.name / walk.name
                if dest.exists():
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(walk, dest)
                print("pulled", dest.relative_to(HERE))
                new += 1
    total = sum(1 for _ in C.DATA_DIR.glob("*/*/meta.json"))
    print(f"{new} new walk(s); {total} in data/walks (assignment cap: {C.MAX_WALKS})")


if __name__ == "__main__":
    main()
