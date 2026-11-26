#!/usr/bin/env python3
"""Deploy Ground Control to the UNO Q.

    python3 tools/deploy.py              # stop whatever runs, copy the app, start it
    python3 tools/deploy.py --copy       # copy only (then: --restart-python)
    python3 tools/deploy.py --restart-python
    python3 tools/deploy.py --logs [N]
    python3 tools/deploy.py --cmd "..."  # run a shell command on the board
    python3 tools/deploy.py --lego       # copy + (re)start the LEGO BLE service on the board host

Reuses the board-finding code in ArduinoUNOq/tools/deploy.py (USB/adb first,
then the HOSTS in board_secrets.py). Only the parts the board needs are copied
(app.yaml, python/, sketch/, assets/), so laptop-only tools, tests and the KWS
dataset never land on the board.
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP_DIR = HERE.parent
sys.path.insert(0, str(APP_DIR.parent / "ArduinoUNOq" / "tools"))

import deploy as kit  # noqa: E402  (ArduinoUNOq/tools/deploy.py)

APP_NAME = "ground-control"
REMOTE_APP = f"{kit.REMOTE_DIR}/{APP_NAME}"
CONTAINER = f"{APP_NAME}-main-1"
APP_ITEMS = ["app.yaml", "python", "sketch", "assets"]
LEGO_REMOTE = "/home/arduino/gc-lego"
LEGO_PY = "/home/arduino/gc-lego-venv/bin/python"


def clean_pycache():
    for p in APP_DIR.rglob("__pycache__"):
        for f in p.iterdir():
            f.unlink()
        p.rmdir()


def copy(board):
    clean_pycache()
    kit.remote(board, f"mkdir -p {REMOTE_APP}")
    kit.push(board, [str(APP_DIR / i) for i in APP_ITEMS], REMOTE_APP)


def lego(board):
    kit.remote(board, f"mkdir -p {LEGO_REMOTE}")
    kit.push(board, [str(APP_DIR / "lego_service" / "lego_server.py")], LEGO_REMOTE)
    kit.remote(board, "pkill -f 'gc-lego-venv/bin/python.*lego_server.py'; sleep 1", check=False)
    # setsid + </dev/null so adb/ssh don't wait on the detached process
    kit.remote(board, f"cd {LEGO_REMOTE} && setsid nohup {LEGO_PY} -u lego_server.py "
                      f"< /dev/null > {LEGO_REMOTE}/lego.log 2>&1 &", check=False, timeout=20)
    time.sleep(2)
    kit.remote(board, f"tail -5 {LEGO_REMOTE}/lego.log", check=False)


MODELS_DIR = "/var/lib/arduino-app-cli/models/llamacpp"


def install_model(board, path):
    """Copy a GGUF to the board and register it with App Lab, then restart the runner.

    App Lab lists a model only if its folder has an .arduino_metadata.yaml (the file its
    downloader writes), and it rebuilds the runner's models.ini from those on every app
    start. The id is "custom/<file stem>", e.g. llamacpp:custom/gc-planner-Q4_0 in app.yaml
    and custom/gc-planner-Q4_0 as the runner's model name."""
    p = Path(path)
    if not p.exists() or p.suffix != ".gguf":
        sys.exit(f"not a .gguf file: {path}")
    name = p.stem
    kit.remote(board, f"mkdir -p {MODELS_DIR}/custom")
    kit.push(board, [str(p)], f"{MODELS_DIR}/custom")
    meta = (f"# Written by Ground Control tools/deploy.py --model (copied over adb).\\n"
            f"models:\\n- downloaded_at: \\042{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\\042\\n"   # \\042 = " for printf
            f"  handler: hf-handler\\n  model_id: llamacpp:custom/{name}\\n  model_origin: user\\n"
            f"  files:\\n  - {p.name}\\n  inputs:\\n    models_repository: llamacpp\\n"
            f"    model_directory: custom\\n    model_url: llamacpp:custom/{p.name}\\n")
    kit.remote(board, f"printf '{meta}' > {MODELS_DIR}/custom/.arduino_metadata.yaml", check=False)
    kit.remote(board, f"grep -q '^\\[custom/{name}\\]' {MODELS_DIR}/models.ini || "
                      f"printf '\\n[custom/{name}]\\nmodel = /models/custom/{p.name}\\n' >> {MODELS_DIR}/models.ini",
               check=False)
    kit.remote(board, f"arduino-app-cli model list | grep llamacpp; df -h / | tail -1", check=False)
    print(f"app.yaml:  model: llamacpp:custom/{name}    config.py:  custom/{name}")
    kit.remote(board, "docker restart ground-control-llamacpp-models-runner-1 > /dev/null; sleep 8; "
                      "curl -s 127.0.0.1:9999/v1/models", check=False, timeout=90)


def main():
    args = sys.argv[1:]
    board = kit.find_board()
    if args[:1] == ["--cmd"]:
        r = kit.remote(board, " ".join(args[1:]), check=False)
        sys.exit(r.returncode if r else 1)
    if args[:1] == ["--logs"]:
        n = args[1] if len(args) > 1 else "40"
        kit.remote(board, f"arduino-app-cli app logs {REMOTE_APP} --tail {n}", check=False)
        return
    if args[:1] == ["--restart-python"]:
        kit.remote(board, f"docker restart {CONTAINER}", check=False)
        return
    if args[:1] == ["--model"]:
        install_model(board, args[1])
        return
    if args[:1] == ["--lego"]:
        lego(board)
        return
    if args[:1] == ["--copy"]:
        copy(board)
        print("Copied (not restarted).")
        return

    r = kit.remote(board, "docker ps --format '{{.Names}}' | grep -- '-main-1$'",
                   check=False, quiet=True, timeout=15)
    running = [n[:-len("-main-1")] for n in r.stdout.decode().split()] if r and r.returncode == 0 else []
    for name in running:
        kit.remote(board, f"arduino-app-cli app stop {kit.REMOTE_DIR}/{name}", check=False, timeout=60)
    copy(board)
    kit.remote(board, f"arduino-app-cli app start {REMOTE_APP}", tty=True)
    time.sleep(5)
    kit.remote(board, f"arduino-app-cli app logs {REMOTE_APP} --tail 25", check=False)
    url = kit.web_url(board)
    if url:
        # The app serves https (self-signed) so the dashboard's mic works; accept the warning once.
        print(f"\nDashboard: {url.replace('http://', 'https://')}")


if __name__ == "__main__":
    main()
