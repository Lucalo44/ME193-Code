#!/usr/bin/env python3
"""
Download the walks the UNO Q logged on its own (LOG = True in app/python/main.py),
recount them with the full model on the laptop, and print a report.

    python stepcount/tools/pull_logs.py                    # download + report
    python stepcount/tools/pull_logs.py --clear            # also free the board (keeps the newest session)
    python stepcount/tools/pull_logs.py --label session_0003 --steps 412 --activity walk

--label turns a logged walk you counted by hand into an MPU-only training take
in stepcount/data/raw/, so walks to class add to the training data.
Finds the board the same way ArduinoUNOq/tools/deploy.py does (USB first, then Wi-Fi).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)

from stepcount import fuse as F

LOCAL = os.path.join(HERE, "stepcount", "data", "board_logs")
REMOTE_LOGS = "/home/arduino/ArduinoApps/gaitid/python/logs"
ARDUINO_TOOLS = os.path.join(HERE, "..", "ArduinoUNOq", "tools")


def deploy_module():
    sys.path.insert(0, os.path.abspath(ARDUINO_TOOLS))
    import deploy                                       # noqa: E402 -- the repo's board connection helper
    return deploy


def pull():
    deploy = deploy_module()
    board = deploy.find_board()
    kind, target = board
    os.makedirs(LOCAL, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        if kind == "adb":
            r = deploy.run(["adb", "pull", REMOTE_LOGS, tmp], check=False)
        else:
            r = deploy.run(["scp", *deploy.SSH_OPTS, "-r", "-q", f"{target}:{REMOTE_LOGS}", tmp], check=False)
        got = sorted(glob.glob(os.path.join(tmp, "logs", "session_*")))
        if r is None or not got:
            raise SystemExit("Nothing downloaded. Is LOG = True on the board, and has it been running?")
        for f in got:                                   # a newer copy of the running session replaces the old one
            shutil.copy(f, LOCAL)
    print(f"Downloaded {len(got)} files to {LOCAL}")
    return deploy, board, [os.path.basename(f) for f in got]


def clear(deploy, board, names):
    sessions = sorted({n.rsplit(".", 1)[0] for n in names})
    old = sessions[:-1]                                 # the newest may still be recording
    if not old:
        print("Only one session on the board; leaving it.")
        return
    files = " ".join(f"{REMOTE_LOGS}/{s}.csv {REMOTE_LOGS}/{s}.json" for s in old)
    deploy.remote(board, f"rm -f {files}", check=False)
    print(f"Removed {len(old)} older sessions from the board (kept {sessions[-1]})")


def load_log(name):
    data = np.genfromtxt(os.path.join(LOCAL, name + ".csv"), delimiter=",", skip_header=1, invalid_raise=False)
    data = data[~np.isnan(data).any(axis=1)] if data.ndim == 2 else np.zeros((0, 7))
    return data[:, 0], data[:, 1:]


def report(model_path):
    from stepcount import net as N
    from stepcount import training as T
    model, ck = N.load(model_path) if os.path.exists(model_path) else (None, None)
    names = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(LOCAL, "session_*.csv")))
    print(f"\n{'session':<28} {'started':<20} {'minutes':>7} {'board':>7} {'laptop':>7} {'steps/min':>9}")
    for name in names:
        summary = {}
        if os.path.exists(os.path.join(LOCAL, name + ".json")):
            with open(os.path.join(LOCAL, name + ".json")) as f:
                summary = json.load(f)
        t, v = load_log(name)
        minutes = (t[-1] - t[0]) / 60 if len(t) > 1 else 0.0
        recount = None
        if model is not None and len(t) > 100:
            take = {"name": name, "meta": {"steps": 0}, "mpu": (t - t[0], v), "lego": None}
            recount = T.count(model, F.on_grid(take, F.take_grid(take)), ck["norm"])
        board = summary.get("steps_on_board")
        best = recount if recount is not None else board
        cad = f"{best / minutes:9.0f}" if best is not None and minutes > 0.1 else f"{'-':>9}"
        print(f"{name:<28} {summary.get('started', '?'):<20} {minutes:7.1f} "
              f"{'-' if board is None else f'{board:.0f}':>7} {'-' if recount is None else f'{recount:.0f}':>7} {cad}")
    if model is None:
        print(f"(no model at {model_path}, so no laptop recount)")


def label(name, steps, activity):
    t, v = load_log(name)
    out = os.path.join(HERE, "stepcount", "data", "raw")
    n = len(glob.glob(os.path.join(out, f"{activity}_*"))) + 1
    folder = os.path.join(out, f"{activity}_{n}")
    F.save_take(folder, {"activity": activity, "steps": steps, "seconds": round(float(t[-1] - t[0]), 2),
                         "sensors": ["mpu"], "source": f"board log {name}",
                         "placement": {"mpu": "right ankle"}}, mpu=(t - t[0], v))
    print(f"Saved {folder} (MPU only) as a training take with {steps} steps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clear", action="store_true", help="delete downloaded sessions from the board, except the newest")
    ap.add_argument("--no-pull", action="store_true", help="just report on what's already downloaded")
    ap.add_argument("--label", metavar="SESSION", help="make a training take from this session (prefix is enough)")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--activity", default="walk")
    ap.add_argument("--model", default=os.path.join(HERE, "stepcount", "models", "stepnet.pt"),
                    help="model for the laptop recount")
    args = ap.parse_args()

    if args.label:
        if args.steps is None:
            raise SystemExit("--label needs --steps (how many steps that walk really was)")
        match = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(LOCAL, args.label + "*.csv")))
        if len(match) != 1:
            raise SystemExit(f"'{args.label}' matches {len(match)} downloaded sessions: {match}")
        label(match[0], args.steps, args.activity)
        return
    if not args.no_pull:
        deploy, board, names = pull()
        if args.clear:
            clear(deploy, board, names)
    report(args.model)


if __name__ == "__main__":
    main()
