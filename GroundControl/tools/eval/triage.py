#!/usr/bin/env python3
"""Triage the board's command log (Part 2, A2) into candidate eval cases.

    venv/bin/python tools/eval/triage.py                 # pulls logs/commands.jsonl from the board
    venv/bin/python tools/eval/triage.py --log path.jsonl

Flags commands that look like failures: an error, a "Sorry" reply, no steps for something
that isn't gibberish, a stop or a new command within 10 s (the user gave up on it), or a
normalizer change. Writes tools/eval/candidates.yaml with an empty `expect:` for a human
to fill in. Labeled cases go into dev.yaml, never straight into the held-out sets.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BOARD_LOG = "/home/arduino/ArduinoApps/ground-control/logs/commands.jsonl"


def flags(rec, nxt):
    f = []
    if rec.get("errors"):
        f.append("error")
    if str(rec.get("reply") or "").startswith("Sorry"):
        f.append("sorry")
    if not rec.get("skills") and len(str(rec.get("text", "")).split()) >= 3:
        f.append("no steps")
    if rec.get("cancelled"):
        f.append("cancelled")
    if nxt and nxt.get("source") == rec.get("source") and _gap(rec, nxt) < 10:
        f.append("followed within 10 s")
    if rec.get("changes"):
        f.append("normalized")
    return f


def _gap(a, b):
    import datetime as dt
    try:
        ta, tb = (dt.datetime.fromisoformat(x["ts"]) for x in (a, b))
        return (tb - ta).total_seconds() - (a.get("total_ms") or 0) / 1000
    except (KeyError, ValueError):
        return 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="")
    a = ap.parse_args()
    path = Path(a.log) if a.log else HERE / "results" / "commands.board.jsonl"
    if not a.log:
        r = subprocess.run(["adb", "pull", BOARD_LOG, str(path)], capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"couldn't pull the board's log: {r.stderr.strip()}")
    recs = [json.loads(line) for line in open(path) if line.strip()]
    known = set()
    for f in ("commands.yaml", "dev.yaml", "strangers.yaml", "candidates.yaml"):
        p = HERE / f
        if p.exists():
            known |= {str(c.get("text", "")).lower() for c in (yaml.safe_load(open(p)) or [])}
    out = []
    for i, rec in enumerate(recs):
        fl = flags(rec, recs[i + 1] if i + 1 < len(recs) else None)
        text = rec.get("raw") or rec.get("text")
        if fl and str(text).lower() not in known and rec.get("source") not in ("eval", "bench"):
            known.add(str(text).lower())
            out.append({"id": f"c{len(out) + 1}", "text": text, "flags": fl, "got": rec.get("skills"),
                        "reply": str(rec.get("reply") or "")[:120], "expect": None})
    dest = HERE / "candidates.yaml"
    old = yaml.safe_load(open(dest)) if dest.exists() else []
    yaml.safe_dump((old or []) + out, open(dest, "w"), sort_keys=False, allow_unicode=True, width=120)
    print(f"{len(recs)} logged commands, {len(out)} new candidates -> {dest.relative_to(HERE.parent.parent)}")


if __name__ == "__main__":
    main()
