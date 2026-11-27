"""Command log (Part 2, A2): every real command, as it arrived and what became of it.

One JSON line per finished job in logs/commands.jsonl (in the app folder, so it survives
restarts): the raw text (the STT transcript for voice), what the normalizer changed, the
plan's skills, the reply, timings and errors. tools/eval/triage.py turns failures into
candidate dev cases for a human to label. Never fed straight into a held-out set.
"""
import json
import threading
import time
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "logs" / "commands.jsonl"
MAX_BYTES = 5_000_000


class CommandLog:
    def __init__(self, path=PATH):
        self.path = Path(path)
        self._lock = threading.Lock()

    def write(self, rec):
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **rec}
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                if self.path.exists() and self.path.stat().st_size > MAX_BYTES:
                    self.path.replace(self.path.with_suffix(".1.jsonl"))
                with open(self.path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
        except OSError as e:
            print(f"[log] {e!r}", flush=True)
