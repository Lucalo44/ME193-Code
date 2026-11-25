"""A small persistent memory: "remember my name is Luca" -> save name=Luca.

Stored as JSON in the app folder (/app/memory.json on the board), so it survives
restarts and redeploys (deploys copy files but never delete). Input comes from a
public broker, so it's treated as plain text: size-capped and never executed.
"""
import difflib
import json
import re
import threading
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "memory.json"
MAX_KEY, MAX_VALUE, MAX_ITEMS = 30, 120, 100


def _key(k):
    k = (k or "").lower().strip()
    k = re.sub(r"^(my|your|the)\s+", "", k)
    return re.sub(r"\s+", " ", k)[:MAX_KEY]


class Memory:
    def __init__(self, path=PATH):
        self.path = Path(path)
        self._lock = threading.Lock()

    def _load(self):
        try:
            d = json.loads(self.path.read_text())
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self, d):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1))
        tmp.replace(self.path)

    def _find(self, d, key):
        if key in d:
            return key
        close = difflib.get_close_matches(key, list(d), n=1, cutoff=0.6)
        if close:
            return close[0]
        for k in d:                       # "favorite color" vs "color"
            if key in k or k in key:
                return k
        return None

    def do(self, op, key, value=""):
        key = _key(key)
        value = (value or "").strip()[:MAX_VALUE]
        with self._lock:
            d = self._load()
            if op == "save":
                if not key or not value:
                    return "I need both what to remember and its value."
                if key not in d and len(d) >= MAX_ITEMS:
                    return "My memory is full. Ask me to forget something first."
                d[key] = value
                self._save(d)
                return f"Got it, I'll remember your {key}: {value}."
            if op == "recall":
                if not key:
                    if not d:
                        return "I don't have anything saved yet."
                    return "I remember: " + "; ".join(f"{k}: {v}" for k, v in d.items()) + "."
                k = self._find(d, key)
                if k is None:
                    return f"I don't know your {key} yet. Tell me and I'll remember it."
                return f"Your {k} is {d[k]}."
            if op == "forget":
                k = self._find(d, key) if key else None
                if k is None:
                    return f"I didn't have anything saved for {key or 'that'}."
                del d[k]
                self._save(d)
                return f"Okay, I forgot your {k}."
        return f"unknown memory operation {op}"


OPS_LIST = ["save", "recall", "forget"]
