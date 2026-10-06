"""
game/streak.py -- the record number of continuous hits.

A "hit" is one of the player's returns that lands legally on the opponent's
side. The current streak counts those within one rally and resets when the
point ends; the record is the best streak ever, saved to RECORD_FILE so it
survives restarts. Every time the record is beaten, `on_record(record)` is
called (main.py publishes it over MQTT).
"""

from __future__ import annotations

import json
import os
from typing import Callable, Optional

import config as C


class StreakTracker:
    def __init__(self, path: Optional[str] = None, on_record: Optional[Callable[[int], None]] = None):
        self.path = path
        self.on_record = on_record or (lambda r: None)
        self.current = 0
        self.record = 0
        if path and os.path.exists(path):
            try:
                with open(path) as f:
                    self.record = int(json.load(f)["record"])
            except (OSError, ValueError, KeyError):
                pass

    def hit(self) -> bool:
        """Count one good return. Returns True if it set a new record."""
        self.current += 1
        if self.current <= self.record:
            return False
        self.record = self.current
        self._save()
        self.on_record(self.record)
        return True

    def reset_streak(self) -> None:
        self.current = 0

    def reset_record(self) -> None:
        self.current = self.record = 0
        self._save()
        self.on_record(0)

    def _save(self) -> None:
        if not self.path:
            return
        try:
            with open(self.path, "w") as f:
                json.dump({"record": self.record}, f)
        except OSError:
            pass

    def snapshot(self) -> dict:
        return {"current": self.current, "record": self.record}


def default_path() -> str:
    return os.path.join(C.HERE, C.RECORD_FILE)
