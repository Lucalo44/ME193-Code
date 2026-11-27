"""Perception priority policy (Part 2, A0): who gets the CPU when.

The board has 4 Cortex-A53 cores and llama.cpp uses all of them while it writes a plan.
Every on-board perception model (keyword spotting, pose, beat tracking) asks this policy
how often it may run in the current mode:

  PLANNING   the LLM is decoding: pose drops to 2 fps; beat tracking keeps running (cheap)
  DANCING    the conductor is performing: perception runs, the planner is idle
  MIRRORING  "copy me": pose runs at full rate
  IDLE       everything runs at its normal rate

Keyword spotting always runs at full rate, in every mode, because "stop" must never wait.
docs/SPIKES.md has the measured tok/s cost of each model and of core pinning.
"""
import os
import threading

MODES = ("PLANNING", "DANCING", "MIRRORING", "IDLE")

# Max runs per second per perception model, by mode (None = no limit).
POLICY = {
    "PLANNING":  {"kws": None, "beats": None, "pose": 2},
    "DANCING":   {"kws": None, "beats": None, "pose": 8},
    "MIRRORING": {"kws": None, "beats": None, "pose": None},
    "IDLE":      {"kws": None, "beats": None, "pose": 5},
}

# Core pinning (measured in docs/SPIKES.md): perception on the last core, llama.cpp on the rest.
PERCEPTION_CORES = {3}


class Budget:
    def __init__(self):
        self._mode = "IDLE"
        self._lock = threading.Lock()
        self._listeners = []

    @property
    def mode(self):
        return self._mode

    def set_mode(self, mode):
        if mode not in MODES:
            raise ValueError(mode)
        with self._lock:
            changed, self._mode = mode != self._mode, mode
        if changed:
            for fn in list(self._listeners):
                fn(mode)

    def on_change(self, fn):
        self._listeners.append(fn)

    def max_fps(self, model):
        """None = run as fast as you like; a number = at most this many runs per second."""
        return POLICY[self._mode].get(model)

    def interval_s(self, model, natural_s):
        fps = self.max_fps(model)
        return natural_s if fps is None else max(natural_s, 1.0 / fps)


def mode_for_state(state):
    """Map the app's status state to a budget mode."""
    return {"PLANNING": "PLANNING", "DANCING": "DANCING", "MIRRORING": "MIRRORING"}.get(state, "IDLE")


def pin_to_perception_cores():
    """Pin the calling thread to the perception core(s). No-op where unsupported (macOS)."""
    try:
        os.sched_setaffinity(0, PERCEPTION_CORES)
        return True
    except (AttributeError, OSError):
        return False
