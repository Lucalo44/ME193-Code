"""Walk recordings and their labels.

A walk is a folder data/walks/<person>/<id>/ holding:
    meta.json    person, mode (demo / dagger), route, start time
    imu.csv      every raw sample: t, ax, ay, az, gx, gy, gz
    taps.json    what was tapped during the walk: [{t, action, source}]
    labels.json  (optional) cleaned maneuvers, written by tools/clean_walks.py

Raw IMU is stored, not features, so a walk can be replayed through the current pipeline
after any change to features or models.

Label cleaning: taps made while walking land late, or get missed. clean() snaps each turn
tap to the real turn in the heading data nearby (within SNAP_STEPS), shifts other taps back
by a reaction time, and adds "arrive" at the end if it was never tapped.
"""
import json
import time
from pathlib import Path

import numpy as np

import config as C


# ---------------- storage ----------------
class WalkWriter:
    def __init__(self, root, person, mode):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = Path(root) / person / f"{stamp}_{mode}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta = {"person": person, "mode": mode, "route": C.ROUTE, "started": stamp, "fs": C.FS}
        self.imu = open(self.dir / "imu.csv", "w")
        self.imu.write("t,ax,ay,az,gx,gy,gz\n")
        self.taps = []
        self.samples = 0
        self._save_meta()

    def add_samples(self, rows):
        for r in rows:
            self.imu.write(",".join(f"{v:.4f}" for v in r) + "\n")
        self.samples += len(rows)

    def tap(self, t, action, source="tap"):
        self.taps.append({"t": float(t), "action": action, "source": source})
        self._save_taps()

    def _save_taps(self):
        (self.dir / "taps.json").write_text(json.dumps(self.taps, indent=1))

    def _save_meta(self):
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=1))

    def close(self):
        self.imu.close()
        self.meta["samples"] = self.samples
        self._save_meta()
        self._save_taps()
        return self.dir

    def discard(self):
        self.imu.close()
        for f in self.dir.iterdir():
            f.unlink()
        self.dir.rmdir()


def load_walk(path):
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    imu = np.loadtxt(path / "imu.csv", delimiter=",", skiprows=1, ndmin=2)
    taps = json.loads((path / "taps.json").read_text()) if (path / "taps.json").exists() else []
    labels = json.loads((path / "labels.json").read_text()) if (path / "labels.json").exists() else None
    return {"path": path, "meta": meta, "imu": imu, "taps": taps, "labels": labels}


def walk_dirs(root=C.DATA_DIR, people=None):
    root = Path(root)
    people = people or sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []
    out = []
    for person in people:
        d = root / person
        if d.exists():
            out += sorted(p for p in d.iterdir() if (p / "meta.json").exists() and (p / "imu.csv").exists())
    return out


# ---------------- cleaning ----------------
def _nearest_step(records, t):
    times = np.array([r["t"] for r in records])
    return int(np.argmin(np.abs(times - t))) if len(times) else 0


def clean(records, taps):
    """records: step records from pipeline.replay(). Returns
    {"maneuvers": [{step, maneuver, tapped_step, how}], "stairs": [[start_step, end_step], ...]}
    with steps as 0-based indexes into records."""
    n = len(records)
    if n == 0:
        return {"maneuvers": [], "stairs": []}
    heading = np.array([r["state"]["heading"] for r in records])
    rate = np.zeros(n)
    rate[1:-1] = heading[2:] - heading[:-2]                    # heading change centred on each step
    maneuvers, stairs_open, stairs = [], None, []
    for tap in sorted(taps, key=lambda x: x["t"]):
        a = tap["action"]
        k = _nearest_step(records, tap["t"])
        if a in ("left", "right"):
            sign = 1 if a == "left" else -1
            lo, hi = max(0, k - C.SNAP_STEPS), min(n, k + C.SNAP_STEPS + 1)
            r = rate[lo:hi] * sign
            j = int(np.argmax(r))
            if r[j] > 20:                                        # a real turn of the right direction
                step, how = lo + j, "snapped to turn"
            else:
                step, how = max(0, k - C.REACTION_STEPS), "no turn found: reaction shift"
            maneuvers.append({"step": step, "maneuver": a, "tapped_step": k, "how": how})
        elif a == "stairs_start":
            step = max(0, k - C.REACTION_STEPS)
            maneuvers.append({"step": step, "maneuver": "stairs", "tapped_step": k, "how": "reaction shift"})
            stairs_open = step
        elif a == "stairs_end":
            if stairs_open is not None:
                stairs.append([stairs_open, max(stairs_open, k - C.REACTION_STEPS)])
                stairs_open = None
        elif a == "arrive":
            maneuvers.append({"step": k, "maneuver": "arrive", "tapped_step": k, "how": "tapped"})
    if stairs_open is not None:
        stairs.append([stairs_open, n - 1])
    if not any(m["maneuver"] == "arrive" for m in maneuvers):
        maneuvers.append({"step": n - 1, "maneuver": "arrive", "tapped_step": None, "how": "end of walk"})
    maneuvers.sort(key=lambda m: m["step"])
    last_arrive = max(m["step"] for m in maneuvers if m["maneuver"] == "arrive")
    maneuvers = [m for m in maneuvers if m["step"] <= last_arrive]   # nothing after arriving
    return {"maneuvers": maneuvers, "stairs": stairs}


def targets(labels, n):
    """Per step: (next maneuver index, steps until it, stairs 0/1), for steps up to arrival."""
    mans = labels["maneuvers"]
    man_idx, dist, stairs = [], [], np.zeros(n)
    for a, b in labels["stairs"]:
        stairs[a:b + 1] = 1
    j = 0
    end = max(m["step"] for m in mans if m["maneuver"] == "arrive")
    for i in range(end + 1):
        while j < len(mans) and mans[j]["step"] < i:
            j += 1
        if j >= len(mans):
            break
        man_idx.append(C.MANEUVERS.index(mans[j]["maneuver"]))
        dist.append(mans[j]["step"] - i)
    m = len(man_idx)
    return np.array(man_idx), np.array(dist, float), stairs[:m]
