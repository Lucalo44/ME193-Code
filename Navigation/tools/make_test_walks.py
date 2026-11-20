"""Synthetic walks for testing the pipeline before any real walk is recorded.

Builds Classroom -> Makerspace-shaped walks from our real GaitID walking recordings
(GaitID/data/raw/walk_*.csv, 50 Hz): straight segments of real walking, with turns injected by
adding a yaw rotation (about gravity) to the gyro, a short pause at a "door", and taps that land
0.3-2.0 s late like real ones. They're written to data/test_walks/, never to data/walks/,
so they can't end up in the real training set.

    python3 tools/make_test_walks.py [--n 6]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402

RAW = C.HERE.parent / "GaitID" / "data" / "raw"
# (segment, seconds): the route template; durations get +-15% per walk
ROUTE = [("walk", 25), ("left", 2.5), ("walk", 18), ("pause", 3), ("walk", 6), ("right", 2.5),
         ("walk", 30), ("left", 2.5), ("walk", 14), ("right", 2.5), ("walk", 10)]


def load(name):
    return np.loadtxt(RAW / f"{name}.csv", delimiter=",", skiprows=1)[:, 1:7]


def straighten(imu, passes=2):
    """The recordings were made walking around a room. Remove their rotation about vertical
    (estimated with the same attitude filter the navigator uses) so the route's injected turns
    are the only turns. The ankle swing itself is left untouched."""
    from nav.imu import Heading
    x = imu.copy()
    for _ in range(passes):
        h = Heading(C.FS)
        for i in range(len(x)):
            h.update(x[i:i + 1])
            x[i, 3:6] -= h.yaw_rate * h.up_body
    return x


def make(rng, walk, still, person, idx):
    out, taps, truth, t = [], [], [], 0.0
    pos = int(rng.integers(len(walk)))
    for kind, secs in ROUTE:
        n = int(secs * (1 + rng.uniform(-0.15, 0.15)) * C.FS)
        src = still if kind == "pause" else walk
        if kind == "pause":
            seg = still[:n].copy()
        else:
            idx_ = (pos + np.arange(n)) % len(walk)
            seg = walk[idx_].copy()
            pos = (pos + n) % len(walk)
        if kind in ("left", "right"):
            up = seg[:, :3].mean(0)
            up /= np.linalg.norm(up)
            rate = (90.0 / (n / C.FS)) * (1 if kind == "left" else -1)
            env = np.sin(np.linspace(0, np.pi, n)) * np.pi / 2       # smooth turn, integrates to 90 deg
            seg[:, 3:6] += (rate * env)[:, None] * up[None, :]
            truth.append({"t": t + (n / C.FS) * 0.5, "action": kind})
            taps.append({"t": t + (n / C.FS) * 0.5 + rng.uniform(0.3, 2.0), "action": kind, "source": "tap"})
        out.append(seg)
        t += n / C.FS
        del src
    imu = np.concatenate(out)
    times = np.arange(len(imu)) / C.FS
    taps.append({"t": times[-1] - rng.uniform(0, 1.0), "action": "arrive", "source": "tap"})
    d = C.TEST_DIR / person / f"test{idx:02d}_demo"
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps({"person": person, "mode": "demo", "route": C.ROUTE,
                                             "started": f"synthetic-{idx}", "fs": C.FS, "synthetic": True, "true_turns": truth}))
    np.savetxt(d / "imu.csv", np.column_stack([times, imu]), delimiter=",", fmt="%.4f",
               header="t,ax,ay,az,gx,gy,gz", comments="")
    (d / "taps.json").write_text(json.dumps(taps, indent=1))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    walk = np.concatenate([straighten(load("walk_1")), straighten(load("walk_2"))])
    still = load("still_1")
    still[:, 3:6] -= still[:, 3:6].mean(0)        # its own gyro bias would not match the walking clips
    for i in range(a.n):
        d = make(rng, walk, still, C.PEOPLE[i % len(C.PEOPLE)], i)
        print("wrote", d.relative_to(C.HERE))


if __name__ == "__main__":
    main()
