"""Review and fix a walk's labels before training.

Taps made while walking land late, or get missed. For each walk this replays the IMU, snaps
the taps to the real turns in the heading data (nav/walks.clean), and draws a review plot:
heading per step, the detected turns, the raw taps, and where each label ended up.

    python3 tools/clean_walks.py                      # every walk: plot + print, nothing saved
    python3 tools/clean_walks.py --save               # also write labels.json (training uses it)
    python3 tools/clean_walks.py --walk <dir> --move 2:57 --drop 4 --add 88:left --save

Manual edits use the label numbers printed for that walk:
    --move N:STEP     put label N at STEP
    --drop N          delete label N
    --add STEP:MANEUVER  add a label (left / right / stairs / arrive)
    --reset           forget labels.json and start again from the taps
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402
from nav.pipeline import replay  # noqa: E402
from nav.policy import Policy  # noqa: E402
from nav.walks import clean, load_walk, walk_dirs  # noqa: E402


def plot(walk, recs, labels, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = list(range(len(recs)))
    heading = [r["state"]["heading"] for r in recs]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(steps, heading, color="#3b82f6", lw=1.8, label="heading (deg)")
    for i, r in enumerate(recs):
        if r["state"]["turned"]:
            ax.axvspan(i - 0.5, i + 0.5, color="#e8a317", alpha=0.25, lw=0)
    times = [r["t"] for r in recs]
    for tap in walk["taps"]:
        k = min(range(len(times)), key=lambda j: abs(times[j] - tap["t"])) if times else 0
        ax.axvline(k, color="#8a97ad", ls=":", lw=1)
        ax.text(k, ax.get_ylim()[1], tap["action"], rotation=90, va="top", ha="right", fontsize=8, color="#8a97ad")
    colors = {"left": "#22b07d", "right": "#d64545", "stairs": "#a855f7", "arrive": "#111111"}
    for n, m in enumerate(labels["maneuvers"]):
        ax.axvline(m["step"], color=colors[m["maneuver"]], lw=2)
        ax.text(m["step"], ax.get_ylim()[0], f" {n}:{m['maneuver']}", rotation=90, va="bottom", fontsize=9,
                color=colors[m["maneuver"]])
    for a, b in labels["stairs"]:
        ax.axvspan(a, b, color="#a855f7", alpha=0.12, lw=0)
    ax.set_xlabel("step")
    ax.set_ylabel("heading (deg)")
    ax.set_title(f"{walk['meta']['person']}/{walk['path'].name}: dotted = taps, solid = cleaned labels, "
                 "orange = detected turns")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk", help="one walk folder (default: all)")
    ap.add_argument("--test", action="store_true", help="use data/test_walks")
    ap.add_argument("--save", action="store_true", help="write labels.json")
    ap.add_argument("--move", action="append", default=[])
    ap.add_argument("--drop", action="append", type=int, default=[])
    ap.add_argument("--add", action="append", default=[])
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()
    if (a.move or a.drop or a.add) and not a.walk:
        raise SystemExit("Manual edits need --walk <folder>.")

    dirs = [Path(a.walk)] if a.walk else walk_dirs(C.TEST_DIR if a.test else C.DATA_DIR)
    for d in dirs:
        w = load_walk(d)
        recs = replay(w["imu"], policy=Policy(None))
        labels = None if a.reset else w["labels"]
        labels = labels or clean(recs, w["taps"])
        mans = labels["maneuvers"]
        for spec in a.move:
            n, step = (int(v) for v in spec.split(":"))
            mans[n] = {**mans[n], "step": step, "how": "moved by hand"}
        for n in sorted(a.drop, reverse=True):
            mans.pop(n)
        for spec in a.add:
            step, man = spec.split(":")
            if man not in C.MANEUVERS:
                raise SystemExit(f"unknown maneuver {man!r}; use one of {C.MANEUVERS}")
            mans.append({"step": int(step), "maneuver": man, "tapped_step": None, "how": "added by hand"})
        mans.sort(key=lambda m: m["step"])

        print(f"\n{w['meta']['person']}/{d.name}  ({len(recs)} steps, {len(w['taps'])} taps)")
        for n, m in enumerate(mans):
            tapped = "" if m.get("tapped_step") is None else f"  (tapped at {m['tapped_step']})"
            print(f"  {n}: step {m['step']:4d}  {m['maneuver']:7s} {m['how']}{tapped}")
        for s0, s1 in labels["stairs"]:
            print(f"     stairs steps {s0}-{s1}")
        out = d / "review.png"
        plot(w, recs, labels, out)
        print(f"  plot: {out}")
        if a.save:
            (d / "labels.json").write_text(json.dumps(labels, indent=1))
            print("  saved labels.json")


if __name__ == "__main__":
    main()
