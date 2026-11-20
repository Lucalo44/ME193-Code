"""How good is the navigator on walks it hasn't seen?

    python3 tools/evaluate.py            # leave-one-walk-out over data/walks
    python3 tools/evaluate.py --test     # same, on the synthetic walks
    python3 tools/evaluate.py --plot     # also save a per-walk plot of predicted vs true distance

Leave-one-walk-out: for every walk, train on all the others and test on it. Reported:
maneuver accuracy (every step, and within 20 steps of a maneuver), distance error in steps,
and when "<maneuver> now" was first said relative to the real maneuver (- = early).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402
from nav import training  # noqa: E402
from nav.walks import load_walk, walk_dirs  # noqa: E402
from train import print_loo  # noqa: E402


def plot_walk(policy, prepared, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    preds = [policy.act(x) for x in prepared["X"]]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(prepared["dist"], color="#8a97ad", lw=2, label="true steps to next maneuver")
    ax.plot([p["steps"] for p in preds], color="#e8a317", lw=1.5, label="predicted")
    for m in prepared["labels"]["maneuvers"]:
        ax.axvline(m["step"], color="#3b82f6", alpha=0.4)
        ax.text(m["step"], ax.get_ylim()[1], m["maneuver"], rotation=90, va="top", fontsize=8)
    ax.set_xlabel("step")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    walks = [load_walk(d) for d in walk_dirs(C.TEST_DIR if a.test else C.DATA_DIR)]
    if len(walks) < 2:
        raise SystemExit("Need at least 2 walks for leave-one-walk-out.")
    results = training.leave_one_out(walks, epochs=a.epochs)
    print_loo(results)
    if a.plot:
        from nav.policy import Policy
        pol = Policy(C.MODEL_DIR / C.POLICY_MODEL)
        if not pol.loaded:
            raise SystemExit("--plot needs a trained models/policy.npz (run tools/train.py).")
        for w in walks:
            p = training.prepare([w], route_len=pol.route_len)[0]
            out = w["path"] / "eval.png"
            plot_walk(pol, p, out)
            print("plot:", out)


if __name__ == "__main__":
    main()
