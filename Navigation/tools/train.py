"""Train the navigator on every walk from both partners (demos + DAgger), then save
models/policy.npz (+ models/stairs.json if there are stairs labels) for the board.

    python3 tools/train.py                  # data/walks/, all people
    python3 tools/train.py --people luca    # one person
    python3 tools/train.py --test           # the synthetic walks in data/test_walks/
    python3 tools/train.py --loo            # also run leave-one-walk-out evaluation

Then tools/sync_app.py copies the models into the UNO Q app.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402
from nav import training  # noqa: E402
from nav.walks import load_walk, walk_dirs  # noqa: E402


def timing_text(t):
    """'left:-2' = said 'turn left now' 2 steps before the real turn; 'missed' = never said it."""
    at = t["called_at"]
    return f"{t['maneuver']}:" + ("missed" if at is None else f"{at:+d}")


def print_loo(results):
    print("\nLeave-one-walk-out (train on the others, test on this one):")
    print(f"  {'walk':32s} {'maneuver acc':>12s} {'within 20 steps':>16s} {'dist err (steps)':>17s}  'now' timing")
    for r in results:
        if "acc" not in r:
            continue
        tim = ", ".join(timing_text(t) for t in r["timing"])
        print(f"  {r['walk']:32s} {r['acc']:12.1%} {r['acc_near']:16.1%} {r['mae_near']:17.1f}  {tim}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", nargs="*")
    ap.add_argument("--test", action="store_true", help="train on data/test_walks (synthetic)")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--loo", action="store_true", help="leave-one-walk-out evaluation")
    ap.add_argument("--out", default=str(C.MODEL_DIR))
    a = ap.parse_args()

    root = C.TEST_DIR if a.test else C.DATA_DIR
    dirs = walk_dirs(root, a.people)
    if not dirs:
        raise SystemExit(f"No walks in {root}. Record some (or run tools/make_test_walks.py and use --test).")
    if len(dirs) > C.MAX_WALKS and not a.test:
        raise SystemExit(f"{len(dirs)} walks: the assignment allows {C.MAX_WALKS} (demos + DAgger). Remove some.")
    walks = [load_walk(d) for d in dirs]
    print(f"{len(walks)} walks: " + ", ".join(f"{w['meta']['person']}/{w['path'].name}" for w in walks))

    trained, stairs, rep = training.fit(walks, epochs=a.epochs, verbose=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    training.to_policy(trained, out / C.POLICY_MODEL)
    if stairs.trained:
        stairs.save(out / C.STAIRS_MODEL)
    elif (out / C.STAIRS_MODEL).exists():
        (out / C.STAIRS_MODEL).unlink()
    print(f"\nroute length {rep['route_len']:.0f} steps (~{rep['route_len'] * C.STRIDE_M:.0f} m)")
    print("stairs model:", rep["stairs"])
    for p in rep["prepared"]:
        moved = [m for m in p["labels"]["maneuvers"] if m.get("tapped_step") is not None]
        shifts = ", ".join(f"{m['maneuver']} {m['tapped_step']}->{m['step']}" for m in moved)
        print(f"  {p['walk']['path'].name}: {p['n']} steps, labels: {shifts}")
    print(f"saved {out / C.POLICY_MODEL}")
    if a.loo and len(walks) > 1:
        print_loo(training.leave_one_out(walks, epochs=a.epochs))


if __name__ == "__main__":
    main()
