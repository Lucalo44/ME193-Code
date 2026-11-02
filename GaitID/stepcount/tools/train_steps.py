#!/usr/bin/env python3
"""
Train StepNet on the recorded takes, cross-validate it against a classic peak
counter, and save the final model.

    python stepcount/tools/train_steps.py
    python stepcount/tools/train_steps.py --data stepcount/data/synthetic --epochs 20
    python stepcount/tools/train_steps.py --folds 0          # skip cross-validation

Cross-validation holds out whole takes: each take is counted by a network that
never saw any part of it. Writes stepcount/models/stepnet.pt.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)

from stepcount import fuse as F
from stepcount import net as N
from stepcount import training as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "stepcount", "data", "raw"))
    ap.add_argument("--out", default=os.path.join(HERE, "stepcount", "models", "stepnet.pt"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--folds", type=int, default=4, help="cross-validation folds over takes (0 = skip)")
    ap.add_argument("--device", default="auto", help="auto (GPU if available), mps, cuda or cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = T.pick_device(args.device)
    print(f"Training on {device}")

    items = [T.prepare(t) for t in F.load_dir(args.data)]
    print(f"{len(items)} takes, {sum(it['steps'] for it in items)} counted steps, "
          f"{sum(len(it['y']) for it in items) / 50 / 60:.1f} min of data")
    for it in items:
        sensors = [s for s, x in it["streams"].items() if not np.isnan(x).all()]
        print(f"  {it['name']:<16} {it['steps']:4d} steps  {'+'.join(sensors):<9} labels: {it['label_info']['method']}")

    results = []
    folds = min(args.folds, len(items))
    if folds >= 2:
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(items))
        for f in range(folds):
            test = [items[i] for i in order[f::folds]]
            train = [items[i] for i in order if i not in order[f::folds]]
            print(f"\nFold {f + 1}/{folds}: training on {len(train)} takes, testing on {len(test)}")
            norm = T.fit_norm(train)
            model = T.train(train, norm, epochs=args.epochs, device=device, seed=args.seed + f)
            h = T.baseline_fit(train)
            for it in test:
                results.append((it, T.count(model, it["streams"], norm), T.baseline_count(it["streams"], h)))

        print(f"\n{'take':<16} {'counted':>8} {'StepNet':>8} {'peaks':>6}")
        for it, nn_c, base in sorted(results, key=lambda r: r[0]["name"]):
            print(f"{it['name']:<16} {it['steps']:8d} {nn_c:8.1f} {base:6d}")
        truth = sum(it["steps"] for it, _, _ in results)
        e_nn = sum(abs(c - it["steps"]) for it, c, _ in results)
        e_b = sum(abs(b - it["steps"]) for it, _, b in results)
        print(f"\nTotal |error| over held-out takes: StepNet {e_nn:.1f} steps ({e_nn / max(truth, 1):.1%}), "
              f"peak counter {e_b} steps ({e_b / max(truth, 1):.1%}), out of {truth}")

    print(f"\nFinal model: training on all {len(items)} takes")
    norm = T.fit_norm(items)
    model = T.train(items, norm, epochs=args.epochs, device=device, seed=args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cv = [{"take": it["name"], "steps": it["steps"], "stepnet": round(c, 2), "peaks": b} for it, c, b in results]
    N.save(model, args.out, norm, F.channel_names(), {"cv": cv, "baseline_height": T.baseline_fit(items)})
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
