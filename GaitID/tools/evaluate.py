#!/usr/bin/env python3
"""
Score the saved models on a folder of recordings they were NOT trained on
(e.g. record at the hip, train on the ankle, and see what breaks).

    python tools/evaluate.py data/hip_test
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from gaitid import config as C
from gaitid import data as D
from gaitid.model import Model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--models", default=os.path.join(HERE, "models"))
    args = ap.parse_args()
    X, y, _ = D.load_dir(args.data)
    for name in ("model1_peak", "model2_full"):
        m = Model.load(os.path.join(args.models, name + ".json"))
        pred = np.array([C.CLASSES.index(m.predict(w)) for w in X])
        print(f"\n{name}: {np.mean(pred == y):.1%} on {len(y)} windows")
        for i, c in enumerate(C.CLASSES):
            sel = y == i
            if sel.any():
                print(f"  {c:>6}: {np.mean(pred[sel] == i):.1%} ({sel.sum()} windows)")


if __name__ == "__main__":
    main()
