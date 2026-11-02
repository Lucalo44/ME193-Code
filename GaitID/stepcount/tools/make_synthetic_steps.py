#!/usr/bin/env python3
"""
Write synthetic two-ankle takes to stepcount/data/synthetic to try the whole
pipeline without hardware. Accuracy on these means nothing.

    python stepcount/tools/make_synthetic_steps.py
    python stepcount/tools/train_steps.py --data stepcount/data/synthetic --epochs 20
"""

from __future__ import annotations

import os
import shutil
import sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)

from stepcount import fuse as F
from stepcount import synth

PLAN = {"walk": 5, "run": 4, "mixed": 3, "still": 2, "bike": 2}

out = os.path.join(HERE, "stepcount", "data", "synthetic")
shutil.rmtree(out, ignore_errors=True)
seed = 0
for activity, n in PLAN.items():
    for i in range(1, n + 1):
        meta, data, _ = synth.make_take(activity, seconds=30 + 6 * i, seed=seed)
        F.save_take(os.path.join(out, f"{activity}_{i}"), meta, mpu=data["mpu"], lego=data["lego"])
        seed += 1
print(f"Wrote {sum(PLAN.values())} takes to {out}")
