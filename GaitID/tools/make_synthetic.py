#!/usr/bin/env python3
"""Write fake recordings to data/synthetic/ to try the pipeline without hardware."""

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from gaitid import config as C
from gaitid import synth

out = os.path.join(HERE, "data", "synthetic")
os.makedirs(out, exist_ok=True)
for label in C.CLASSES:
    for k in range(4):
        synth.write_csv(os.path.join(out, f"{label}_{k + 1}.csv"),
                        synth.make_recording(label, 30, seed=hash((label, k)) % 2**32))
print(f"Wrote {len(C.CLASSES) * 4} recordings to {out}")
