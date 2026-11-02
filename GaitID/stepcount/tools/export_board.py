#!/usr/bin/env python3
"""
Export a trained StepNet for the UNO Q (numpy weights, no PyTorch needed there),
and check that the numpy copy counts the same as PyTorch on an MPU-only take.

    python stepcount/tools/export_board.py
    python stepcount/tools/export_board.py --model stepcount/models/stepnet_synthetic.pt

Writes stepcount/models/stepnet_board.npz. Then run tools/sync_app.py and
redeploy so the board picks it up.
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
from stepcount import synth
from stepcount.board import BoardStepNet, StepCounter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "stepcount", "models", "stepnet.pt"))
    ap.add_argument("--out", default=os.path.join(HERE, "stepcount", "models", "stepnet_board.npz"))
    args = ap.parse_args()
    model, ck = N.load(args.model)
    arrays = {k: v.numpy() for k, v in model.state_dict().items()}
    np.savez(args.out, kernel=ck["cfg"]["kernel"], dilations=np.array(ck["cfg"]["dilations"]),
             mean=np.array(ck["norm"]["mean"]), std=np.array(ck["norm"]["std"]),
             channels=np.array(ck["channels"]), **arrays)
    print(f"Saved {args.out} ({os.path.getsize(args.out) / 1024:.0f} KB)")

    # check: PyTorch vs the numpy copy on a synthetic walk with only the MPU
    meta, data, _ = synth.make_take("walk", seconds=30, seed=12345)
    take = {"name": "check", "meta": meta, "mpu": data["mpu"], "lego": None}
    grid = F.take_grid(take)
    streams = F.on_grid(take, grid)
    torch_dens = N.predict_density(model, F.normalize(F.build(streams), ck["norm"]))
    board = BoardStepNet(args.out)
    mpu = streams["mpu"][~np.isnan(streams["mpu"]).any(axis=1)]
    np_dens = board.density(board.features(mpu))
    counter = StepCounter(board)
    for i, row in enumerate(mpu):
        counter.add(row)
        if i % 50 == 49:
            counter.update()
    counter.update(flush=True)
    print(f"Check on an MPU-only synthetic walk ({meta['steps']} steps): PyTorch {torch_dens.sum():.2f}, "
          f"numpy {np_dens.sum():.2f}, numpy streaming {counter.total:.2f}")


if __name__ == "__main__":
    main()
