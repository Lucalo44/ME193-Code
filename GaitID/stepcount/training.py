"""
Training and evaluation for StepNet (tools/train_steps.py is the command line).

Every batch is a set of random 8 s crops from the takes, each augmented the
way real recordings differ from each other:
  * strap rotation   -- each IMU's axes rotated up to 20 degrees
  * gain / noise     -- +/-10 % amplitude, a little sensor noise
  * clock skew       -- the Double Motor shifted up to +/-60 ms against the MPU
  * sensor dropout   -- 25 % of crops lose the Double Motor, 15 % the MPU
                        (never both), so one network works with either or both
Loss = density MSE + COUNT_WEIGHT * (predicted - labelled steps in the crop)^2.
"""

from __future__ import annotations

import numpy as np
import torch

from . import config as C
from . import fuse as F
from . import labels as L
from .net import StepNet, predict_density

BATCH, ITERS_PER_EPOCH = 24, 40


def prepare(take):
    """Grid a take, label its steps from the hand count, build the target density."""
    grid = F.take_grid(take)
    streams = F.on_grid(take, grid)
    idx, info = L.label_steps(streams, take["meta"]["steps"])
    return {"name": take["name"], "activity": take["meta"].get("activity", "?"),
            "steps": int(take["meta"]["steps"]), "streams": streams,
            "y": L.density(idx, len(grid)), "step_idx": idx, "label_info": info}


def _rotation(rng, max_deg):
    from .synth import _rotation as rot
    return rot(rng, max_deg)


# Augmentation strengths; set any to 0 to switch that augmentation off.
AUG = {"rotate_deg": 20, "gain": 0.10, "shift": 3, "drop_lego": 0.25, "drop_mpu": 0.15}


def augment(streams, rng, aug=None):
    aug = AUG if aug is None else aug
    out = {}
    for s, x in streams.items():
        x = x.copy()
        if not np.isnan(x).all():
            if aug["rotate_deg"]:
                R = _rotation(rng, aug["rotate_deg"])
                x[:, 0:3] = x[:, 0:3] @ R.T
                x[:, 3:6] = x[:, 3:6] @ R.T
            if aug["gain"]:
                x *= rng.uniform(1 - aug["gain"], 1 + aug["gain"])
                x[:, 0:3] += rng.normal(0, 0.01, x[:, 0:3].shape)
                x[:, 3:6] += rng.normal(0, 1.0, x[:, 3:6].shape)
        out[s] = x
    shift = int(rng.integers(-aug["shift"], aug["shift"] + 1)) if aug["shift"] else 0
    if shift:
        lego = np.full_like(out["lego"], np.nan)
        if shift > 0:
            lego[shift:] = out["lego"][:-shift]
        else:
            lego[:shift] = out["lego"][-shift:]
        out["lego"] = lego
    have = {s for s, x in out.items() if not np.isnan(x).all()}
    r = rng.random()
    drop = "lego" if r < aug["drop_lego"] else "mpu" if r < aug["drop_lego"] + aug["drop_mpu"] else None
    if drop and len(have - {drop}) > 0:
        out[drop] = np.full_like(out[drop], np.nan)
    return out


def _crop(item, W, rng):
    T = len(item["y"])
    if T >= W:
        a = int(rng.integers(0, T - W + 1))
        return {s: x[a:a + W] for s, x in item["streams"].items()}, item["y"][a:a + W], np.ones(W, np.float32)
    pad = W - T
    streams = {s: np.vstack([x, np.full((pad, x.shape[1]), np.nan)]) for s, x in item["streams"].items()}
    return streams, np.concatenate([item["y"], np.zeros(pad, np.float32)]), \
        np.concatenate([np.ones(T, np.float32), np.zeros(pad, np.float32)])


def fit_norm(items):
    return F.fit_norm([F.build(it["streams"]) for it in items])


def pick_device(name="auto"):
    """'auto' = the Mac's GPU when available: ~20x faster than CPU for these dilated convs."""
    if name != "auto":
        return name
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def train(items, norm, epochs=40, device="auto", seed=0, log=print):
    device = pick_device(device)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    W = int(C.WINDOW_S * C.FS)
    model = StepNet(len(F.channel_names())).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * ITERS_PER_EPOCH)
    weights = np.array([len(it["y"]) for it in items], float)
    weights /= weights.sum()
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for _ in range(ITERS_PER_EPOCH):
            Xs, ys, ms = [], [], []
            for _ in range(BATCH):
                it = items[rng.choice(len(items), p=weights)]
                streams, y, m = _crop(it, W, rng)
                Xs.append(F.normalize(F.build(augment(streams, rng)), norm).T)
                ys.append(y)
                ms.append(m)
            x = torch.from_numpy(np.stack(Xs)).to(device)
            y = torch.from_numpy(np.stack(ys)).to(device)
            m = torch.from_numpy(np.stack(ms)).to(device)
            pred = model(x) * m
            dens = (((pred - y) * m) ** 2).sum() / m.sum() * 100
            count = ((pred.sum(1) - (y * m).sum(1)) ** 2).mean()
            loss = dens + C.COUNT_WEIGHT * count
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        if log and (epoch % 10 == 9 or epoch == epochs - 1):
            log(f"    epoch {epoch + 1:3d}/{epochs}  loss {total / ITERS_PER_EPOCH:.3f}")
    return model.cpu().eval()


def count(model, streams, norm):
    """Total steps in a gridded take: the sum of the predicted density."""
    return float(predict_density(model, F.normalize(F.build(streams), norm)).sum())


# ---------------- classic baseline ----------------
def baseline_count(streams, height):
    """Classic counter: peaks of the impact signal (in g, not rescaled) above `height`,
    at least MIN_STEP_GAP_S apart."""
    from scipy.signal import find_peaks
    sig = L.impact_signal(streams, rescale=False)
    return len(find_peaks(sig, height=height, distance=max(1, int(C.MIN_STEP_GAP_S * C.FS)))[0])


def baseline_fit(items):
    """The impact threshold with the smallest total count error on these takes."""
    grid = np.linspace(0.01, 1.0, 100)                    # g
    err = [sum(abs(baseline_count(it["streams"], h) - it["steps"]) for it in items) for h in grid]
    return float(grid[int(np.argmin(err))])
