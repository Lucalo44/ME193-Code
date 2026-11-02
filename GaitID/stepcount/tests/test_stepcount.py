import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stepcount import config as C
from stepcount import fuse as F
from stepcount import labels as L
from stepcount import synth
from stepcount import training as T
from stepcount.net import StepNet, predict_density


def take(activity, seed=0, seconds=30, tmp=None, drop=None):
    meta, data, steps = synth.make_take(activity, seconds=seconds, seed=seed)
    if drop:
        data[drop] = None
    folder = os.path.join(tmp, f"{activity}_{seed}")
    F.save_take(folder, meta, mpu=data["mpu"], lego=data["lego"])
    return F.load_take(folder), steps


def test_roundtrip_and_grid(tmp_path):
    t, _ = take("walk", tmp=str(tmp_path))
    grid = F.take_grid(t)
    assert np.allclose(np.diff(grid), 1 / C.FS)
    X = F.build(F.on_grid(t, grid))
    assert X.shape == (len(grid), len(F.channel_names()))


def test_missing_sensor_is_zero_with_present_flag(tmp_path):
    t, _ = take("walk", tmp=str(tmp_path), drop="lego")
    items = [T.prepare(t)]
    Z = F.normalize(F.build(items[0]["streams"]), T.fit_norm(items))
    names = F.channel_names()
    lego = [i for i, n in enumerate(names) if n.startswith("lego_")]
    assert np.all(Z[:, lego] == 0)
    assert np.all(Z[:, names.index("mpu_present")] == 1)


@pytest.mark.parametrize("activity", ["walk", "run"])
def test_labels_find_the_true_steps(tmp_path, activity):
    t, true = take(activity, seed=3, tmp=str(tmp_path))
    item = T.prepare(t)
    assert len(item["step_idx"]) == len(true)
    # labels sit at mid-swing: each one should come a steady 0.1-0.4 s before the strike that follows it
    found = F.take_grid(t)[0] + item["step_idx"] / C.FS
    nxt = np.searchsorted(true, found)
    d = true[nxt[nxt < len(true)]] - found[nxt < len(true)]
    assert len(set(nxt)) >= len(true) - 1                       # one label per step
    assert 0.1 < np.median(d) < 0.4 and np.percentile(d, 75) - np.percentile(d, 25) < 0.06


def test_labels_fill_in_the_other_foot(tmp_path):
    t, true = take("walk", seed=4, tmp=str(tmp_path), drop="lego")      # one ankle only
    item = T.prepare(t)
    assert len(item["step_idx"]) == len(true)


def test_density_sums_to_steps():
    y = L.density(np.array([10, 50, 99]), 100)
    assert y[:90].sum() == pytest.approx(2.0, abs=0.01)                    # last step half cut off
    assert L.density(np.array([30, 60]), 100).sum() == pytest.approx(2.0, abs=1e-4)


def test_net_shape_and_nonnegative():
    net = StepNet(len(F.channel_names()))
    d = predict_density(net, np.random.default_rng(0).normal(size=(300, len(F.channel_names()))).astype(np.float32))
    assert d.shape == (300,) and (d >= 0).all()
    assert net.receptive_field == 1 + (C.KERNEL - 1) * sum(C.DILATIONS)


def test_short_training_learns_to_count(tmp_path):
    items = [T.prepare(take(a, seed=s, seconds=24, tmp=str(tmp_path))[0])
             for s, a in enumerate(["walk", "run", "walk", "still", "run", "bike"])]
    norm = T.fit_norm(items)
    model = T.train(items, norm, epochs=6, log=None)
    test = T.prepare(take("walk", seed=40, seconds=24, tmp=str(tmp_path))[0])
    assert abs(T.count(model, test["streams"], norm) - test["steps"]) < 0.25 * test["steps"]


def test_board_numpy_net_matches_pytorch(tmp_path):
    """The UNO Q's numpy StepNet (stepcount/board.py) must give PyTorch's density on MPU-only data."""
    import torch
    from stepcount.board import BoardStepNet, StepCounter, _running_mean
    from scipy.ndimage import uniform_filter1d
    x = np.random.default_rng(0).normal(size=(200, 3))
    assert np.allclose(_running_mean(x), uniform_filter1d(x, C.FS, axis=0, mode="nearest"))

    torch.manual_seed(0)
    net = StepNet(len(F.channel_names())).eval()
    t, _ = take("walk", seed=7, tmp=str(tmp_path), drop="lego")
    items = [T.prepare(t)]
    norm = T.fit_norm(items)
    path = str(tmp_path / "board.npz")
    np.savez(path, kernel=C.KERNEL, dilations=np.array(C.DILATIONS), mean=np.array(norm["mean"]),
             std=np.array(norm["std"]), channels=np.array(F.channel_names()),
             **{k: v.numpy() for k, v in net.state_dict().items()})
    mpu = items[0]["streams"]["mpu"]
    mpu = mpu[~np.isnan(mpu).any(axis=1)]
    ref = predict_density(net, F.normalize(F.build(items[0]["streams"]), norm))
    board = BoardStepNet(path)
    got = board.density(board.features(mpu))
    assert np.allclose(got, ref[:len(got)], atol=1e-4)
    counter = StepCounter(board)
    for i, row in enumerate(mpu):
        counter.add(row)
        if i % 50 == 49:
            counter.update()
    assert counter.update(flush=True) == pytest.approx(got.sum(), rel=0.02)
