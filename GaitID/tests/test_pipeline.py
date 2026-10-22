import subprocess
import sys

import numpy as np
import pytest

from gaitid import config as C
from gaitid import features as F
from gaitid import synth
from gaitid.model import Model


def window(label, seed=0):
    return synth.make_recording(label, 4, seed)[: C.WINDOW]


def test_feature_shapes():
    w = window("walk")
    assert F.peak_feature(w, "az").shape == (1,)
    assert F.full_features(w).shape == (len(F.full_feature_names()),)


def test_peak_ignores_gravity_offset():
    w = window("still")
    shifted = w.copy()
    shifted[:, 2] += 5
    assert F.peak_feature(w, "az") == pytest.approx(F.peak_feature(shifted, "az"))


def test_peak_orders_activities():
    p = {c: F.peak_feature(window(c), "az")[0] for c in C.CLASSES}
    assert p["still"] < p["bike"] < p["walk"] < p["run"]


def test_dominant_frequency_is_step_rate():
    names = F.full_feature_names()
    f = F.full_features(synth.make_recording("walk", 4, 1)[: C.WINDOW])
    assert 0.8 <= f[names.index("amag_freq")] <= 2.2


def test_train_and_predict(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for label in C.CLASSES:
        for k in range(4):
            synth.write_csv(str(data / f"{label}_{k}.csv"), synth.make_recording(label, 30, k))
    out = tmp_path / "models"
    r = subprocess.run([sys.executable, "tools/train.py", "--data", str(data), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    m1 = Model.load(str(out / "model1_peak.json"))
    m2 = Model.load(str(out / "model2_full.json"))
    assert m1.spec["cv_accuracy"] > 0.7
    assert m2.spec["cv_accuracy"] > 0.9
    for label in C.CLASSES:
        assert m2.predict(window(label, 99)) == label
