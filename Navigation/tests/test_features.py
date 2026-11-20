"""Quick consistency checks:  python3 -m pytest tests/   (from Navigation/)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nav.features import feature_names, observation  # noqa: E402
from nav.imu import Heading, parse_drain  # noqa: E402
from nav.odometry import Odometry  # noqa: E402


def test_observation_matches_names():
    odo = Odometry()
    st = odo.on_step(0.5, 0.0)
    obs = observation(st, np.array([0.1, 0.8, 0.1, 0.0]), 300)
    assert obs.shape == (len(feature_names()),)
    assert np.all(np.isfinite(obs))


def test_parse_drain():
    rows = parse_drain("1000,0,0,1,0,0,0;1020,0.1,0,0.99,1,2,3;garbage;")
    assert rows.shape == (2, 7)
    assert rows[1, 0] == 1.02


def test_heading_integrates_pure_yaw():
    th = np.radians(30)                     # tilted device, rotating about world vertical
    R = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
    g, w = R.T @ np.array([0, 0, 1.0]), R.T @ np.array([0, 0, 30.0])
    s = np.column_stack([np.tile(g, (150, 1)), np.tile(w, (150, 1))])
    assert abs(Heading(50).update(s) - 90.0) < 1.0


def test_turn_detection():
    odo = Odometry()
    for i in range(30):
        odo.on_step(i * 0.5, 0.0 if i < 15 else 90.0)
    assert [s for _, s in odo.turns] == [1]
