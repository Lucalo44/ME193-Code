"""Game recording: every swing gets an outcome, shots get results, misses and
disputes are logged -- the raw material for tools/game_report.py."""

import json
import os

import config as C
from game import state_machine as SM
from game.loop import Game
from game.recorder import GameRecorder
from hardware.sim_paddle import SimPaddle
from hardware.swing import Calibration, ImuSample
from tests.test_game_loop import FakeClock, advance, swing_at, wait_incoming


def recorded_game(tmp_path):
    clock = FakeClock()
    rec = GameRecorder("tester", root=str(tmp_path))
    game = Game(SimPaddle(Calibration()), None, seed=1, clock=clock, recorder=rec)
    return game, clock, rec


def read(rec, name):
    with open(os.path.join(rec.dir, f"{name}.jsonl")) as f:
        return [json.loads(line) for line in f]


def test_hit_and_miss_outcomes_are_recorded(tmp_path):
    game, clock, rec = recorded_game(tmp_path)
    game.start_game("medium")
    assert rec.dir and os.path.isdir(rec.dir)
    inc = wait_incoming(game, clock)
    swing_at(game, inc.t - inc.early_s - 0.3)                 # whiff: too early
    advance(game, clock, inc.t - game.sim_t - 0.05)
    swing_at(game, inc.t)                                      # on time
    advance(game, clock, 3.0, until=lambda: game.sm.phase == SM.POINT_OVER)
    rec.flush()
    swings = read(rec, "swings")
    assert [s["outcome"] for s in swings][:2] == ["early", "hit"]
    assert swings[1]["ball"]["required"] in ("forehand", "backhand")
    events = read(rec, "events")
    shots = [e for e in events if e["name"] == "player_shot"]
    assert shots and shots[0]["swing_id"] == swings[1]["id"]
    assert any(e["name"] == "shot_result" and e["swing_id"] == swings[1]["id"] for e in events)


def test_no_swing_is_logged_with_the_ball_and_can_be_disputed(tmp_path):
    game, clock, rec = recorded_game(tmp_path)
    game.start_game("medium")
    wait_incoming(game, clock)
    advance(game, clock, 3.0, until=lambda: game.sm.phase == SM.POINT_OVER)
    game.command({"type": "key", "key": "x", "down": True})
    advance(game, clock, 0.05)
    events = read(rec, "events")
    missed = [e for e in events if e["name"] == "missed_ball"]
    assert missed and "arrival_t" in missed[0]
    disputes = [e for e in events if e["name"] == "dispute"]
    assert disputes and disputes[0]["reason"] == "NO SWING"


def test_streams_are_written_and_session_closes(tmp_path):
    game, clock, rec = recorded_game(tmp_path)
    game.start_game("slow")
    rec.on_sample(ImuSample(1.0, 0, 0, 0, 0, 0, 1000, 0, 0, 0))
    rec.on_frame({"type": "frame", "t": 1.0, "label": "ready", "conf": 0.9, "feat": [0.1] * 27})
    rec.close()
    with open(os.path.join(rec.dir, "imu.csv")) as f:
        assert len(f.readlines()) == 2
    with open(os.path.join(rec.dir, "pose.csv")) as f:
        lines = f.readlines()
    assert lines[0].startswith("t,label,conf,f0") and len(lines) == 2
    with open(os.path.join(rec.dir, "session.json")) as f:
        assert json.load(f)["counts"]["imu"] == 1


def test_nothing_is_written_before_a_game_starts(tmp_path):
    game, clock, rec = recorded_game(tmp_path)
    advance(game, clock, 0.5)
    assert rec.dir is None and not os.listdir(tmp_path)
    assert C.GAME_LOG_DIR == "game_logs"


def fake_session(tmp_path, monkeypatch, swings, samples=()):
    monkeypatch.setattr(C, "GAME_LOG_DIR", str(tmp_path))
    rec = GameRecorder("p1", meta={"hand": "left"})
    rec.open()
    for s in samples:
        rec.on_sample(s)
    for sw in swings:
        rec.swing(sw)
    rec.close()
    return rec


def swing_rec(i, required, camera, hand="left", outcome="hit"):
    return {"id": i, "t": float(i), "outcome": outcome, "handedness": hand, "strength": 0.5, "return_speed": 8.0,
            "topspin": 0.0, "sidespin": 0.0, "peak_accel_g": 3.0, "peak_gyro_dps": 500.0,
            "features": {"gz_int": 1.0}, "camera": camera, "ball": {"required": required, "dt": 0.0}}


def replay_as_recorded(monkeypatch):
    """Stand-in for re-detecting swings from imu.csv: one swing at each recorded time."""
    import tools.calibrate_swing as cs
    from hardware.swing import SwingEvent
    monkeypatch.setattr(cs, "replay_game_swings", lambda session: [
        SwingEvent(t_peak=sw["t"], strength=0.5, return_speed=8, topspin=0, sidespin=0, peak_accel_g=3,
                   peak_gyro_dps=500, features=sw["features"]) for sw in session.swings])


def test_only_reliably_labelled_game_swings_become_training_data(tmp_path, monkeypatch):
    from tools.calibrate_swing import game_sets_for
    replay_as_recorded(monkeypatch)
    fake_session(tmp_path, monkeypatch, [
        swing_rec(1, "forehand", ["forehand", 0.9]),               # used
        swing_rec(2, "backhand", ["backhand", 0.8]),               # used
        swing_rec(3, "forehand", ["forehand", 0.6]),               # camera unsure
        swing_rec(4, "forehand", ["backhand", 0.9]),               # camera disagrees
        swing_rec(5, "either", ["forehand", 0.9]),                 # no side rule
        swing_rec(6, "forehand", ["forehand", 0.9], hand="right"),  # lobby on the other hand
        swing_rec(7, "forehand", ["forehand", 0.9], outcome="late"),
    ])
    sets = game_sets_for("p1", "left")
    assert len(sets["game_fh"]) == 1 and len(sets["game_bh"]) == 1
    assert sets["game_fh"][0].weight == 1.0                    # the only (so newest) session


def test_game_swings_are_balanced_between_strokes(tmp_path, monkeypatch):
    from tools.calibrate_swing import GAME_CLASS_RATIO, game_sets_for
    replay_as_recorded(monkeypatch)
    fake_session(tmp_path, monkeypatch, [swing_rec(i, "forehand", ["forehand", 0.9]) for i in range(10)]
                 + [swing_rec(20 + i, "backhand", ["backhand", 0.9]) for i in range(2)])
    sets = game_sets_for("p1", "left")
    assert len(sets["game_bh"]) == 2 and len(sets["game_fh"]) == 2 * GAME_CLASS_RATIO
    assert sets["game_fh"][-1].t_peak == 9.0                    # the most recent are kept


def test_report_tells_a_missed_swing_from_no_swing(tmp_path, monkeypatch):
    from tools.game_report import load_sessions, motion_near
    still = [ImuSample(10 + i * 0.015, 0, 0, 0, 0, 0, 1000, 0, 0, 0) for i in range(60)]
    moving = [ImuSample(20 + i * 0.015, 0, 0, 0, 0, 0, 1000, 0, 0, 600 if 20 <= i < 30 else 0) for i in range(60)]
    fake_session(tmp_path, monkeypatch, [], still + moving)
    s = load_sessions("p1", None, None)[0]
    gyro_still, _ = motion_near(s, 10.45)
    gyro_moving, _ = motion_near(s, 20.45)
    assert gyro_still < s.cal.gyro_threshold <= gyro_moving


def test_game_swings_are_re_detected_from_the_raw_stream(tmp_path, monkeypatch):
    """With real paddle data, the training swing comes from replaying imu.csv."""
    from tests.synth import recording
    from tools.calibrate_swing import game_sets_for
    samples, centers = recording([{"peak_g": 4, "peak_dps": 600}], rest_s=1.0)
    fake_session(tmp_path, monkeypatch, [swing_rec(1, "forehand", ["forehand", 0.9])], samples)
    import tools.game_report as gr
    sess = gr.load_sessions("p1", None, None)[0]
    for sw in sess.swings:                       # point the record at the synthetic swing
        sw["t"] = centers[0]
    monkeypatch.setattr(gr, "load_sessions", lambda *a: [sess])
    sets = game_sets_for("p1", "left")
    assert len(sets["game_fh"]) == 1 and sets["game_fh"][0].features
