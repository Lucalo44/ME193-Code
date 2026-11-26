import threading
import time

import config
from dance.compiler import ScoreError, compile_score, normalize
from dance.conductor import Conductor
from hal.performers import CarPerformer, SimLegoPerformer
from hal.sim_hal import SimHal
from skills.matrix import Matrix

SCORE = {"bpm": 100, "sections": [
    {"bars": 2, "mode": "mirror", "car": "shimmy", "lego": "wave"},
    {"bars": 1, "mode": "call_response", "car": "spin", "lego": "clap"},
    {"bars": 1, "mode": "unison", "car": "bow", "lego": "bow"}]}


def test_compile_durations_and_count_in():
    c = compile_score(SCORE)
    assert c["beats"] == 16
    spb = 0.6
    assert abs(c["seconds"] - (16 + config.COUNT_IN_BEATS) * spb) < 1e-6
    digits = [k["params"]["digit"] for k in c["timelines"]["car"] if "digit" in k["params"]]
    assert digits == ["4", "3", "2", "1"]
    assert c["timelines"]["car"][-1]["beat"] == 16


def test_mirror_swaps_lego_arms():
    a = compile_score({"bpm": 100, "sections": [{"bars": 1, "mode": "unison", "car": "freeze", "lego": "wave"}]})
    b = compile_score({"bpm": 100, "sections": [{"bars": 1, "mode": "mirror", "car": "freeze", "lego": "wave"}]})
    pa = next(k for k in a["timelines"]["lego"] if k["action"] == "pose")["params"]
    pb = next(k for k in b["timelines"]["lego"] if k["action"] == "pose")["params"]
    assert pa["a"] == pb["b"] and pa["b"] == pb["a"]


def test_call_response_alternates():
    c = compile_score({"bpm": 120, "sections": [{"bars": 1, "mode": "call_response", "car": "spin", "lego": "clap"}]})
    lego_moves = [(k["beat"], k["move"]) for k in c["timelines"]["lego"]]
    assert (0, "freeze") in lego_moves and any(b >= 2 and m == "clap" for b, m in lego_moves)


def test_sixty_second_cap():
    n = normalize({"bpm": 60, "sections": [{"bars": 4, "mode": "unison", "car": "spin", "lego": "wave"}] * 4})
    beats = sum(s["bars"] * 4 for s in n["sections"])
    assert (beats + config.COUNT_IN_BEATS) * 1.0 <= config.MAX_DANCE_S


def test_bad_move_rejected():
    try:
        compile_score({"bpm": 100, "sections": [{"bars": 1, "mode": "unison", "car": "moonwalk", "lego": "wave"}]})
        assert False
    except ScoreError:
        pass


def test_conductor_sim_duet_in_sync():
    hal = SimHal(latency_ms=2)
    perf = {"car": CarPerformer(hal, Matrix(hal)), "lego": SimLegoPerformer(ble_ms=20, jitter_ms=5, motor_ms=10)}
    c = Conductor(perf)
    r = c.perform({"bpm": 160, "sections": [{"bars": 1, "mode": "unison", "car": "shimmy", "lego": "pump"}]})
    assert r["completed"]
    assert r["max_sync_error_ms"] < 80
    assert hal.left == 0 and hal.right == 0


def test_conductor_stop_mid_dance():
    hal = SimHal(latency_ms=1)
    c = Conductor({"car": CarPerformer(hal, Matrix(hal)), "lego": SimLegoPerformer()})
    cancel = threading.Event()
    threading.Timer(1.2, cancel.set).start()
    t0 = time.time()
    r = c.perform({"bpm": 100, "sections": [{"bars": 4, "mode": "unison", "car": "spin", "lego": "windmill"}]},
                  cancel=cancel)
    assert not r["completed"]
    assert time.time() - t0 < 1.6
    assert hal.left == 0



def test_figure_frames_are_valid_and_follow_the_lanes():
    from dance.figure import figure_timeline
    comp = compile_score({"bpm": 120, "sections": [{"bars": 1, "mode": "unison", "car": "shimmy", "lego": "pump"},
                                                    {"bars": 1, "mode": "unison", "car": "bow", "lego": "bow"}]})
    tl = figure_timeline(comp)
    assert all(len(k["params"]["frame"]) == 104 and set(k["params"]["frame"]) <= {0, 7} for k in tl)
    assert [k["beat"] for k in tl[:4]] == [-4, -3, -2, -1]              # 4-3-2-1 count-in
    assert len({tuple(k["params"]["frame"]) for k in tl}) >= 5           # it actually moves
    assert tl == sorted(tl, key=lambda k: k["t"])


def test_matrix_only_dance_moves_no_motors():
    hal = SimHal(latency_ms=1)
    m = Matrix(hal)
    from hal.performers import MatrixFigurePerformer
    c = Conductor({"car": CarPerformer(hal, m), "lego": SimLegoPerformer(), "figure": MatrixFigurePerformer(m)})
    r = c.perform({"bpm": 160, "sections": [{"bars": 1, "mode": "unison", "car": "spin", "lego": "windmill"}]},
                  where="matrix")
    assert r["completed"] and r["where"] == "matrix"
    assert hal.drive_log == []                                          # wheels never touched
    assert r["performers"].keys() == {"figure"}


def test_where_all_runs_robots_and_figure():
    hal = SimHal(latency_ms=1)
    m = Matrix(hal)
    from hal.performers import MatrixFigurePerformer
    c = Conductor({"car": CarPerformer(hal, m), "lego": SimLegoPerformer(), "figure": MatrixFigurePerformer(m)})
    r = c.perform({"bpm": 160, "sections": [{"bars": 1, "mode": "unison", "car": "shimmy", "lego": "wave"}]},
                  where="all")
    assert r["performers"].keys() == {"car", "lego", "figure"} and hal.drive_log
