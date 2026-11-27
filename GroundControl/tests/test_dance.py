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
    assert all(len(k["params"]["frame"]) == 104 and set(k["params"]["frame"]) <= set(range(8)) for k in tl)
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


# ---- style library, rhythms and two dancers (dance/styles.py, rhythm.py, figure.py)

def test_style_library_matches_names_people_use():
    from dance.styles import match
    cases = {"waltz": "waltz", "a Viennese waltz": "waltz", "salsa dance": "salsa", "hip-hop": "hip hop",
             "b-boy": "breakdance", "lindy hop": "swing", "cha cha cha": "cha-cha", "the robot": "robot",
             "traditional west african dance": "west african", "ballerina": "ballet"}
    for name, key in cases.items():
        assert (match(name) or {}).get("key") == key, name
    assert match("mazurka") is None and match("") is None       # not in the library: the choreographer


def test_every_profile_compiles_at_every_length_and_its_poses_exist():
    import config
    from dance.figure import figure_timeline, pose
    from dance.styles import PROFILES, match, score_for
    for key, p in PROFILES.items():
        for s in p["sections"]:
            for cyc in s.get("fig", {}).values():
                [pose(x) for x in cyc]                           # raises on a typo
        for bars in range(1, 9):
            sc = score_for(match(key), bars)
            comp = compile_score(sc)
            assert comp["beats"] == bars * p["meter"] and comp["meter"] == p["meter"], (key, bars)
            assert comp["seconds"] <= config.MAX_DANCE_S
            for dancers in (1, 2):
                tl = figure_timeline({**comp, "dancers": dancers})
                assert all(len(k["params"]["frame"]) == 104 for k in tl)
    assert score_for(match("salsa"), 4, bpm=140)["bpm"] == 140   # the user's tempo wins
    assert score_for(match("salsa"), 4)["bpm"] == 95             # otherwise the style's own


def test_waltz_counts_in_three_and_lights_every_third_beat():
    from dance.figure import figure_timeline
    from dance.styles import match, score_for
    comp = compile_score(score_for(match("waltz"), 2))
    assert comp["count_in"] == 3 and comp["beats"] == 6
    downbeats = [k["beat"] for k in comp["timelines"]["car"] if k["move"] == "downbeat" and k["params"].get("icon")]
    assert downbeats == [0, 3]
    tl = figure_timeline(comp)
    corner = lambda k: k["params"]["frame"][5 * 13]             # the tall part of the beat light
    assert sorted({k["beat"] for k in tl if corner(k) == 7}) == [0, 3]


def test_rhythm_patterns_on_the_16th_grid():
    from dance.rhythm import is_hit
    assert [is_hit("x.x.", i) for i in range(4)] == [True, False, True, False]
    assert all(is_hit("xxxx", i) for i in range(16))                       # 16th notes
    tango = "x.......x.......x...x...x......."                             # a 2-bar phrase
    assert [i for i in range(32) if is_hit(tango, i)] == [0, 8, 16, 20, 24]
    assert [i for i in range(12) if is_hit("x.x.x.x.x.x.xxxx", i, meter=3)] == [0, 2, 4, 6, 8, 10]  # cut to 3/4
    assert not any(is_hit("", i) for i in range(16))


def test_figure_flashes_on_hits_and_changes_pose_only_on_the_rhythm():
    from dance.figure import figure_timeline
    sc = {"bpm": 100, "sections": [{"bars": 1, "mode": "unison", "car": "step", "lego": "wave",
                                    "rhythm": {"car": "x.......", "lego": "x.......x.x.x.xx"}}]}
    comp = compile_score(sc)
    tl = [k for k in figure_timeline(comp) if 0 <= k["beat"] < comp["beats"]]
    bright = sorted({k["beat"] for k in tl if max(k["params"]["frame"][1:12]) == 7 and
                     max(v for i, v in enumerate(k["params"]["frame"]) if i % 13 not in (0, 12)) == 7})
    assert bright == [0, 2, 2.5, 3, 3.5, 3.75]                             # the lego rhythm, 16ths at the end


def test_two_dancers_stay_in_their_halves_and_mirror_faces_each_other():
    from dance.figure import figure_timeline
    sc = {"bpm": 100, "dancers": 2, "sections": [{"bars": 1, "mode": "mirror", "car": "step", "lego": "point"}]}
    comp = compile_score(sc)
    assert comp["dancers"] == 2
    tl = [k for k in figure_timeline(comp) if 0 <= k["beat"] < comp["beats"]]
    for k in tl:
        f = k["params"]["frame"]
        assert all(f[r * 13 + 6] == 0 for r in range(8))                    # a gap between the partners
        assert any(f[r * 13 + c] for r in range(8) for c in range(1, 6))    # left dancer
        assert any(f[r * 13 + c] for r in range(8) for c in range(7, 12))   # right dancer
    # point is one-armed: mirrored, the right dancer points left, toward its partner
    f = tl[0]["params"]["frame"]
    assert f[2 * 13 + 7] and not f[2 * 13 + 11]


def test_dancer_count_comes_from_the_words_then_the_style():
    from dance.styles import dancers_for
    assert dancers_for("make two dancers salsa on the matrix") == 2
    assert dancers_for("two stick figures dance together on the matrix at 120 bpm") == 2
    assert dancers_for("the robot spins while the lego claps, on the matrix") == 2
    assert dancers_for("make a stick figure waltz", default=2) == 1            # asked for one
    assert dancers_for("dance at 120 bpm on the matrix for 2 bars") == 1       # "2 bars" is not two dancers
    assert dancers_for("waltz on the matrix", default=2) == 2                  # partner dance
    assert dancers_for("dance on the robots and the matrix", where="all") == 2  # one per robot


def test_choreographer_feel_becomes_rhythms():
    from dance.styles import apply_feel
    sc = {"bpm": 100, "sections": [{"bars": 2, "mode": "unison", "car": "step", "lego": "freeze"}]}
    apply_feel(sc, "driving")
    assert sc["sections"][0]["rhythm"] == {"car": "x.x.", "lego": ""}         # a freeze stays still
    assert compile_score(sc)["blocks"][0]["rhythm"] == "x.x."


def test_style_and_dancers_read_from_the_words_when_the_plan_is_generic():
    from dance.styles import dancers_for, match_text
    t = "loop two breakdancers on the matrix at 120bpm"
    assert dancers_for(t) == 2 and match_text(t)["key"] == "breakdance"
    assert match_text("make two tango dancers on the matrix")["key"] == "tango"
    assert match_text("the robot spins while the lego claps") is None     # "robot" is the car here
    assert match_text("swing your arms and dance") is None
    assert match_text("do the robot on the matrix")["key"] == "robot"
    assert match_text("dance 4 bars at 100 bpm") is None


def test_the_head_is_the_brightest_part_and_hands_never_match_it():
    from dance.figure import ARMS, render
    for arms in ARMS:
        f = render(arms=arms)
        head = [f[r * 13 + c] for r in (0, 1) for c in (5, 6, 7)]
        others = [v for i, v in enumerate(f) if not (i // 13 in (0, 1) and i % 13 in (5, 6, 7))]
        assert min(head) == 7 and max(others) < 7, arms                # a raised hand is dimmer
