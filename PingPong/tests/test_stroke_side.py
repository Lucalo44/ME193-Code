from game.rules import required_stroke, stroke_ok
from vision.pose import StrokeJudge, judge_predictions


def test_right_handed_centerline_rule():
    assert required_stroke(0.4, "right") == "forehand"
    assert required_stroke(-0.4, "right") == "backhand"


def test_left_handed_is_inverted():
    assert required_stroke(0.4, "left") == "backhand"
    assert required_stroke(-0.4, "left") == "forehand"


def test_deadband_accepts_either():
    assert required_stroke(0.03, "right", deadband=0.05) == "either"
    assert required_stroke(-0.05, "right", deadband=0.05) == "either"
    assert required_stroke(0.06, "right", deadband=0.05) == "forehand"


def test_stroke_ok():
    assert stroke_ok("forehand", "forehand")
    assert not stroke_ok("forehand", "backhand")
    assert stroke_ok("either", "backhand")
    assert not stroke_ok("either", None)       # ready-only / no pose is always wrong
    assert not stroke_ok("forehand", "ready")


def test_judge_ignores_ready_and_weights_confidence():
    preds = [("ready", 1.0), ("forehand", 0.4), ("forehand", 0.4), ("backhand", 1.0)]
    stroke, conf = judge_predictions(preds)
    assert stroke == "backhand" and abs(conf - 1.0 / 1.8) < 1e-9


def test_judge_ready_only_is_none():
    assert judge_predictions([("ready", 1.0), (None, 0.0)]) == (None, 0.0)


def test_stroke_judge_uses_time_window():
    j = StrokeJudge(history_s=2.0)
    for i in range(20):
        t = 10.0 + i * 0.033
        j.add(t, "forehand" if t < 10.3 else "backhand", 1.0)
    assert j.judge(10.1, window=0.1)[0] == "forehand"
    assert j.judge(10.55, window=0.1)[0] == "backhand"
    assert j.judge(20.0, window=0.1)[0] is None


# --- decide_stroke: WRONG STROKE only when the evidence is sure --------------
from game.rules import decide_stroke  # noqa: E402


def test_camera_sure_wrong_is_wrong():
    assert decide_stroke("backhand", ("forehand", 0.9))[0] == "forehand"


def test_unsure_or_missing_camera_gets_benefit_of_the_doubt():
    assert decide_stroke("backhand", ("forehand", 0.6))[0] == "backhand"     # unsure vote
    assert decide_stroke("backhand", (None, 0.0))[0] == "backhand"           # only "ready" / no pose


def test_paddle_decides_when_camera_unsure():
    assert decide_stroke("backhand", (None, 0.0), ("forehand", 0.9)) == ("forehand", 0.9, "paddle")
    assert decide_stroke("backhand", (None, 0.0), ("forehand", 0.6))[0] == "backhand"


def test_sources_disagreeing_confidently_is_benefit_of_the_doubt():
    assert decide_stroke("backhand", ("forehand", 0.95), ("backhand", 0.9))[0] == "backhand"
    assert decide_stroke("backhand", ("backhand", 0.9), ("forehand", 0.95))[0] == "backhand"


def test_both_sure_wrong_is_wrong_and_either_side_always_ok():
    assert decide_stroke("forehand", ("backhand", 0.8), ("backhand", 0.8)) == ("backhand", 0.8, "camera")
    assert decide_stroke("either", ("backhand", 0.99), ("forehand", 0.99))[2] == "either side"
