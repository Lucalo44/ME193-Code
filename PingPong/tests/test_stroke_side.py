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
