"""Pose arm tracking: body-relative arm points, paddle-arm choice, ready position,
and the separate vision process."""

import time
from types import SimpleNamespace

from vision.arm import ArmTracker, arm_points
from vision.process import VisionProcess


def pose(left_wrist=(0.62, 0.55), right_wrist=(0.38, 0.55), scale=1.0, shift=0.0):
    """33 landmarks in a mirrored image: the subject's shoulders at x 0.42 / 0.58."""
    lm = [SimpleNamespace(x=0.5, y=0.5, visibility=0.9) for _ in range(33)]
    def put(i, x, y):
        lm[i] = SimpleNamespace(x=0.5 + (x - 0.5) * scale + shift, y=0.5 + (y - 0.5) * scale, visibility=0.9)
    put(11, 0.58, 0.35); put(12, 0.42, 0.35)        # shoulders
    put(23, 0.56, 0.65); put(24, 0.44, 0.65)        # hips -> torso length 0.30
    put(13, 0.64, 0.45); put(14, 0.36, 0.45)        # elbows
    put(15, *left_wrist); put(16, *right_wrist)     # wrists
    return lm


def test_arm_points_are_body_relative():
    near = arm_points(pose(scale=1.0))
    far = arm_points(pose(scale=0.5, shift=0.1))   # smaller (further away) and off to the side
    for a, b in zip(near["arms"], far["arms"]):
        for k in ("shoulder", "elbow", "wrist"):
            assert abs(a[k][0] - b[k][0]) < 1e-3 and abs(a[k][1] - b[k][1]) < 1e-3
    # y is up: shoulders are one torso length above the hips.
    assert abs(near["arms"][0]["shoulder"][1] - 1.0) < 1e-3


def test_default_paddle_arm_is_on_the_handed_side_of_the_screen():
    tr = ArmTracker("right")
    tr.update(1.0, arm_points(pose()))
    i = tr.paddle_arm()
    assert tr.history[-1][1][i]["shoulder"][0] > 0          # mirrored view: right hand on screen right
    tr_left = ArmTracker("left")
    tr_left.update(1.0, arm_points(pose()))
    assert tr_left.paddle_arm() != i


def test_swings_teach_which_arm_holds_the_paddle():
    tr = ArmTracker("right")                                # default would pick the screen-right arm
    t = 0.0
    for swing in range(3):
        for k in range(12):                                 # the screen-LEFT wrist is the one swinging
            t += 1 / 30
            x = 0.38 - 0.1 * (k % 6)
            tr.update(t, arm_points(pose(right_wrist=(x, 0.55))))
        tr.on_swing(t - 0.15)
    i = tr.paddle_arm()
    assert tr.chosen_by == "swings" and tr.history[-1][1][i]["shoulder"][0] < 0


def test_hand_offset_from_ready_position():
    tr = ArmTracker("right")
    tr.update(1.0, arm_points(pose()))
    assert tr.state(1.0)["hand"] == [0.0, 0.0] and not tr.state(1.0)["zeroed"]
    tr.zero()
    tr.update(1.1, arm_points(pose(left_wrist=(0.62 + 0.15, 0.55 - 0.06))))   # hand out to the side and up
    st = tr.state(1.1)
    assert st["ok"] and st["zeroed"]
    assert abs(st["hand"][0] - 0.5) < 0.01 and abs(st["hand"][1] - 0.2) < 0.01   # in torso lengths
    assert not tr.state(2.0)["ok"]                                              # stale -> not tracking


def test_vision_process_runs_in_its_own_process():
    previews = []
    vp = VisionProcess(0, None, on_preview=previews.append, source="synthetic")
    vp.start()
    try:
        deadline = time.time() + 15
        while time.time() < deadline and not (vp.pose_detected and len(vp.arm.history) > 5):
            time.sleep(0.05)
        assert vp.camera.status == "ok" and vp.stroke_check_enabled
        assert vp.prediction == "forehand"
        assert vp.arm.state(time.monotonic())["ok"]
        assert vp.judge.judge(vp.arm.history[-1][0], window=0.2)[0] == "forehand"
        vp.tags_active = True                               # synthetic source confirms tag 1 after 8 frames
        deadline = time.time() + 5
        while time.time() < deadline and vp.confirmed_tags.empty():
            time.sleep(0.05)
        assert vp.confirmed_tags.get_nowait() == 1
        assert previews
    finally:
        vp.stop()
    assert not vp._proc.is_alive()
