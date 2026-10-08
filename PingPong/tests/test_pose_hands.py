"""Shared pose data between left- and right-handed players (POSE_HAND_MODE)."""
import numpy as np

from vision.pose import FEATURE_DIM, PoseClassifier, mirror_features

rng = np.random.default_rng(0)
FH, BH = rng.normal(size=FEATURE_DIM), rng.normal(size=FEATURE_DIM)   # two distinct "poses"


def noisy(f):
    return f + rng.normal(scale=0.01, size=FEATURE_DIM)


def test_mirror_twice_is_identity_and_swaps_sides():
    f = np.arange(FEATURE_DIM, dtype=float)
    m = mirror_features(f)
    assert np.allclose(mirror_features(m), f)
    assert m[0] == -f[0] and m[1] == f[1]                 # nose: x flipped, y kept
    assert np.allclose(m[3:6], f[6:9] * [-1, 1, 1])       # left shoulder <- mirrored right shoulder


def test_mirror_mode_lets_a_left_hander_use_right_handed_data():
    c = PoseClassifier(k=3, hand="right", mode="mirror")
    for _ in range(5):
        c.add_sample(noisy(FH), "forehand", player="sam")
        c.add_sample(noisy(BH), "backhand", player="sam")
    c.set_hand("left")
    # A left-hander's forehand is the right-hander's forehand seen in a mirror.
    assert c.predict(mirror_features(FH))[0] == "forehand"
    assert c.predict(mirror_features(BH))[0] == "backhand"


def test_left_handed_data_is_unchanged_for_its_own_player():
    c = PoseClassifier(k=3, hand="left", mode="mirror")
    for _ in range(5):
        c.add_sample(noisy(FH), "forehand")
        c.add_sample(noisy(BH), "backhand")
    assert c.predict(FH)[0] == "forehand" and c.predict(BH)[0] == "backhand"


def test_separate_mode_only_uses_the_players_hand():
    c = PoseClassifier(k=3, hand="right", mode="separate")
    for _ in range(3):
        c.add_sample(noisy(FH), "forehand")
        c.add_sample(noisy(BH), "backhand")
    assert c.trained
    c.set_hand("left")
    assert not c.trained and c.predict(FH) == (None, 0.0)


def test_save_load_keeps_player_and_hand(tmp_path):
    path = str(tmp_path / "pose.npz")
    c = PoseClassifier(hand="right")
    c.add_sample(FH, "forehand", player="sam")
    c.add_sample(BH, "backhand", hand="left", player="luca")
    c.save(path)
    d = PoseClassifier(hand="right")
    assert d.load(path)
    assert d.hands == ["right", "left"] and d.players == ["sam", "luca"]


def test_old_files_without_hand_take_the_legacy_hand(tmp_path):
    path = str(tmp_path / "old.npz")
    np.savez(path, X=np.stack([FH, BH]), y=np.array(["forehand", "backhand"]), k=np.array([5]))
    c = PoseClassifier(hand="right")
    assert c.load(path, legacy_hand="left")
    assert c.hands == ["left", "left"] and c.players == ["original", "original"]


def test_clear_removes_only_one_players_samples():
    c = PoseClassifier(hand="right")
    c.add_sample(FH, "forehand", player="sam")
    c.add_sample(BH, "backhand", player="luca")
    assert c.remove_player("sam") == 1
    assert c.players == ["luca"] and c.y == ["backhand"]
