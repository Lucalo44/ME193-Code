from perception.budget import Budget, mode_for_state


def test_keyword_spotting_is_never_throttled_and_pose_drops_while_planning():
    b = Budget()
    for mode in ("PLANNING", "DANCING", "MIRRORING", "IDLE"):
        b.set_mode(mode)
        assert b.max_fps("kws") is None                 # stop must never wait
    b.set_mode("PLANNING")
    assert b.max_fps("pose") == 2 and b.interval_s("pose", 0.1) == 0.5
    b.set_mode("MIRRORING")
    assert b.interval_s("pose", 0.1) == 0.1


def test_modes_follow_the_app_state_and_notify():
    b, seen = Budget(), []
    b.on_change(seen.append)
    for st in ("PLANNING", "EXECUTING", "DANCING", "IDLE"):
        b.set_mode(mode_for_state(st))
    assert seen == ["PLANNING", "IDLE", "DANCING", "IDLE"]
