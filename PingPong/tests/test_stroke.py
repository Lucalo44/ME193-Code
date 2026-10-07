"""Live (Wii-style) stroke tracking from the paddle's yaw."""

from hardware.stroke import StrokeTracker, backswing_from_recording

DT = 0.015


def run(tr, segments, t0=0.0, yaw0=0.0):
    """segments: [(target_yaw, seconds)] -- yaw moves linearly to each target."""
    t, yaw, out = t0, yaw0, []
    for target, secs in segments:
        steps = max(1, int(secs / DT))
        for i in range(steps):
            t += DT
            y = yaw + (target - yaw) * (i + 1) / steps
            out.append(tr.update(t, y))
        yaw = target
    return out, t, yaw


def test_slow_draw_back_tracks_the_backswing():
    tr = StrokeTracker(90, -70)
    states, *_ = run(tr, [(0, 0.3), (45, 0.5), (90, 0.5)])
    assert states[-1]["side"] == "forehand" and states[-1]["mode"] == "ready"
    assert abs(states[-1]["phase"] - 1.0) < 0.02
    half = states[len(states) // 2 + 10]
    assert 0.3 < half["phase"] < 0.8                     # in step with the draw-back


def test_fast_forward_swing_sweeps_through_contact_into_follow_through():
    tr = StrokeTracker(90, -70)
    states, t, yaw = run(tr, [(90, 0.6), (90, 0.2), (-40, 0.2)])   # 650 deg/s swing
    fwd = [s for s in states if s["mode"] == "forward"]
    assert fwd and all(s["side"] == "forehand" for s in fwd)
    assert min(s["phase"] for s in fwd) < -0.3           # carried on past the ball
    states, *_ = run(tr, [(-40, 0.3), (0, 0.6)], t, yaw)     # settles, back to ready
    assert states[-1]["mode"] == "ready"


def test_backhand_draws_back_the_other_way():
    tr = StrokeTracker(90, -70)
    states, *_ = run(tr, [(-70, 0.8)])
    assert states[-1]["side"] == "backhand" and abs(states[-1]["phase"] - 1.0) < 0.02
    states, *_ = run(tr, [(-70, 0.1), (30, 0.15)], 0.8, -70)
    assert any(s["mode"] == "forward" and s["side"] == "backhand" for s in states)


def test_quick_return_from_follow_through_is_not_a_backhand():
    tr = StrokeTracker(90, -70)
    # Forehand stroke ending at -25 deg, then a quick snap back to ready.
    states, *_ = run(tr, [(90, 0.6), (90, 0.1), (-25, 0.18), (-25, 0.5), (0, 0.08), (0, 0.4)])
    assert not any(s["mode"] == "forward" and s["side"] == "backhand" for s in states)


def test_rally_rhythm_back_to_back_forehands():
    tr = StrokeTracker(90, -70)
    states, *_ = run(tr, [(90, 0.6), (-30, 0.18), (90, 0.3), (90, 0.1), (-30, 0.18), (60, 0.3)])
    starts = sum(1 for a, b in zip(states, states[1:]) if a["mode"] != "forward" and b["mode"] == "forward")
    assert starts == 2
    assert all(s["side"] == "forehand" for s in states if s["mode"] == "forward")


def test_backswing_learned_from_recording():
    times = [i * DT for i in range(200)]
    peaks = [0.9, 1.9, 2.8]
    yaws = []
    for t in times:
        y = 0.0
        for p in peaks:
            if p - 0.4 <= t < p:
                y = 85 + (p - t) * 10          # drawn back to ~85-89 before each swing
        yaws.append(y)
    assert 84 < backswing_from_recording(times, yaws, peaks) < 90
