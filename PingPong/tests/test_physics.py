import math
import random

import config as C
from game import physics as P


def fly(ball, seconds, rng=None):
    events = []
    for _ in range(int(seconds / P.DT)):
        events += P.step(ball, P.DT, rng or random.Random(0))
        if ball.dead:
            break
    return events


def test_free_fall_matches_gravity():
    b = P.Ball((0.0, 0.5, 2.5), (0, 0, 0))  # off the table end, falling
    fly(b, 0.2)
    expected = 0.5 - 0.5 * C.GRAVITY * 0.2 ** 2
    assert abs(b.pos[1] - expected) < 0.01


def test_table_bounce_restitution_and_side():
    b = P.Ball((0.2, 0.3, -0.5), (0, 0, 0))
    events = fly(b, 0.5)
    bounces = [e for e in events if e.kind == "bounce"]
    assert bounces and bounces[0].side == "player"
    assert b.pos[1] > 0.15  # came back up most of the way (e = 0.9 -> ~0.24 m)


def test_floor_kills_ball_off_table():
    b = P.Ball((0.0, 0.3, 2.0), (0, 0, 0.5))
    events = fly(b, 2.0)
    assert any(e.kind == "floor" for e in events)
    assert b.dead


def test_topspin_dips_and_backspin_floats():
    start, vel = (0.0, 0.25, -1.5), P.launch_velocity(8.0, math.radians(8), (0.0, 1.0))
    land = {}
    for name, top in (("top", 120.0), ("flat", 0.0), ("back", -120.0)):
        b = P.Ball(start, vel, P.spin_vector(vel, top, 0.0))
        land[name] = P.plane_landing(b).pos[2]
    assert land["top"] < land["flat"] < land["back"]


def test_topspin_kicks_forward_on_bounce():
    out = {}
    for name, top in (("top", 120.0), ("back", -120.0)):
        vel = (0.0, -2.0, 4.0)
        b = P.Ball((0.0, 0.05, 0.5), vel, P.spin_vector(vel, top, 0.0))
        fly(b, 0.05)
        out[name] = b.vel[2]
    assert out["top"] > out["back"]


def test_sidespin_curves():
    vel = P.launch_velocity(8.0, math.radians(10), (0.0, 1.0))
    right = P.plane_landing(P.Ball((0, 0.25, -1.5), vel, P.spin_vector(vel, 0.0, 80.0))).pos[0]
    left = P.plane_landing(P.Ball((0, 0.25, -1.5), vel, P.spin_vector(vel, 0.0, -80.0))).pos[0]
    assert right > 0.05 and left < -0.05


def test_low_shot_hits_net_and_stays_on_hitter_side():
    vel = P.launch_velocity(6.0, math.radians(-2), (0.0, 1.0))
    b = P.Ball((0.0, 0.12, -1.0), vel)
    events = fly(b, 1.0)
    assert any(e.kind == "net" for e in events)
    assert b.pos[2] < 0.05


def test_aim_spin_aware_hits_target_and_clears_net():
    for speed in (C.BALL_SPEED_SLOW, C.BALL_SPEED_MEDIUM, C.BALL_SPEED_FAST):
        start = (0.2, 0.22, C.OPPONENT_HIT_PLANE_Z)
        shot = P.aim(start, (-0.4, -0.85), speed, 40.0, 10.0, spin_aware=True)
        land = P.plane_landing(P.Ball(start, shot.vel, shot.spin))
        assert land.net_clearance > 0
        assert P.on_table(land.pos[0], land.pos[2]) and land.pos[2] < 0


def test_good_opponent_shot_reaches_player_plane():
    start = (0.0, 0.22, C.OPPONENT_HIT_PLANE_Z)
    for tx in (-0.5, 0.5):
        shot = P.aim(start, (tx, -0.85), C.BALL_SPEED_MEDIUM, 30.0, 0.0)
        arr = P.predict_receive(P.Ball(start, shot.vel, shot.spin), "player", C.PLAYER_HIT_PLANE_Z)
        assert arr is not None
        assert 0.3 < arr.t < 1.2
        assert (arr.pos[0] > 0) == (tx > 0)


def test_full_assist_always_lands():
    rng = random.Random(3)
    start = (0.1, 0.25, C.PLAYER_HIT_PLANE_Z)
    for speed in (C.RETURN_SPEED_MIN, 9.0, C.RETURN_SPEED_MAX):
        for top in (-0.5, 0.0, 1.0):
            shot = P.player_return(start, speed, top, 0.0, rng, assist=1.0)
            arr = P.predict_receive(P.Ball(start, shot.vel, shot.spin), "cpu", C.OPPONENT_HIT_PLANE_Z)
            assert arr is not None, (speed, top)


def test_hard_flat_without_assist_goes_further_than_soft():
    rng = random.Random(1)
    start = (0.0, 0.25, C.PLAYER_HIT_PLANE_Z)
    soft = P.player_return(start, C.RETURN_SPEED_MIN, -1.0, 0.0, rng, assist=0.0)
    hard = P.player_return(start, C.RETURN_SPEED_MAX, -1.0, 0.0, rng, assist=0.0)
    d_soft = P.plane_landing(P.Ball(start, soft.vel, soft.spin)).pos[2]
    d_hard = P.plane_landing(P.Ball(start, hard.vel, hard.spin)).pos[2]
    assert d_hard > d_soft
