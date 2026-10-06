"""Player serving: 5-serve rotation, toss + strike out of the air, serve rules."""

import math

import config as C
from game import physics as P
from game import state_machine as SM
from hardware.swing import Calibration, ImuSample, SwingDetector, TossDetector
from tests.synth import recording
from tests.test_game_loop import advance, make_game, swing_at


def player_serving_game(seed=1):
    game, clock, events = make_game(seed=seed)
    game.match.match_first_server = "player"
    game.start_game("medium")          # match.reset() picks up the first server
    assert game.match.server == "player"
    return game, clock, events


def toss(game, clock):
    advance(game, clock, C.SERVE_ARM_DELAY_S + 0.05)
    assert game.serve_state == "await_toss" and game.paddle.toss_armed
    game.command({"type": "key", "key": " ", "down": True})      # Space = toss on the keyboard paddle
    advance(game, clock, 0.02)
    assert game.serve_state == "tossed" and game.sm.phase == SM.RALLY
    return game.incoming


def names(events):
    return [e["name"] for e in events]


def test_ball_waits_in_hand_until_the_toss():
    game, clock, events = player_serving_game()
    advance(game, clock, 3.0)
    assert game.sm.phase == SM.SERVE and game.serve_state == "await_toss"
    assert game.ball.pos == C.SERVE_POS


def test_toss_then_strike_is_a_legal_serve():
    game, clock, events = player_serving_game()
    inc = toss(game, clock)
    assert inc.serve and inc.required == "either"
    assert game.ball.vel[1] > 0                                   # going up
    advance(game, clock, inc.t - game.sim_t - 0.03)
    swing_at(game, inc.t)
    advance(game, clock, 0.05)
    hits = [e for e in events if e["name"] == "hit" and e["who"] == "player"]
    assert hits and hits[0]["serve"]
    assert game.flight.serve and game.flight.hitter == "player"
    advance(game, clock, 1.5)
    bounces = [e["side"] for e in events if e["name"] == "bounce"]
    assert bounces[:2] == ["player", "cpu"]                       # own half first, then theirs


def test_the_toss_motion_is_not_counted_as_the_serve_stroke():
    game, clock, events = player_serving_game()
    inc = toss(game, clock)
    swing_at(game, game.toss_t + 0.05)                            # paddle jerk right at the toss
    advance(game, clock, 0.05)
    assert not game.incoming.resolved
    assert not any(e["name"] == "miss" for e in events)


def test_toss_without_a_swing_is_a_missed_serve():
    game, clock, events = player_serving_game()
    toss(game, clock)
    advance(game, clock, 2.0, until=lambda: game.sm.phase == SM.POINT_OVER)
    assert [e for e in events if e["name"] == "miss"][0]["reason"] == "MISSED SERVE"
    assert game.match.points["cpu"] == 1


def test_swinging_too_early_on_the_serve():
    game, clock, events = player_serving_game()
    inc = toss(game, clock)
    early = inc.t - C.HIT_WINDOW_S - 0.08
    assert early > game.toss_t + C.TOSS_IGNORE_S
    advance(game, clock, early - game.sim_t + 0.01)
    swing_at(game, early)
    advance(game, clock, 0.05)
    assert [e for e in events if e["name"] == "miss"][0]["reason"] == "EARLY"


def test_serve_that_skips_own_half_is_a_fault():
    game, clock, events = player_serving_game()
    game.sm.to(SM.RALLY, game.sim_t)
    # Straight onto the opponent's half, no bounce on the server's side.
    shot = P.aim((0.2, 0.3, -1.5), (0.1, 0.8), 6.0)
    game.ball = P.Ball((0.2, 0.3, -1.5), shot.vel, shot.spin)
    game._begin_flight("player", serve=True)
    advance(game, clock, 1.5, until=lambda: game.sm.phase == SM.POINT_OVER)
    assert [e for e in events if e["name"] == "point"][0] == \
        {**[e for e in events if e["name"] == "point"][0], "winner": "cpu", "reason": "SERVE FAULT"}


def test_let_replays_the_serve_without_scoring():
    game, clock, events = player_serving_game()
    game.sm.to(SM.RALLY, game.sim_t)
    game.flight = None
    game._begin_flight("player", serve=True)
    game.flight.own_bounced = True
    game.flight.net = True
    game._referee(P.PhysicsEvent("bounce", (0.1, 0.02, 0.4), side="cpu"))
    assert "let" in names(events)
    assert game.match.points == {"player": 0, "cpu": 0}
    advance(game, clock, C.POINT_OVER_S + 0.2)
    assert game.sm.phase == SM.SERVE and game.match.server == "player"


def test_server_switches_after_five_points():
    game, clock, events = make_game()
    game.start_game("medium")
    servers = []
    for _ in range(10):
        servers.append(game.match.server)
        game.match.award_point("cpu")
    assert servers == ["cpu"] * 5 + ["player"] * 5


def test_cpu_serves_bounce_on_its_own_half_first():
    game, clock, events = make_game(seed=4)
    game.start_game("fast")
    advance(game, clock, 3.0, until=lambda: sum(e["name"] == "bounce" for e in events) >= 2)
    bounces = [e["side"] for e in events if e["name"] == "bounce"]
    assert bounces[:2] == ["cpu", "player"]


# ---- toss detection on the real paddle -------------------------------------------------

def flick(peak_g, gyro=0.0):
    """Synthetic upward jolt: gravity is +z in these recordings."""
    out = []
    for i in range(200):
        t = i * 0.015
        u = (t - 1.5) / 0.05
        out.append(ImuSample(t, 0, 0, 0, 0, 0, 1000 + 1000 * peak_g * math.exp(-u * u), 0, 0,
                             gyro * math.exp(-u * u)))
    return out


def tosses(samples):
    sd = SwingDetector(Calibration())
    td = TossDetector(sd)
    out = []
    for s in samples:
        sd.push(s)
        ev = td.push(s)
        if ev:
            out.append(ev)
    return out


def test_upward_flick_is_a_toss_and_sets_height():
    soft, hard = tosses(flick(1.2)), tosses(flick(3.0))
    assert len(soft) == 1 and len(hard) == 1
    assert C.TOSS_HEIGHT_MIN <= soft[0].height < hard[0].height <= C.TOSS_HEIGHT_MAX


def test_swings_and_twists_are_not_tosses():
    assert tosses(flick(0.4)) == []                                # too gentle
    assert tosses(flick(2.0, gyro=400)) == []                      # rotating = swing
    assert tosses(recording([{"peak_g": 4, "peak_dps": 600}] * 3)[0]) == []
