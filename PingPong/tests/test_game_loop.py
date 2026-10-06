"""Headless end-to-end tests of the game loop with the sim paddle and a fake
clock: hits, EARLY / LATE / NO SWING / WRONG STROKE misses, scoring."""

import json

import config as C
from game import state_machine as SM
from game.loop import Game
from hardware.sim_paddle import SimPaddle
from hardware.swing import Calibration, SwingEvent


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make_game(seed=1):
    clock = FakeClock()
    events = []
    game = Game(SimPaddle(Calibration()), None, publish_event=lambda m: events.append(json.loads(m)),
                seed=seed, clock=clock)
    return game, clock, events


def advance(game, clock, seconds, until=None):
    dt = 1.0 / C.LOOP_HZ
    for _ in range(int(seconds / dt)):
        clock.t += dt
        game.update(dt)
        if until and until():
            return True
    return False


def swing_at(game, sim_t, stroke=None, strength=0.5):
    ev = SwingEvent(t_peak=sim_t + game.clock_offset + C.LATENCY_OFFSET_S, strength=strength,
                    return_speed=8.0, topspin=0.2, sidespin=0.0, peak_accel_g=3.0,
                    peak_gyro_dps=500, stroke_hint=stroke)
    game.paddle.swing_events.put(ev)


def wait_incoming(game, clock):
    assert advance(game, clock, 5.0, until=lambda: game.incoming is not None and not game.incoming.resolved)
    return game.incoming


def names(events):
    return [e["name"] for e in events]


def test_serve_then_good_hit_reaches_cpu():
    game, clock, events = make_game()
    game.start_game("medium")
    inc = wait_incoming(game, clock)
    advance(game, clock, inc.t - game.sim_t - 0.05)
    swing_at(game, inc.t)
    advance(game, clock, 0.3)
    hits = [e for e in events if e["name"] == "hit" and e["who"] == "player"]
    assert hits, names(events)
    assert game.flight.hitter == "player"
    assert game.sm.phase == SM.RALLY


def test_no_swing_is_a_miss():
    game, clock, events = make_game()
    game.start_game("medium")
    wait_incoming(game, clock)
    advance(game, clock, 2.0, until=lambda: game.sm.phase == SM.POINT_OVER)
    misses = [e for e in events if e["name"] == "miss"]
    assert misses and misses[0]["reason"] == "NO SWING"
    assert game.match.points["cpu"] == 1


def test_early_and_late():
    for offset, reason in ((-0.4, "EARLY"), (0.35, "LATE")):
        game, clock, events = make_game()
        game.start_game("slow")
        inc = wait_incoming(game, clock)
        target = inc.t + offset
        advance(game, clock, max(0.0, target - game.sim_t + 0.02))
        swing_at(game, target)
        advance(game, clock, 0.05)
        misses = [e for e in events if e["name"] == "miss"]
        assert misses and misses[0]["reason"] == reason, (reason, names(events))


def test_wrong_stroke():
    game, clock, events = make_game()
    game.start_game("medium")
    inc = wait_incoming(game, clock)
    wrong = {"forehand": "backhand", "backhand": "forehand"}.get(inc.required)
    if wrong is None:
        return  # landed in the deadband; any stroke is fine
    advance(game, clock, inc.t - game.sim_t - 0.05)
    swing_at(game, inc.t, stroke=wrong)
    advance(game, clock, 0.05)
    misses = [e for e in events if e["name"] == "miss"]
    assert misses and misses[0]["reason"] == "WRONG STROKE"
    assert misses[0]["needed"] == inc.required


def test_rally_continues_and_points_accumulate():
    game, clock, events = make_game(seed=7)
    game.start_game("medium")
    for _ in range(300):
        if game.sm.phase == SM.RALLY and game.incoming and not game.incoming.resolved and game.pending is None:
            inc = game.incoming
            advance(game, clock, max(0.0, inc.t - game.sim_t - 0.03))
            swing_at(game, inc.t)
        advance(game, clock, 0.05)
    points = game.match.points["player"] + game.match.points["cpu"] + 11 * sum(game.match.games.values())
    assert points >= 3
    assert sum(1 for e in events if e["name"] == "hit" and e["who"] == "player") >= 3


def test_tags_ignored_mid_rally():
    sm = SM.StateMachine()
    assert sm.tags_active
    sm.to(SM.SERVE, 0)
    assert not sm.tags_active


def test_dead_ball_still_resolves_hit_window():
    """Regression: the ball can hit the floor before the hit window expires;
    the point must still be settled."""
    game, clock, events = make_game()
    game.start_game("fast")
    inc = wait_incoming(game, clock)
    advance(game, clock, inc.t - game.sim_t + 0.05)
    game.ball.dead = True
    advance(game, clock, 1.0, until=lambda: game.sm.phase == SM.POINT_OVER)
    assert game.sm.phase == SM.POINT_OVER
    assert [e for e in events if e["name"] == "miss"][0]["reason"] == "NO SWING"
