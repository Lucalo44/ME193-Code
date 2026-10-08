"""Hit-stop: the ball waits at the paddle for a late-reported real swing, then
leaves from the paddle instead of reappearing down the table."""

import json
import math

import config as C
from game import state_machine as SM
from game.loop import Game
from hardware.sim_paddle import SimPaddle
from hardware.swing import Calibration, SwingEvent
from tests.test_game_loop import FakeClock, advance


class LateReportingPaddle(SimPaddle):
    """Behaves like the real paddle: swings are reported some time after the strike."""
    kind = "real"
    busy = False

    def swing_in_progress(self):
        return self.busy


def make(seed=1):
    clock = FakeClock()
    events = []
    game = Game(LateReportingPaddle(Calibration()), None,
                publish_event=lambda m: events.append(json.loads(m)), seed=seed, clock=clock)
    game.start_game("medium")
    assert advance(game, clock, 5.0, until=lambda: game.incoming is not None and not game.incoming.resolved)
    return game, clock, events


def report_swing(game, t_peak_sim):
    game.paddle.swing_events.put(SwingEvent(
        t_peak=t_peak_sim + game.clock_offset + C.LATENCY_OFFSET_S, strength=0.5, return_speed=8.0,
        topspin=0.2, sidespin=0.0, peak_accel_g=3.0, peak_gyro_dps=500))


def dist(a, b):
    return math.dist(a, b)


def test_ball_waits_at_the_paddle_and_leaves_from_it():
    game, clock, events = make()
    inc = game.incoming
    game.paddle.busy = True                                   # player is mid-swing as the ball arrives
    advance(game, clock, inc.t - game.sim_t + 0.15)           # 0.15 s past arrival
    assert game.hold is not None
    assert dist(game.ball.pos, inc.pos) < 1e-9                # held exactly at the contact point
    report_swing(game, inc.t + 0.05)                          # swing peaked 50 ms after arrival, reported now
    advance(game, clock, 0.02)
    hit = [e for e in events if e["name"] == "hit" and e["who"] == "player"]
    assert hit and dist(hit[0]["pos"], inc.pos) < 1e-3        # contact at the paddle (events are rounded)
    assert dist(game.ball.pos, inc.pos) < 0.25                # leaves from the paddle -- no jump down the table
    assert game.ball.vel[2] > 0                               # heading back toward the CPU


def test_held_ball_is_released_when_no_swing_comes():
    game, clock, events = make()
    inc = game.incoming
    game.paddle.busy = False
    advance(game, clock, inc.t - game.sim_t + 0.05)
    assert game.hold is not None                              # brief wait in case a swing is starting
    advance(game, clock, C.HIT_HOLD_WAIT_S + 0.05)
    assert game.hold is None                                  # released, the ball carries on
    advance(game, clock, 1.5, until=lambda: game.sm.phase == SM.POINT_OVER)
    assert [e for e in events if e["name"] == "miss"][0]["reason"] == "NO SWING"


def test_hold_never_lasts_longer_than_the_limit():
    game, clock, events = make()
    inc = game.incoming
    game.paddle.busy = True                                   # waggling, but no swing ever reported
    advance(game, clock, inc.t - game.sim_t + C.HIT_HOLD_MAX_S + 0.1)
    assert game.hold is None


def test_keyboard_paddle_is_never_held():
    from tests.test_game_loop import make_game
    game, clock, events = make_game()
    game.start_game("medium")
    advance(game, clock, 5.0, until=lambda: game.incoming is not None and not game.incoming.resolved)
    advance(game, clock, game.incoming.t - game.sim_t + 0.1)
    assert game.hold is None
