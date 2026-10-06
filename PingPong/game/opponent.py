"""
game/opponent.py -- the computer player.

It predicts where the player's shot will reach its hit plane by forward-
simulating the ball with the same physics, slides toward that spot after a
reaction delay (shorter at faster settings), and either returns the ball or
misses. Miss chance rises with the incoming ball's speed and spin, so strong,
spinny returns win points.

Returns alternate roughly 50/50 left/right of the centerline, which is what
forces the player to switch between forehand and backhand.

The opponent faces the player (looking toward -z) and is right-handed, so its
forehand side is world -x.
"""

from __future__ import annotations

import math
import random
from typing import Optional

import config as C
from game import physics as P


class Opponent:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.x = 0.0
        self.target_x = 0.0
        self.move_after = 0.0            # sim time the reaction delay ends
        self.swing: Optional[str] = None
        self.swing_started = -10.0
        self.plan: Optional[P.Arrival] = None
        self.plan_t = 0.0                # sim time of predicted arrival
        self.will_miss = False
        self.speed_setting = "medium"

    # -- lifecycle --------------------------------------------------------
    def reset(self) -> None:
        self.x = self.target_x = 0.0
        self.swing = None
        self.plan = None
        self.will_miss = False

    @property
    def base_speed(self) -> float:
        return C.SPEED_BY_SETTING[self.speed_setting]

    def update(self, dt: float, now: float) -> None:
        if now < self.move_after:
            return
        err = self.target_x - self.x
        # Eased approach, capped at the max lateral speed.
        v = max(-C.OPPONENT_MAX_SPEED, min(C.OPPONENT_MAX_SPEED, err * 8.0))
        self.x += v * dt if abs(v * dt) < abs(err) else err

    def snapshot(self, now: float) -> dict:
        return {
            "x": round(self.x, 4),
            "swing": self.swing,
            "swing_t": round(now - self.swing_started, 3),
        }

    # -- reacting to the player's shot ------------------------------------
    def on_incoming(self, ball: P.Ball, now: float) -> None:
        self.plan = P.predict_receive(ball, "cpu", C.OPPONENT_HIT_PLANE_Z)
        self.move_after = now + C.OPPONENT_REACTION_S[self.speed_setting]
        self.swing = None
        if self.plan is None:
            # Ball is going out / into the net: drift back to center.
            self.target_x = 0.0
            self.will_miss = False
            return
        self.plan_t = now + self.plan.t
        self.target_x = max(-1.0, min(1.0, self.plan.pos[0]))
        speed = P.v_norm(self.plan.vel)
        spin = P.v_norm(ball.spin) / C.SPIN_MAX_RADS
        p_miss = (C.OPPONENT_MISS_BASE
                  + C.OPPONENT_MISS_PER_SPEED * max(0.0, speed - C.OPPONENT_MISS_SPEED_REF)
                  + C.OPPONENT_MISS_PER_SPIN * min(1.0, spin))
        self.will_miss = self.rng.random() < min(C.OPPONENT_MISS_MAX, p_miss)

    def start_swing(self, ball_x: float, now: float) -> None:
        self.swing = "forehand" if ball_x < self.x else "backhand"
        self.swing_started = now

    # -- hitting ----------------------------------------------------------
    def strike(self, ball: P.Ball, now: float, player_strength: float = 0.0) -> Optional[P.Shot]:
        """Called when the ball reaches the opponent. Returns the shot, or
        None for a whiff (ball sails past)."""
        if now - self.swing_started > 0.4:
            self.start_swing(ball.pos[0], now)
        out_of_reach = abs(ball.pos[0] - self.x) > C.OPPONENT_REACH
        if out_of_reach or (self.will_miss and self.rng.random() < 0.5):
            return None
        speed = (self.base_speed
                 + self.rng.uniform(-C.OPPONENT_SPEED_JITTER, C.OPPONENT_SPEED_JITTER)
                 + C.OPPONENT_HARD_RETURN_BONUS * player_strength)
        if self.will_miss:
            return self._error_shot(ball.pos, speed)
        return self._good_shot(ball.pos, speed)

    def serve(self, now: float) -> tuple:
        """Put the ball in play from the CPU's end. Returns (start_pos, Shot)."""
        start = (self.x, 0.22, C.OPPONENT_HIT_PLANE_Z)
        self.start_swing(self.x - 0.1, now)
        return start, self._good_shot(start, self.base_speed)

    def _pick_target(self) -> tuple:
        side = self.rng.choice((-1.0, 1.0))
        if self.rng.random() < C.OPPONENT_WIDE_SHOT_CHANCE:
            x = side * self.rng.uniform(0.6, 0.68)
        else:
            x = side * self.rng.uniform(0.18, 0.55)
        z = C.OPPONENT_TARGET_Z + self.rng.uniform(-0.3, 0.25)
        return x, z

    def _spin(self) -> tuple:
        top = self.rng.uniform(-0.4, 1.0) * C.OPPONENT_SPIN_MAX * C.SPIN_MAX_RADS
        side = self.rng.uniform(-0.25, 0.25) * C.SIDESPIN_MAX_RADS
        return top, side

    def _good_shot(self, start: P.Vec, speed: float) -> P.Shot:
        top, side = self._spin()
        return P.aim(start, self._pick_target(), speed, top, side, spin_aware=True)

    def _error_shot(self, start: P.Vec, speed: float) -> P.Shot:
        top, side = self._spin()
        x, z = self._pick_target()
        if self.rng.random() < 0.5:
            # Long: aim past the player's end line.
            return P.aim(start, (x, -P.HALF_L - self.rng.uniform(0.15, 0.5)), speed, top, side)
        # Into the net: same shot, launched a few degrees too low.
        shot = P.aim(start, (x, z), speed, top, side)
        heading = P.v_unit((shot.vel[0], 0.0, shot.vel[2]))
        vel = P.launch_velocity(speed, shot.theta - math.radians(self.rng.uniform(6, 10)),
                                (heading[0], heading[2]))
        return P.Shot(vel, P.spin_vector(vel, top, side), shot.theta, (x, z))
