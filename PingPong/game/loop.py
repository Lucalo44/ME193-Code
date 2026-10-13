"""
game/loop.py -- the fixed-timestep game loop that combines every input.

Inputs:   paddle swing events (real or sim), vision results (tags + pose
          judge), key commands from the browser/terminal.
Outputs:  state snapshots and discrete events, as JSON strings, to whatever
          publish functions were passed in (the WebSocket server in main.py).

Time: `sim_t` is the game clock (seconds, frozen while paused), advanced in
fixed physics sub-steps of 1/PHYSICS_HZ. Swing and camera timestamps are
time.monotonic(); `clock_offset` converts them (sim = mono - clock_offset).

Serving
-------
The server switches every SERVES_PER_TURN points (Match.server). When it's the
player's serve the ball waits in their hand ("await_toss") until a toss:
an upward flick of the real paddle (hardware TossDetector) or Space on the
keyboard paddle. The ball flies straight up, and the player strikes it out of
the air with a normal swing as it falls back to SERVE_CONTACT_Y -- the usual
hit window applies (EARLY / LATE / MISSED SERVE). Serves (both sides) must
bounce on the server's half first, then the receiver's; a serve that clips
the net and still lands is a let and is replayed.

The referee
-----------
After each hit the ball must first bounce on the receiver's half. A ball
that lands on the hitter's own half, goes into the net, or reaches the floor
without bouncing loses the point for the hitter. When the player is the
receiver, the hit window decides instead (EARLY / LATE / WRONG STROKE /
NO SWING), see _on_swing and _check_player_expiry.
"""

from __future__ import annotations

import json
import math
import os
import queue
import random
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import config as C
from game import physics as P
from game import state_machine as SM
from game.opponent import Opponent
from game.rules import Match, decide_stroke, other, required_stroke, stroke_ok
from game.streak import StreakTracker
from server import protocol

SERVE_HEIGHT = 0.22


def serve_speed(strength: float) -> float:
    s = max(0.0, min(1.0, strength)) ** C.RETURN_STRENGTH_CURVE
    return C.SERVE_SPEED_MIN + (C.SERVE_SPEED_MAX - C.SERVE_SPEED_MIN) * s


@dataclass
class Flight:
    hitter: str
    receiver: str
    t0: float
    bounces: int = 0              # bounces on the receiver's half
    net: bool = False
    struck: bool = False          # CPU receiver already swung at it
    serve: bool = False           # a serve: must bounce on the server's half first
    own_bounced: bool = False     # ... and has it?


@dataclass
class Incoming:
    t: float                      # sim time the ball reaches the player's hit plane
    pos: P.Vec
    required: str                 # "forehand" | "backhand" | "either"
    resolved: bool = False
    serve: bool = False           # the player's own serve toss (any stroke is fine)
    early_s: float = 0.2          # hit window: this long before arrival ...
    late_s: float = 0.2           # ... to this long after (see hit_window)


@dataclass
class PendingHit:
    t: float                      # sim time of contact
    swing: object                 # hardware.swing.SwingEvent
    stroke: Optional[str]
    confidence: float
    required: str
    serve: bool = False
    swing_id: Optional[int] = None


class Game:
    def __init__(self, paddle, vision=None,
                 publish_state: Optional[Callable[[str], None]] = None,
                 publish_event: Optional[Callable[[str], None]] = None,
                 debug: bool = False, seed: Optional[int] = None,
                 clock: Callable[[], float] = time.monotonic,
                 streak: Optional[StreakTracker] = None, mqtt_status: Callable[[], str] = lambda: "off",
                 settings_file: Optional[str] = None, recorder=None,
                 on_score: Optional[Callable[[dict], None]] = None):
        self.paddle = paddle
        self._on_score = on_score or (lambda score: None)   # main.py: publish over MQTT
        self.recorder = recorder                  # game/recorder.py, or None (no recording)
        self._swing_id = 0
        self._last_hit_id: Optional[int] = None   # the player's shot in flight, for its result
        self._last_call: Optional[dict] = None    # last miss call, for the "that was wrong" key
        self.vision = vision
        self._publish_state = publish_state or (lambda m: None)
        self._publish_event = publish_event or (lambda m: None)
        self.clock = clock
        self.debug = debug
        self.show_hint = C.SHOW_STROKE_HINT
        self.rng = random.Random(seed)
        self.sm = SM.StateMachine()
        self.match = Match()
        self.opponent = Opponent(self.rng)
        self.ball = P.Ball(dead=True)
        self.speed_setting = "medium"
        self.sim_t = 0.0
        self._acc = 0.0
        self.clock_offset = clock()
        self.flight: Optional[Flight] = None
        self.incoming: Optional[Incoming] = None
        self.pending: Optional[PendingHit] = None
        self.history: deque = deque(maxlen=int(1.5 * C.PHYSICS_HZ))
        self.point_result = None
        self.player_strength = 0.0
        self.latency: Optional[dict] = None
        self.commands: "queue.Queue[dict]" = queue.Queue()
        self._stop = threading.Event()
        self._last_tag = (None, 0.0)
        self.streak = streak or StreakTracker()
        self.serve_state: Optional[str] = None   # None | "cpu" | "await_toss" | "tossed"
        self.toss_t = -10.0
        self._toss_diag_t = -1.0                 # last rejected flick already reported
        self._swing_while_awaiting: Optional[float] = None
        self.hold: Optional[dict] = None         # hit-stop: ball waiting at the paddle (see _hold_ball)
        self._planned_serve = None               # CPU serve worked out while the ball is in hand
        self._mqtt_status = mqtt_status
        self._settings_file = settings_file
        self.handedness = C.HANDEDNESS
        self.set_handedness(self._load_settings().get("handedness", C.HANDEDNESS), save=False)
        self._warm_up_serves()

    # -- player settings (remembered between runs) --------------------------
    def _load_settings(self) -> dict:
        if not self._settings_file or not os.path.exists(self._settings_file):
            return {}
        try:
            with open(self._settings_file) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def set_handedness(self, handedness: str, save: bool = True) -> None:
        """Left- or right-handed: flips the forehand/backhand side rule, the paddle's
        default backswing direction, the paddle arm the camera follows and the
        on-screen paddle."""
        if handedness not in ("left", "right"):
            return
        self.handedness = handedness
        if hasattr(self.paddle, "set_handedness"):
            self.paddle.set_handedness(handedness)
        arm = getattr(self.vision, "arm", None)
        if arm is not None:
            arm.handedness = handedness
            arm.votes = [0, 0]                      # let swings re-confirm the paddle arm
        if hasattr(self.vision, "set_hand"):
            self.vision.set_hand(handedness)        # stroke classifier: mirror / pick that hand's data
        if save and self._settings_file:
            data = self._load_settings()
            data["handedness"] = handedness
            try:
                with open(self._settings_file, "w") as f:
                    json.dump(data, f, indent=2)
            except OSError as exc:
                print(f"Could not save settings: {exc}")

    def _warm_up_serves(self) -> None:
        """Find a legal CPU serve once per speed at startup, so the serve search
        (tens of ms the first time) never runs during play."""
        speed0 = self.opponent.speed_setting
        state = self.rng.getstate()
        for setting in C.SPEED_BY_SETTING:
            self.opponent.speed_setting = setting
            self.opponent.plan_serve()
        self.opponent.speed_setting = speed0
        self.rng.setstate(state)

    # ======================================================================
    # Running
    # ======================================================================
    def command(self, msg: dict) -> None:
        """Thread-safe: queue a key command from the browser or terminal."""
        self.commands.put(msg)

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        period = 1.0 / C.LOOP_HZ
        publish_every = max(1, C.LOOP_HZ // C.STATE_BROADCAST_HZ)
        last = next_t = self.clock()
        n = 0
        while not self._stop.is_set():
            now = self.clock()
            dt, last = min(0.1, now - last), now
            self.update(dt)
            n += 1
            if n % publish_every == 0:
                self.publish()
            next_t += period
            delay = next_t - self.clock()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = self.clock()

    def update(self, dt: float) -> None:
        """One loop iteration covering `dt` seconds of wall time."""
        self._drain_commands()
        paused = self.sm.paused
        if not paused:
            self._acc += dt
            while self._acc >= P.DT:
                self._acc -= P.DT
                self.sim_t += P.DT
                self._substep()
        self.clock_offset = self.clock() - (self.sim_t + self._acc)
        self.paddle.tick(dt)
        if self.recorder is not None:
            self.recorder.flush()
        self._drain_swings(ignore=paused)
        self._drain_tosses(ignore=paused)
        self._poll_tags()
        if not paused:
            self.opponent.update(dt, self.sim_t)
            self._phase_logic()

    # ======================================================================
    # Phases
    # ======================================================================
    def _auto_zero_arm(self) -> None:
        """If the arm's ready position was never captured (no Z press), take it the
        first time the player holds the paddle still at ready while a point is served."""
        arm = getattr(self.vision, "arm", None)
        if arm is None or any(n is not None for n in arm.neutral) or self.sm.phase != SM.SERVE:
            return
        st = getattr(self.paddle, "stroke_state", lambda: None)()
        if st is None or (st["mode"] == "ready" and abs(st["phase"]) < 0.15):
            arm.zero()

    def _phase_logic(self) -> None:
        self._auto_zero_arm()
        ph, now = self.sm.phase, self.sim_t
        elapsed = self.sm.elapsed(now)
        if ph == SM.SERVE:
            if self.player_serves:
                # Ball in the player's hand, waiting for the toss.
                if self.serve_state != "await_toss":
                    self.serve_state = "await_toss"
                    self._emit("serve_prompt", by="player")
                self.paddle.toss_armed = elapsed >= C.SERVE_ARM_DELAY_S
                self.ball = P.Ball(C.SERVE_POS)
                self._report_rejected_toss()
                sw = self._swing_while_awaiting
                if sw is not None and self.sim_t - sw > 0.4:
                    self._swing_while_awaiting = None
                    self._emit("message", text="That was read as a swing -- toss with a sharp flick straight up")
            else:
                if self.serve_state != "cpu":
                    # Plan the serve now, while the ball is still in hand and nothing
                    # is moving, so the computation never freezes play mid-motion.
                    self.serve_state = "cpu"
                    self._planned_serve = self.opponent.plan_serve()
                self.paddle.toss_armed = False
                self.ball = P.Ball((self.opponent.x, SERVE_HEIGHT, C.OPPONENT_HIT_PLANE_Z))
                if elapsed >= C.SERVE_DELAY_S:
                    start, shot = self.opponent.serve(now, self._planned_serve)
                    self._planned_serve = None
                    self.ball = P.Ball(start, shot.vel, shot.spin)
                    self.sm.to(SM.RALLY, now)
                    self._begin_flight("cpu", serve=True)
                    self._emit("serve", pos=start, by="cpu")
                    self._emit("hit", who="cpu", pos=start, speed=P.v_norm(shot.vel))
        elif ph == SM.RALLY:
            f = self.flight
            plan = self.opponent.plan
            if f and f.hitter == "player" and plan and self.opponent.swing is None \
                    and now >= self.opponent.plan_t - 0.18:
                self.opponent.start_swing(plan.pos[0], now)
            if f and now - f.t0 > C.RALLY_TIMEOUT_S:
                self._ball_dead()
        elif ph == SM.POINT_OVER:
            r = self.point_result
            wait = C.GAME_BREAK_S if (r and r.game_over) else C.POINT_OVER_S
            if elapsed >= wait:
                if r and r.match_over:
                    self.sm.to(SM.GAME_OVER, now)
                    self.ball.dead = True
                else:
                    if r and r.game_over:
                        self.match.start_next_game()
                    self.sm.to(SM.SERVE, now)
        elif ph == SM.LATENCY_CAL:
            if now > self.latency["beats"][-1] + 1.0:
                self._finish_latency_cal()

    def start_game(self, setting: str) -> None:
        if self.sm.phase not in (SM.LOBBY, SM.GAME_OVER):
            self.reset()
        self.speed_setting = setting
        self.opponent.speed_setting = setting
        self.opponent.reset()
        self.match.reset()
        self.streak.new_game()                    # the record is per game
        self._clear_rally()
        self.sm.to(SM.SERVE, self.sim_t)
        self._publish_score("game_start")
        if self.recorder is not None:
            self.recorder.open()
            self._record("game_start", speed=setting, handedness=self.handedness, paddle=self.paddle.kind,
                         latency_offset_s=C.LATENCY_OFFSET_S)
            self._record_paddle_state("paddle_state")
        self._emit("message", text=f"{setting.upper()} - game on!")

    def reset(self) -> None:
        self.sm.reset(self.sim_t)
        self._clear_rally()
        self.ball = P.Ball(dead=True)
        self.opponent.reset()
        self.latency = None

    @property
    def player_serves(self) -> bool:
        return C.PLAYER_SERVE and self.match.server == "player"

    def _clear_rally(self) -> None:
        self.flight = None
        self.incoming = None
        self.pending = None
        self.serve_state = None
        self.paddle.toss_armed = False
        self.hold = None

    # ======================================================================
    # Physics sub-step + referee
    # ======================================================================
    def _substep(self) -> None:
        ph = self.sm.phase
        if ph not in (SM.RALLY, SM.POINT_OVER):
            return
        if ph == SM.RALLY and self.pending and self.sim_t >= self.pending.t:
            self._execute_player_hit()
        if ph == SM.RALLY and self._hold_ball():
            return                            # ball waiting at the paddle
        if self.ball.dead:
            if ph == SM.RALLY:
                self._check_player_expiry()   # a dead ball may still be waiting on the hit window
            return
        events = P.step(self.ball, P.DT, self.rng)
        self.history.append((self.sim_t, self.ball.pos))
        for ev in events:
            self._emit(ev.kind, pos=ev.pos, side=ev.side)
            if self.sm.phase == SM.RALLY:
                self._referee(ev)
        if self.sm.phase == SM.RALLY:
            self._check_cpu_strike()
            self._check_player_expiry()

    def _referee(self, ev: P.PhysicsEvent) -> None:
        f = self.flight
        if f is None:
            return
        if ev.kind == "net":
            f.net = True
            if f.hitter == "cpu":
                self._predict_incoming(serve=f.serve and not f.own_bounced)
        elif ev.kind == "bounce":
            if ev.side == f.hitter:
                if f.serve and not f.own_bounced and f.bounces == 0:
                    f.own_bounced = True          # a serve's first bounce, on the server's half
                    return
                self._end_point(f.receiver, "SERVE FAULT" if f.serve and not f.net else
                                "NET" if f.net else "OWN SIDE")
                return
            if f.serve and not f.own_bounced:
                self._end_point(f.receiver, "SERVE FAULT")   # skipped the server's half
                return
            if f.serve and f.net and f.bounces == 0:
                self._let()
                return
            f.bounces += 1
            if f.hitter == "player" and f.bounces == 1:
                self._count_hit()
            if f.receiver == "player":
                if self.incoming is None:
                    self._predict_incoming(bounced=True)
                # A second bounce is settled by the hit-window expiry.
            elif f.bounces >= 2:
                self._end_point("player", "CPU MISSED")
        elif ev.kind == "floor":
            self._ball_dead()

    def _ball_dead(self) -> None:
        """Ball hit the floor or stalled: decide who lost the rally."""
        f = self.flight
        if f is None:
            return
        if f.bounces == 0:
            self._end_point(f.receiver, "NET" if f.net else "OUT")
        elif f.receiver == "cpu":
            self._end_point("player", "CPU MISSED")
        elif self.incoming is None or self.incoming.resolved and self.pending is None:
            self._end_point("cpu", "NO SWING")
        # Otherwise the player's hit window is still open; expiry decides.

    def _check_cpu_strike(self) -> None:
        f = self.flight
        if not f or f.hitter != "player" or f.bounces < 1 or f.struck:
            return
        b = self.ball
        if b.pos[2] >= C.OPPONENT_HIT_PLANE_Z or (b.vel[1] < 0 and b.pos[1] < 0.07):
            f.struck = True
            shot = self.opponent.strike(b, self.sim_t, self.player_strength)
            if shot is None:
                return  # whiff: the ball flies on and the floor settles it
            self._record("cpu_shot", pos=b.pos, ball_vel=b.vel, ball_spin=b.spin, vel=shot.vel, spin=shot.spin)
            self.ball = P.Ball(b.pos, shot.vel, shot.spin)
            self._begin_flight("cpu")
            self._emit("hit", who="cpu", pos=b.pos, speed=P.v_norm(shot.vel))

    def _hold_ball(self) -> bool:
        """Hit-stop. A real swing registers ~0.1-0.25 s after the strike (plus
        Bluetooth delay), by which time the ball would be well past the paddle,
        and the return would appear out of nowhere further down the table. So
        when the ball reaches the contact point while a swing is under way, hold
        it there until the swing registers (_execute_player_hit then launches it
        from right here) or it's clear there's no hit, and let it carry on.
        Returns True while the ball is held."""
        inc = self.incoming
        if self.hold is not None:
            if inc is None or inc.resolved:
                self.hold = None              # hit executed, or a miss was called
                return False
            waited = self.sim_t - self.hold["t0"]
            busy = self.paddle.swing_in_progress()
            if waited > C.HIT_HOLD_MAX_S or (not busy and waited > C.HIT_HOLD_WAIT_S):
                self.hold = None              # no hit coming: release the ball
                return False
            # Soft catch: ease to a stop instead of freezing dead.
            ease = C.HIT_HOLD_EASE_S * (1.0 - math.exp(-waited / C.HIT_HOLD_EASE_S)) if C.HIT_HOLD_EASE_S > 0 else 0.0
            p0, v = self.hold["pos"], self.hold["vel"]
            self.ball.pos = (p0[0] + v[0] * ease, p0[1] + v[1] * ease, p0[2] + v[2] * ease)
            return True
        if inc is None or inc.resolved or self.pending is not None or self.sim_t < inc.t \
                or self.paddle.kind not in ("real", "replay") or self.sim_t - inc.t > 0.05:
            return False
        self.hold = {"t0": self.sim_t, "pos": inc.pos, "vel": self.ball.vel}
        self.ball.pos = inc.pos
        return True

    def _check_player_expiry(self) -> None:
        if self.hold is not None:
            return                            # still waiting at the paddle
        inc = self.incoming
        if inc and not inc.resolved and self.pending is None \
                and self.sim_t > inc.t + inc.late_s + C.LATE_ZONE_S:
            inc.resolved = True
            # No swing registered in time: was the paddle moving at all? (tools/game_report.py
            # checks imu.csv around `arrival_t` to tell a missed swing from no swing.)
            self._record("missed_ball", required=inc.required, serve=inc.serve, pos=inc.pos,
                         arrival_t=inc.t + self.clock_offset, ball_vel=self.ball.vel,
                         swing_in_progress=self.paddle.swing_in_progress())
            self._player_miss("MISSED SERVE" if inc.serve else "NO SWING")

    def _begin_flight(self, hitter: str, serve: bool = False) -> None:
        self.flight = Flight(hitter, other(hitter), self.sim_t, serve=serve)
        if hitter == "cpu":
            self._predict_incoming(serve=serve)
        else:
            self.incoming = None
            self.opponent.on_incoming(self.ball, self.sim_t, serve=serve)

    def hit_window(self, ball_speed: float) -> tuple:
        """(early, late) seconds around the ball's arrival during which a swing
        strikes it: the difficulty's reach divided by the ball's speed, so faster
        balls give tighter windows."""
        v = max(0.5, ball_speed)
        clamp = lambda x: max(C.HIT_WINDOW_MIN_S, min(C.HIT_WINDOW_MAX_S, x))  # noqa: E731
        s = self.speed_setting
        return clamp(C.HIT_REACH_EARLY_M[s] / v), clamp(C.HIT_REACH_LATE_M[s] / v)

    def _predict_incoming(self, bounced: bool = False, serve: bool = False) -> None:
        arr = P.predict_receive(self.ball, "player", C.PLAYER_HIT_PLANE_Z, bounced=bounced, serve=serve)
        if arr is None:
            self.incoming = None
            return
        early, late = self.hit_window(P.v_norm(arr.vel))
        self.incoming = Incoming(self.sim_t + arr.t, arr.pos, required_stroke(arr.pos[0], self.handedness),
                                 early_s=early, late_s=late)

    def _count_hit(self) -> None:
        """One continuous hit: the player's return landed in. Tracks the record."""
        self._record("shot_result", swing_id=self._last_hit_id, result="in", pos=self.ball.pos)
        if self.streak.hit():
            self._emit("record", record=self.streak.record)

    def _end_point(self, winner: str, reason: str, game_penalty: bool = False, **extra) -> None:
        f = self.flight
        if winner == "cpu" and f is not None and f.hitter == "player" and f.bounces == 0:
            self._record("shot_result", swing_id=self._last_hit_id, result=reason, pos=self.ball.pos)
        if winner == "cpu":
            self._last_call = {"reason": reason, "sim_t": self.sim_t, "swing_id": extra.get("swing_id"), **extra}
            self.streak.reset_streak()      # only losing a point breaks the streak
            self._emit("miss", reason=reason, **extra)
        result = self.match.award_game(winner) if game_penalty else self.match.award_point(winner)
        self.point_result = result
        self._emit("point", winner=winner, reason=reason, score=self.match.snapshot())
        self._publish_score("point", winner=winner, reason=reason, game_over=result.game_over,
                            match_over=result.match_over)
        if result.game_over:
            self._emit("game_over", match_over=result.match_over, winner=result.game_winner,
                       match_winner=result.match_winner, score=self.match.snapshot())
        self._clear_rally()
        self.opponent.target_x = 0.0
        self.opponent.plan = None
        self.sm.to(SM.POINT_OVER, self.sim_t)

    def _let(self) -> None:
        """A serve clipped the net and landed: no point, serve again."""
        self._emit("let")
        self._clear_rally()
        self.point_result = None
        self.opponent.target_x = 0.0
        self.opponent.plan = None
        self.sm.to(SM.POINT_OVER, self.sim_t)

    def _player_miss(self, reason: str, **extra) -> None:
        penalty = reason == "WRONG STROKE" and C.WRONG_STROKE_PENALTY == "game"
        self._end_point("cpu", reason, game_penalty=penalty, **extra)

    # ======================================================================
    # Player swings
    # ======================================================================
    def _drain_swings(self, ignore: bool = False) -> None:
        while True:
            try:
                ev = self.paddle.swing_events.get_nowait()
            except queue.Empty:
                return
            if not ignore:
                self._on_swing(ev)

    def _drain_tosses(self, ignore: bool = False) -> None:
        while True:
            try:
                ev = self.paddle.toss_events.get_nowait()
            except queue.Empty:
                return
            if not ignore and self.sm.phase == SM.SERVE and self.serve_state == "await_toss" \
                    and self.sm.elapsed(self.sim_t) >= C.SERVE_ARM_DELAY_S:
                self._toss(ev)

    def _report_rejected_toss(self) -> None:
        """Tell the player why a flick didn't count as a toss (real paddle)."""
        last = getattr(getattr(self.paddle, "toss", None), "last", None)
        if not last or last["t"] <= self._toss_diag_t or last["result"] == "toss":
            return
        self._toss_diag_t = last["t"]
        if last["t"] < self.clock_offset + self.sm.entered_at + C.SERVE_ARM_DELAY_S:
            return  # from before the serve came up
        self._swing_while_awaiting = None
        self._emit("message", text=f"Toss not counted: {last['result']}")

    def _toss(self, ev) -> None:
        """Throw the ball straight up; the player must hit it as it comes down."""
        start = C.SERVE_POS
        vel, t_contact = P.toss_contact(start, ev.height, C.SERVE_CONTACT_Y)
        self.ball = P.Ball(start, vel)
        contact = (start[0], C.SERVE_CONTACT_Y, start[2])
        self.toss_t = self.sim_t
        self._swing_while_awaiting = None
        self.flight = None                     # nothing to referee until it's struck
        # The window opens once the ball is falling (a swing while it's still rising is a
        # whiff), and its width follows the ball's falling speed at contact.
        t_apex = vel[1] / C.GRAVITY
        early, late = self.hit_window(C.GRAVITY * max(0.0, t_contact - t_apex))
        early = min(early, max(C.HIT_WINDOW_MIN_S, t_contact - t_apex))
        self.incoming = Incoming(self.sim_t + t_contact, contact, "either", serve=True, early_s=early, late_s=late)
        self.serve_state = "tossed"
        self.paddle.toss_armed = False
        self.sm.to(SM.RALLY, self.sim_t)
        self._emit("toss", height=ev.height, pos=start)

    def _on_swing(self, ev) -> None:
        t = ev.t_hit - self.clock_offset           # sim time of contact (hardware/swing.py: t_contact)
        t_corr = t - C.LATENCY_OFFSET_S
        arm = getattr(self.vision, "arm", None)
        if arm is not None and self.paddle.kind != "sim":
            arm.on_swing(ev.t_peak)                 # teaches it which arm holds the paddle
        self._emit("swing", trace=ev.trace(self.paddle.calibration), **ev.summary())
        self._swing_id += 1
        rec = self._swing_record(ev, t, t_corr) if self.recorder is not None and self.recorder.active else None
        outcome = self._handle_swing(ev, t, t_corr, rec)
        if rec is not None:
            rec["outcome"] = outcome
            self.recorder.swing(rec)

    def _handle_swing(self, ev, t: float, t_corr: float, rec: Optional[dict]) -> str:
        """Act on one swing; returns what it was (the recorded outcome)."""
        if self.sm.phase == SM.LATENCY_CAL:
            self._latency_swing(t)
            return "latency_cal"
        if self.sm.phase == SM.SERVE and self.serve_state == "await_toss" and self.paddle.kind != "sim":
            self._swing_while_awaiting = self.sim_t   # reported shortly unless a toss follows
            return "awaiting_toss"
        inc = self.incoming
        if self.sm.phase != SM.RALLY or inc is None or inc.resolved or self.pending:
            return "no_ball"
        if inc.serve and t < self.toss_t + C.TOSS_IGNORE_S:
            return "toss_motion"  # the tossing motion itself, not the serve stroke
        dt = t_corr - inc.t
        if dt > inc.late_s + C.LATE_ZONE_S:
            return "stray"  # stray swing, long after the ball
        if dt < -inc.early_s:
            # Too early: a whiff, not a miss -- the player can swing again (Wii-style).
            self._emit("early_swing", by_s=-dt)
            return "early"
        if dt > inc.late_s:
            inc.resolved = True
            self._player_miss("LATE", by_s=dt, swing_id=self._swing_id)
            return "late"
        if inc.serve:
            # Any stroke can serve; no pose check.
            inc.resolved = True
            self.pending = PendingHit(max(t_corr, inc.t), ev, "serve", 1.0, "either", serve=True,
                                      swing_id=self._swing_id)
            return "serve_hit"
        stroke, conf, why = self._judge(ev, inc.required)
        if rec is not None:
            rec["call"] = {"stroke": stroke, "conf": conf, "source": why}
        inc.resolved = True
        if not stroke_ok(inc.required, stroke):
            self._player_miss("WRONG STROKE", needed=inc.required, judged=stroke, confidence=conf, source=why,
                              swing_id=self._swing_id)
            return "wrong_stroke"
        self.pending = PendingHit(max(t_corr, inc.t), ev, stroke, conf, inc.required, swing_id=self._swing_id)
        return "hit"

    def _swing_record(self, ev, t: float, t_corr: float) -> dict:
        """Everything known about a swing at the moment it registers (before the game acts on it)."""
        rec = {"id": self._swing_id, "t": ev.t_peak, "t_contact": ev.t_hit, "sim_t": t, "phase": self.sm.phase,
               "serve_state": self.serve_state, "handedness": self.handedness, "speed": self.speed_setting,
               **ev.summary(), "features": ev.features}
        v = self.vision
        if v is not None and getattr(v, "judge", None) is not None:
            rec["camera"] = list(v.judge.judge(ev.t_peak))     # (stroke or None, vote share)
            rec["camera_on"] = bool(v.stroke_check_enabled)
        inc = self.incoming
        if inc is not None and not inc.resolved:
            rec["ball"] = {"required": inc.required, "serve": inc.serve, "pos": inc.pos,
                           "arrival_t": inc.t + self.clock_offset, "dt": t_corr - inc.t,
                           "early_s": inc.early_s, "late_s": inc.late_s, "vel": self.ball.vel}
        return rec

    def _judge(self, ev, required: str):
        """(stroke, confidence, which source decided)."""
        if self.vision is None or not self.vision.stroke_check_enabled:
            if ev.stroke_hint:
                return ev.stroke_hint, 1.0, "keyboard"
            # No pose check: always the right stroke.
            return ("backhand" if required == "backhand" else "forehand"), 1.0, "no camera"
        imu = (ev.imu_stroke, ev.imu_stroke_conf) if C.STROKE_USE_IMU and ev.imu_stroke else None
        return decide_stroke(required, self.vision.judge.judge(ev.t_peak), imu)

    def _ball_pos_at(self, t: float) -> P.Vec:
        inc = self.incoming
        if inc is not None and t <= inc.t + 1e-6:
            return inc.pos
        if not self.history:
            return self.ball.pos
        return min(self.history, key=lambda h: abs(h[0] - t))[1]

    def _execute_player_hit(self) -> None:
        h = self.pending
        held = self.hold
        self.hold = None
        pos = self.ball.pos if held else self._ball_pos_at(h.t)     # held: right where the paddle caught it
        pos = (pos[0], max(pos[1], 0.06), pos[2])
        sw = h.swing
        if h.serve:
            speed = serve_speed(sw.strength)
            shot = P.serve_shot(pos, speed, sw.topspin, sw.sidespin, "player", self.rng)
            self.serve_state = None
        else:
            speed = sw.return_speed
            shot = P.player_return(pos, speed, sw.topspin, sw.sidespin, self.rng)
        self.pending = None
        self.incoming = None
        self.player_strength = sw.strength
        ball_in = self.ball
        self.ball = P.Ball(pos, shot.vel, shot.spin)
        self.flight = Flight("player", "cpu", self.sim_t, serve=h.serve)
        # Contact happened at h.t; catch the ball up to the present -- unless it was
        # held at the paddle, in which case it leaves from the paddle right now.
        lag = 0.0 if held else self.sim_t - h.t
        for _ in range(int(lag / P.DT)):
            for ev in P.step(self.ball, P.DT, self.rng):
                self._referee(ev)
            if self.sm.phase != SM.RALLY:
                return
        self._last_hit_id = h.swing_id
        self._record("player_shot", swing_id=h.swing_id, pos=pos, serve=h.serve, stroke=h.stroke,
                     required=h.required, held=bool(held), ball_vel=ball_in.vel, ball_spin=ball_in.spin,
                     vel=shot.vel, spin=shot.spin, speed=speed)
        self.opponent.on_incoming(self.ball, self.sim_t, serve=h.serve)
        self.paddle.haptic()
        self._emit("hit", who="player", pos=pos, speed=speed, strength=sw.strength, serve=h.serve,
                   topspin=sw.topspin, sidespin=sw.sidespin, stroke=h.stroke,
                   confidence=h.confidence, required=h.required)

    # ======================================================================
    # Latency calibration
    # ======================================================================
    def start_latency_cal(self) -> None:
        if self.sm.phase not in (SM.LOBBY, SM.GAME_OVER):
            return
        period = C.LATENCY_BEAT_PERIOD_S
        self.latency = {"beats": [self.sim_t + 2.0 + i * period for i in range(C.LATENCY_BEATS)],
                        "offsets": []}
        self.sm.to(SM.LATENCY_CAL, self.sim_t)

    def _latency_swing(self, t: float) -> None:
        beats = self.latency["beats"]
        nearest = min(beats, key=lambda b: abs(b - t))
        if abs(t - nearest) < C.LATENCY_BEAT_PERIOD_S / 2:
            self.latency["offsets"].append(t - nearest)

    def _finish_latency_cal(self) -> None:
        offsets = self.latency["offsets"]
        if len(offsets) >= 5:
            med = statistics.median(offsets)
            C.LATENCY_OFFSET_S = med
            with open(os.path.join(C.HERE, C.LATENCY_CALIBRATION_FILE), "w") as f:
                json.dump({"latency_offset_s": med, "samples": offsets}, f, indent=2)
            text = f"Latency offset {med * 1000:+.0f} ms saved"
        else:
            text = f"Only {len(offsets)} swings matched a beat - not saved"
        self._emit("message", text=text)
        self.latency = None
        self.sm.to(SM.LOBBY, self.sim_t)

    # ======================================================================
    # Tags and commands
    # ======================================================================
    def _poll_tags(self) -> None:
        v = self.vision
        if v is None:
            return
        v.tags_active = self.sm.tags_active and not self.sm.paused
        lead, progress = v.tag_lead, round(v.tag_progress, 1)
        if (lead, progress) != self._last_tag:
            self._last_tag = (lead, progress)
            self._emit("tag_progress", id=lead, label=C.TAG_IDS.get(lead), progress=progress)
        while not v.confirmed_tags.empty():
            tag_id = v.confirmed_tags.get_nowait()
            if self.sm.tags_active and tag_id in C.TAG_IDS:
                self._emit("tag_confirmed", id=tag_id, label=C.TAG_IDS[tag_id])
                self.start_game(C.TAG_IDS[tag_id])

    def _drain_commands(self) -> None:
        while True:
            try:
                msg = self.commands.get_nowait()
            except queue.Empty:
                return
            if msg.get("type") == "key":
                self._on_key(str(msg.get("key", "")), bool(msg.get("down", True)), bool(msg.get("shift", False)))
            elif msg.get("type") == "set" and "handedness" in msg:
                self.set_handedness(str(msg["handedness"]))
                self._emit("message", text=f"{self.handedness.upper()}-HANDED")

    def _on_key(self, key: str, down: bool, shift: bool) -> None:
        sim = self.paddle.kind == "sim"
        self.paddle.handle_key(key, down, shift)
        if not down:
            return
        k = key.lower()
        if k == "p":
            self.sm.paused = not self.sm.paused
        elif k == "r":
            self.reset()
        elif k == "z":
            arm = getattr(self.vision, "arm", None)
            if arm is not None:
                arm.zero()                          # the arm's ready position too
            ok = self.paddle.zero()
            if ok:
                self._record_paddle_state("zero")
            self._emit("message", text="Paddle zeroed" if ok else "Zero failed - is the paddle connected?")
        elif k == "g" or (k == "d" and not sim):   # in sim mode D is sidespin
            self.debug = not self.debug
        elif k == "h":
            self.show_hint = not self.show_hint
        elif k in ("1", "2", "3"):
            self.start_game(("slow", "medium", "fast")[int(k) - 1])
        elif k == "l":
            self.start_latency_cal()
        elif k == "x":
            self._dispute()

    def _dispute(self) -> None:
        """X: the player says the last miss call was wrong. Logged as ground truth for
        tools/game_report.py (how often each kind of call is disputed)."""
        if self.recorder is None or not self.recorder.active:
            self._emit("message", text="Not recording -- nothing to mark")
            return
        if self._last_call is None:
            self._emit("message", text="No call to mark yet")
            return
        self._record("dispute", **self._last_call)
        self._emit("message", text=f"Marked: {self._last_call['reason']} was the wrong call")
        self._last_call = None

    # ======================================================================
    # Output
    # ======================================================================
    def _emit(self, name: str, **fields) -> None:
        self._publish_event(protocol.event_message(name, **fields))
        self._record(name, **fields)

    def _publish_score(self, event: str, **fields) -> None:
        """The score, for MQTT: after every point and at the start of a game."""
        m = self.match.snapshot()
        score = {"event": event, "player": m["player"], "cpu": m["cpu"],
                 "games_player": m["games_player"], "games_cpu": m["games_cpu"],
                 "speed": self.speed_setting, "streak": self.streak.current, "record": self.streak.record}
        score.update(fields)
        self._on_score(score)

    def _record_paddle_state(self, name: str) -> None:
        """The paddle's zeroed ready pose, so a recording can be replayed exactly
        (tools/calibrate_swing.py re-detects game swings from imu.csv)."""
        cal = self.paddle.calibration
        self._record(name, neutral=cal.neutral, gravity=cal.gravity, gyro_bias=cal.gyro_bias,
                     accel_raw_per_g=cal.accel_raw_per_g, noise_g=cal.noise_g, noise_dps=cal.noise_dps)

    def _record(self, name: str, **fields) -> None:
        """Log an event to the game recording only (not sent to the browser)."""
        if self.recorder is not None:
            self.recorder.event(name, self.clock(), self.sim_t, fields)

    def snapshot(self) -> dict:
        ph = self.sm.phase
        inc = self.incoming if (self.incoming and not self.incoming.resolved) else None
        v = self.vision
        status = {
            "paddle": self.paddle.status,
            "paddle_kind": self.paddle.kind,
            "calibration": self.paddle.calibration.source,
            "camera": v.camera.status if v else "off",
            "pose": bool(v and v.pose_detected),
            "stroke_check": bool(v and v.stroke_check_enabled),
            "pose_label": v.prediction if v else None,
            "mqtt": self._mqtt_status(),
        }
        tag = None
        if v and v.tag_lead is not None and v.tag_progress > 0:
            tag = {"id": v.tag_lead, "label": C.TAG_IDS.get(v.tag_lead), "progress": v.tag_progress}
        latency = None
        if self.latency:
            latency = {"beats": self.latency["beats"], "count": len(self.latency["offsets"])}
        debug = None
        if self.debug:
            plan = self.opponent.plan
            debug = {
                "arrival": list(inc.pos) if inc else None,
                "t_to_arrival": (inc.t - self.sim_t) if inc else None,
                "window_open": bool(inc and -inc.early_s <= self.sim_t - inc.t <= inc.late_s),
                "window": [inc.early_s, inc.late_s] if inc else None,
                "cpu_plan": list(plan.pos) if plan else None,
                "speed": P.v_norm(self.ball.vel),
                "topspin": P.topspin_component(self.ball.vel, self.ball.spin),
                "hit_plane_z": C.PLAYER_HIT_PLANE_Z,

                "latency_offset_s": C.LATENCY_OFFSET_S,
                "toss": getattr(getattr(self.paddle, "toss", None), "last", None),
            }
        return dict(
            t=self.sim_t,
            phase=ph,
            paused=self.sm.paused,
            ball={"pos": list(self.ball.pos), "vel": list(self.ball.vel), "spin": list(self.ball.spin),
                  "visible": ph in (SM.SERVE, SM.RALLY, SM.POINT_OVER)},
            opponent=self.opponent.snapshot(self.sim_t),
            paddle=self.paddle.orientation(),
            stroke=getattr(self.paddle, "stroke_state", lambda: None)(),
            hold=self.hold is not None,           # hit-stop: the ball is waiting at the paddle
            arm=v.arm.state(time.monotonic()) if (v is not None and C.POSE_ARM and getattr(v, "arm", None))
            else None,
            required_stroke=inc.required if inc else None,
            incoming={"x": inc.pos[0], "t_to_arrival": inc.t - self.sim_t, "required": inc.required,
                      "window": [inc.early_s, inc.late_s]}
            if inc else None,
            # Where/when the player's paddle should meet the ball -- kept until the
            # ball is actually struck (incoming above disappears once a swing is accepted).
            contact={"pos": list(self.incoming.pos), "t_to_contact": self.incoming.t - self.sim_t,
                     "serve": self.incoming.serve, "swung": self.incoming.resolved}
            if self.incoming else None,
            score=self.match.snapshot(),
            streak=self.streak.snapshot(),
            serve={"server": self.match.server, "state": self.serve_state,
                   "paddle_kind": self.paddle.kind},
            speed_setting=self.speed_setting,
            status=status,
            tag=tag,
            latency=latency,
            debug=debug,
            settings={"hint": self.show_hint, "handedness": self.handedness, "debug": self.debug,
                      },
        )

    def publish(self) -> None:
        self._publish_state(protocol.state_message(**self.snapshot()))
