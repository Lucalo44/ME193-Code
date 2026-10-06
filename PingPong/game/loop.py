"""
game/loop.py -- the fixed-timestep game loop that combines every input.

Inputs:   paddle swing events (real or sim), vision results (tags + pose
          judge), key commands from the browser/terminal.
Outputs:  state snapshots and discrete events, as JSON strings, to whatever
          publish functions were passed in (the WebSocket server in main.py).

Time: `sim_t` is the game clock (seconds, frozen while paused), advanced in
fixed physics sub-steps of 1/PHYSICS_HZ. Swing and camera timestamps are
time.monotonic(); `clock_offset` converts them (sim = mono - clock_offset).

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
from game.rules import Match, other, required_stroke, stroke_ok
from game.streak import StreakTracker
from server import protocol

SERVE_HEIGHT = 0.22


@dataclass
class Flight:
    hitter: str
    receiver: str
    t0: float
    bounces: int = 0              # bounces on the receiver's half
    net: bool = False
    struck: bool = False          # CPU receiver already swung at it


@dataclass
class Incoming:
    t: float                      # sim time the ball reaches the player's hit plane
    pos: P.Vec
    required: str                 # "forehand" | "backhand" | "either"
    resolved: bool = False


@dataclass
class PendingHit:
    t: float                      # sim time of contact
    swing: object                 # hardware.swing.SwingEvent
    stroke: Optional[str]
    confidence: float
    required: str


class Game:
    def __init__(self, paddle, vision=None,
                 publish_state: Optional[Callable[[str], None]] = None,
                 publish_event: Optional[Callable[[str], None]] = None,
                 debug: bool = False, seed: Optional[int] = None,
                 clock: Callable[[], float] = time.monotonic,
                 streak: Optional[StreakTracker] = None, mqtt_status: Callable[[], str] = lambda: "off"):
        self.paddle = paddle
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
        self._mqtt_status = mqtt_status

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
        self._drain_swings(ignore=paused)
        self._poll_tags()
        if not paused:
            self.opponent.update(dt, self.sim_t)
            self._phase_logic()

    # ======================================================================
    # Phases
    # ======================================================================
    def _phase_logic(self) -> None:
        ph, now = self.sm.phase, self.sim_t
        elapsed = self.sm.elapsed(now)
        if ph == SM.SERVE:
            self.ball = P.Ball((self.opponent.x, SERVE_HEIGHT, C.OPPONENT_HIT_PLANE_Z))
            if elapsed >= C.SERVE_DELAY_S:
                start, shot = self.opponent.serve(now)
                self.ball = P.Ball(start, shot.vel, shot.spin)
                self.sm.to(SM.RALLY, now)
                self._begin_flight("cpu")
                self._emit("serve", pos=start)
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
        self._clear_rally()
        self.sm.to(SM.SERVE, self.sim_t)
        self._emit("message", text=f"{setting.upper()} - game on!")

    def reset(self) -> None:
        self.streak.reset_streak()
        self.sm.reset(self.sim_t)
        self._clear_rally()
        self.ball = P.Ball(dead=True)
        self.opponent.reset()
        self.latency = None

    def _clear_rally(self) -> None:
        self.flight = None
        self.incoming = None
        self.pending = None

    # ======================================================================
    # Physics sub-step + referee
    # ======================================================================
    def _substep(self) -> None:
        ph = self.sm.phase
        if ph not in (SM.RALLY, SM.POINT_OVER):
            return
        if ph == SM.RALLY and self.pending and self.sim_t >= self.pending.t:
            self._execute_player_hit()
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
                self._predict_incoming()
        elif ev.kind == "bounce":
            if ev.side == f.hitter:
                self._end_point(f.receiver, "NET" if f.net else "OWN SIDE")
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
            self.ball = P.Ball(b.pos, shot.vel, shot.spin)
            self._begin_flight("cpu")
            self._emit("hit", who="cpu", pos=b.pos, speed=P.v_norm(shot.vel))

    def _check_player_expiry(self) -> None:
        inc = self.incoming
        if inc and not inc.resolved and self.pending is None \
                and self.sim_t > inc.t + C.HIT_WINDOW_S + C.LATE_ZONE_S:
            inc.resolved = True
            self._player_miss("NO SWING")

    def _begin_flight(self, hitter: str) -> None:
        self.flight = Flight(hitter, other(hitter), self.sim_t)
        if hitter == "cpu":
            self._predict_incoming()
        else:
            self.incoming = None
            self.opponent.on_incoming(self.ball, self.sim_t)

    def _predict_incoming(self, bounced: bool = False) -> None:
        arr = P.predict_receive(self.ball, "player", C.PLAYER_HIT_PLANE_Z, bounced=bounced)
        if arr is None:
            self.incoming = None
            return
        self.incoming = Incoming(self.sim_t + arr.t, arr.pos, required_stroke(arr.pos[0]))

    def _count_hit(self) -> None:
        """One continuous hit: the player's return landed in. Tracks the record."""
        if self.streak.hit():
            self._emit("record", record=self.streak.record)

    def _end_point(self, winner: str, reason: str, game_penalty: bool = False, **extra) -> None:
        self.streak.reset_streak()
        if winner == "cpu":
            self._emit("miss", reason=reason, **extra)
        result = self.match.award_game(winner) if game_penalty else self.match.award_point(winner)
        self.point_result = result
        self._emit("point", winner=winner, reason=reason, score=self.match.snapshot())
        if result.game_over:
            self._emit("game_over", match_over=result.match_over, winner=result.game_winner,
                       match_winner=result.match_winner, score=self.match.snapshot())
        self._clear_rally()
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

    def _on_swing(self, ev) -> None:
        t = ev.t_peak - self.clock_offset          # sim time of the swing peak
        t_corr = t - C.LATENCY_OFFSET_S
        self._emit("swing", trace=ev.trace(self.paddle.calibration), **ev.summary())
        if self.sm.phase == SM.LATENCY_CAL:
            self._latency_swing(t)
            return
        inc = self.incoming
        if self.sm.phase != SM.RALLY or inc is None or inc.resolved or self.pending:
            return
        dt = t_corr - inc.t
        if dt < -C.EARLY_ZONE_S or dt > C.HIT_WINDOW_S + C.LATE_ZONE_S:
            return  # stray swing, nowhere near the ball
        if dt < -C.HIT_WINDOW_S:
            inc.resolved = True
            self._player_miss("EARLY", by_s=-dt)
            return
        if dt > C.HIT_WINDOW_S:
            inc.resolved = True
            self._player_miss("LATE", by_s=dt)
            return
        stroke, conf = self._judge(ev, inc.required)
        inc.resolved = True
        if not stroke_ok(inc.required, stroke):
            self._player_miss("WRONG STROKE", needed=inc.required, judged=stroke, confidence=conf)
            return
        self.pending = PendingHit(max(t_corr, inc.t), ev, stroke, conf, inc.required)

    def _judge(self, ev, required: str):
        if self.vision is None or not self.vision.stroke_check_enabled:
            if ev.stroke_hint:
                return ev.stroke_hint, 1.0
            # No pose check: always the right stroke.
            return ("backhand" if required == "backhand" else "forehand"), 1.0
        return self.vision.judge.judge(ev.t_peak)

    def _ball_pos_at(self, t: float) -> P.Vec:
        inc = self.incoming
        if inc is not None and t <= inc.t + 1e-6:
            return inc.pos
        if not self.history:
            return self.ball.pos
        return min(self.history, key=lambda h: abs(h[0] - t))[1]

    def _execute_player_hit(self) -> None:
        h = self.pending
        pos = self._ball_pos_at(h.t)
        pos = (pos[0], max(pos[1], 0.06), pos[2])
        sw = h.swing
        shot = P.player_return(pos, sw.return_speed, sw.topspin, sw.sidespin, self.rng)
        self.pending = None
        self.incoming = None
        self.player_strength = sw.strength
        self.ball = P.Ball(pos, shot.vel, shot.spin)
        self.flight = Flight("player", "cpu", self.sim_t)
        # Contact happened at h.t; catch the ball up to the present.
        for _ in range(int((self.sim_t - h.t) / P.DT)):
            for ev in P.step(self.ball, P.DT, self.rng):
                self._referee(ev)
            if self.sm.phase != SM.RALLY:
                return
        self.opponent.on_incoming(self.ball, self.sim_t)
        self.paddle.haptic()
        self._emit("hit", who="player", pos=pos, speed=sw.return_speed, strength=sw.strength,
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
            ok = self.paddle.zero()
            self._emit("message", text="Paddle zeroed" if ok else "Zero failed - is the paddle connected?")
        elif k == "g" or (k == "d" and not sim):   # in sim mode D is sidespin
            self.debug = not self.debug
        elif k == "h":
            self.show_hint = not self.show_hint
        elif k in ("1", "2", "3"):
            self.start_game(("slow", "medium", "fast")[int(k) - 1])
        elif k == "l":
            self.start_latency_cal()

    # ======================================================================
    # Output
    # ======================================================================
    def _emit(self, name: str, **fields) -> None:
        self._publish_event(protocol.event_message(name, **fields))

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
                "window_open": bool(inc and abs(self.sim_t - inc.t) <= C.HIT_WINDOW_S),
                "cpu_plan": list(plan.pos) if plan else None,
                "speed": P.v_norm(self.ball.vel),
                "topspin": P.topspin_component(self.ball.vel, self.ball.spin),
                "hit_plane_z": C.PLAYER_HIT_PLANE_Z,
                "hit_window_s": C.HIT_WINDOW_S,
                "latency_offset_s": C.LATENCY_OFFSET_S,
            }
        return dict(
            t=self.sim_t,
            phase=ph,
            paused=self.sm.paused,
            ball={"pos": list(self.ball.pos), "vel": list(self.ball.vel), "spin": list(self.ball.spin),
                  "visible": ph in (SM.SERVE, SM.RALLY, SM.POINT_OVER)},
            opponent=self.opponent.snapshot(self.sim_t),
            paddle=self.paddle.orientation(),
            required_stroke=inc.required if inc else None,
            incoming={"x": inc.pos[0], "t_to_arrival": inc.t - self.sim_t, "required": inc.required}
            if inc else None,
            score=self.match.snapshot(),
            streak=self.streak.snapshot(),
            speed_setting=self.speed_setting,
            status=status,
            tag=tag,
            latency=latency,
            debug=debug,
            settings={"hint": self.show_hint, "handedness": C.HANDEDNESS, "debug": self.debug,
                      "hit_window_s": C.HIT_WINDOW_S},
        )

    def publish(self) -> None:
        self._publish_state(protocol.state_message(**self.snapshot()))
