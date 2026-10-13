"""
game/style.py -- an opponent that plays like a recorded player.

build_profile() turns a player's recorded games (game_logs/<player>/, see
game/recorder.py) into a style profile; StyleOpponent plays from it.

What a profile holds, all measured from the player's real games:
  * returns -- every return they hit: the ball it answered (speed, and
    whether it needed their forehand or backhand), and what they did with it:
    stroke, shot speed, topspin and sidespin (-1..1, as the paddle read
    them), and where it landed or how it failed (NET / OUT / OWN SIDE).
  * serves -- speed, spin and result of every serve.
  * misses -- per stroke side and incoming-speed band, how many balls they
    faced and how many they whiffed outright (NO SWING / LATE). Calls the
    player marked wrong with X, and WRONG STROKE calls (a rule, not a miss),
    don't count.

Only play with the player's own hand picked in the lobby is used, and every
sample is weighted by its session's age (GAME_DATA_RECENCY): recent games
count a little more, both in which shots get copied and in the miss rates.

How the opponent uses it: when a ball comes, it picks one of the player's
real returns to a similar ball (same stroke side, weighted toward a similar
speed) and copies it -- same speed and spin, aimed where that shot landed,
turned around to the other end of the table (a 180-degree turn keeps the
player's own left and right, and their handedness). If that return went into
the net or long, so does this one. It whiffs as often as the player did on
that kind of ball, and serves with one of their serves' speed and spin.

Where returns land is partly the game's doing (the player's aim is randomized
around the table's middle and pushed by their sidespin), so placement follows
the player only as much as the game let it; speed, spin, stroke mix, errors and
misses are theirs.
"""

from __future__ import annotations

import glob
import json
import math
import os
import random
import time
from typing import Dict, List, Optional

import config as C
from game import physics as P
from game.opponent import Opponent

SPEED_BANDS = (3.3, 4.2)          # m/s: incoming balls are slow / medium / fast around these


def _band(speed: float) -> int:
    return sum(speed > b for b in SPEED_BANDS)


def _jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------
# Building a profile from recorded games
# --------------------------------------------------------------------------

def player_hand(player: str, sessions_meta: List[dict]) -> str:
    """The hand the player plays with: their calibration tag (imu_logs/<player>/player.json),
    else what their sessions were recorded with."""
    try:
        with open(os.path.join(C.HERE, "imu_logs", player, "player.json")) as f:
            hand = json.load(f).get("hand")
        if hand in ("left", "right"):
            return hand
    except (OSError, ValueError):
        pass
    hands = [m["hand"] for m in sessions_meta if m.get("hand") in ("left", "right")]
    return max(set(hands), key=hands.count) if hands else "right"


def build_profile(player: str, log_root: Optional[str] = None) -> Optional[dict]:
    """The style profile of `player` from every recorded session, or None if
    there are no recorded shots. Only play with the player's own hand picked in
    the lobby counts (with the other hand, forehand and backhand swap), and each
    sample carries its session's weight `w` (session_weights: recent games a
    little more)."""
    from game.recorder import hand_timeline, session_weights
    root = os.path.join(log_root or os.path.join(C.HERE, C.GAME_LOG_DIR), player)
    sessions = sorted(p for p in glob.glob(os.path.join(root, "*")) if os.path.exists(os.path.join(p, "session.json")))
    metas = []
    for path in sessions:
        with open(os.path.join(path, "session.json")) as f:
            metas.append(json.load(f))
    hand = player_hand(player, metas)
    returns, serves = [], []
    faced: Dict[str, List[float]] = {}      # "stroke/band" -> [balls, whiffs], weighted
    skipped = 0

    def count(key: str, w: float, whiff: bool) -> None:
        cell = faced.setdefault(key, [0.0, 0.0])
        cell[0] = round(cell[0] + w, 4)
        cell[1] = round(cell[1] + (w if whiff else 0.0), 4)

    for path, weight in zip(sessions, session_weights(len(sessions))):
        swings = {sw["id"]: sw for sw in _jsonl(os.path.join(path, "swings.jsonl"))}
        events = _jsonl(os.path.join(path, "events.jsonl"))
        hand_at = hand_timeline(events, list(swings.values()))
        results = {e["swing_id"]: e for e in events if e["name"] == "shot_result"}
        disputed = [e["sim_t"] for e in events if e["name"] == "dispute"]

        def was_disputed(sim_t: float) -> bool:
            return any(abs(sim_t - d) < 0.1 for d in disputed)

        for e in events:
            if e["name"] != "player_shot" or e.get("swing_id") not in swings:
                continue
            sw = swings[e["swing_id"]]
            if (sw.get("handedness") or hand_at(e["sim_t"])) != hand:
                skipped += 1
                continue
            res = results.get(e["swing_id"], {})
            shot = {"speed": round(e["speed"], 2), "topspin": sw.get("topspin", 0.0),
                    "sidespin": sw.get("sidespin", 0.0), "result": res.get("result", "unknown"),
                    "land": [round(v, 3) for v in res["pos"][::2]] if res.get("result") == "in" else None,
                    "w": round(weight, 4)}
            if e.get("serve"):
                serves.append(shot)
                continue
            ball = sw.get("ball") or {}
            in_speed = P.v_norm(e["ball_vel"]) if e.get("ball_vel") else P.v_norm(ball.get("vel") or (0, 0, 0))
            stroke = ball.get("required") if ball.get("required") in ("forehand", "backhand") else e.get("stroke")
            shot.update({"in_speed": round(in_speed, 2), "stroke": stroke})
            returns.append(shot)
            count(f"{stroke}/{_band(in_speed)}", weight, False)
        # Balls they faced but didn't return: whiffs (NO SWING, LATE) count, unless disputed.
        for e in events:
            if e["name"] == "missed_ball" and not e.get("serve") and not was_disputed(e["sim_t"]) \
                    and hand_at(e["sim_t"]) == hand:
                count(f"{e['required']}/{_band(P.v_norm(e['ball_vel']))}", weight, True)
        for sw in swings.values():
            ball = sw.get("ball") or {}
            if sw.get("outcome") == "late" and not ball.get("serve") and not was_disputed(sw["sim_t"]) \
                    and sw.get("handedness") == hand:
                count(f"{ball.get('required')}/{_band(P.v_norm(ball.get('vel') or (0, 0, 0)))}", weight, True)
    if not returns:
        return None
    return {"player": player, "hand": hand,
            "built": time.strftime("%Y-%m-%d %H:%M:%S"), "sessions": len(sessions),
            "recency": C.GAME_DATA_RECENCY, "skipped_other_hand": skipped,
            "returns": returns, "serves": serves, "faced": faced}


def profile_path(player: str) -> str:
    return os.path.join(C.HERE, C.STYLE_DIR, f"{player}.json")


def load_profile(player: str) -> Optional[dict]:
    """The freshest profile: rebuilt from the player's recorded games when they're
    here (and saved to styles/ for sharing), else the saved styles/<player>.json."""
    prof = build_profile(player)
    if prof is not None:
        save_profile(prof)
        return prof
    path = profile_path(player)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_profile(prof: dict) -> str:
    path = profile_path(prof["player"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(prof, f, indent=1)
    return path


def summarize(prof: dict) -> List[str]:
    """A few lines describing the style, for the terminal."""
    r = prof["returns"]
    n_in = sum(s["result"] == "in" for s in r)
    fh = sum(s["stroke"] == "forehand" for s in r)
    speeds = sorted(s["speed"] for s in r)
    top = sum(s["topspin"] > 0.2 for s in r)
    back = sum(s["topspin"] < -0.2 for s in r)
    balls = sum(v[0] for v in prof["faced"].values())
    whiffs = sum(v[1] for v in prof["faced"].values())
    return [
        f"{prof['player']}: {prof['hand']}-handed, {len(r)} returns and {len(prof['serves'])} serves "
        f"from {prof['sessions']} session(s)",
        f"  returns land in {n_in / len(r):.0%}; forehand {fh / len(r):.0%}; "
        f"speed {speeds[len(speeds) // 2]:.1f} m/s (typical {speeds[len(speeds) // 10]:.1f}-{speeds[9 * len(speeds) // 10]:.1f})",
        f"  topspin {top / len(r):.0%}, backspin {back / len(r):.0%}; whiffs {whiffs:.0f}/{balls:.0f} balls "
        f"(weighted, recent sessions count more)",
    ]


# --------------------------------------------------------------------------
# The opponent
# --------------------------------------------------------------------------

class StyleOpponent(Opponent):
    """Plays like the player in `profile` (see the module docstring)."""

    def __init__(self, rng: random.Random, profile: dict):
        super().__init__(rng)
        self.profile = profile
        self.hand = profile.get("hand", "right")
        self._whiff = False

    # Strokes: a right-hander facing the player has its forehand on world -x
    # (game/opponent.py); a left-hander on +x.
    def _stroke_for(self, ball_x: float) -> str:
        fh_side = ball_x < self.x if self.hand == "right" else ball_x > self.x
        return "forehand" if fh_side else "backhand"

    def start_swing(self, ball_x: float, now: float) -> None:
        self.swing = self._stroke_for(ball_x)
        self.swing_started = now

    def on_incoming(self, ball: P.Ball, now: float, serve: bool = False) -> None:
        super().on_incoming(ball, now, serve=serve)
        self.will_miss = False                       # errors come from the sampled return instead
        self._whiff = False
        if self.plan is None:
            return
        stroke = self._stroke_for(self.plan.pos[0])
        balls, whiffs = self.profile["faced"].get(f"{stroke}/{_band(P.v_norm(self.plan.vel))}", [0, 0])
        p = (whiffs + 0.5) / (balls + 5.0)            # smoothed: few balls -> close to a small default
        self._whiff = self.rng.random() < p * C.STYLE_MISS_SCALE

    def _sample(self, pool: List[dict], in_speed: Optional[float] = None) -> Optional[dict]:
        if not pool:
            return None
        if in_speed is None:
            return self.rng.choices(pool, weights=[s.get("w", 1.0) for s in pool], k=1)[0]
        weights = [s.get("w", 1.0) * (math.exp(-((s.get("in_speed", in_speed) - in_speed) / 0.8) ** 2) + 1e-3)
                   for s in pool]
        return self.rng.choices(pool, weights=weights, k=1)[0]

    def strike(self, ball: P.Ball, now: float, player_strength: float = 0.0) -> Optional[P.Shot]:
        if now - self.swing_started > 0.4:
            self.start_swing(ball.pos[0], now)
        if abs(ball.pos[0] - self.x) > C.OPPONENT_REACH or self._whiff:
            return None
        stroke = self._stroke_for(ball.pos[0])
        pool = [s for s in self.profile["returns"] if s["stroke"] == stroke] or self.profile["returns"]
        s = self._sample(pool, P.v_norm(ball.vel))
        top = s["topspin"] * C.SPIN_MAX_RADS
        side = s["sidespin"] * C.SIDESPIN_MAX_RADS
        speed = s["speed"]
        if s["result"] == "in" and s["land"]:
            # Turned around to this end of the table, kept a little inside the edges.
            x = max(-P.HALF_W + 0.08, min(P.HALF_W - 0.08, -s["land"][0] + self.rng.uniform(-0.05, 0.05)))
            z = max(-P.HALF_L + 0.1, min(-0.15, -s["land"][1] + self.rng.uniform(-0.05, 0.05)))
            return self._land(ball.pos, (x, z), speed, top, side)
        if s["result"] == "NET":
            return self._net_shot(ball.pos, speed, top, side)
        if s["result"] in ("OUT", "OWN SIDE"):
            x = self.rng.uniform(-0.5, 0.5)
            return P.aim(ball.pos, (x, -P.HALF_L - self.rng.uniform(0.15, 0.5)), speed, top, side)
        return self._good_shot(ball.pos, speed)

    @staticmethod
    def _land(start: P.Vec, target: tuple, speed: float, top: float, side: float) -> P.Shot:
        """Hit `target` with this speed and spin. The aim solver gets the length right,
        but sidespin curls the ball sideways, so re-aim against the curl; if the shot
        still misses the table (the player's own returns had the game's aim assist),
        take a little pace off, as the player effectively did."""
        best = None
        for factor in (1.0, 0.92, 0.85, 0.78, 0.7):
            aim_x = target[0]
            for _ in range(3):
                shot = P.aim(start, (aim_x, target[1]), speed * factor, top, side, spin_aware=True)
                land = P.plane_landing(P.Ball(start, shot.vel, shot.spin))
                if P.on_table(land.pos[0], land.pos[2]) and land.pos[2] < 0 and land.net_clearance > 0:
                    return shot
                aim_x += target[0] - land.pos[0]
            best = best or shot
        return best

    def _net_shot(self, start: P.Vec, speed: float, top: float, side: float) -> P.Shot:
        x, z = self._pick_target()
        shot = P.aim(start, (x, z), speed, top, side)
        heading = P.v_unit((shot.vel[0], 0.0, shot.vel[2]))
        vel = P.launch_velocity(speed, shot.theta - math.radians(self.rng.uniform(6, 10)), (heading[0], heading[2]))
        return P.Shot(vel, P.spin_vector(vel, top, side), shot.theta, (x, z))

    def _serve(self, start: P.Vec) -> P.Shot:
        good = [s for s in self.profile["serves"] if s["result"] == "in"] or self.profile["serves"]
        s = self._sample(good)
        if s is None:
            top, side = self._spin()
            return P.serve_shot(start, self.base_speed * C.CPU_SERVE_SPEED_FACTOR, top / C.SPIN_MAX_RADS,
                                side / C.SIDESPIN_MAX_RADS, "cpu", self.rng, assist=1.0)
        return P.serve_shot(start, s["speed"], s["topspin"], s["sidespin"], "cpu", self.rng, assist=1.0)

    def plan_serve(self) -> tuple:
        start = (self.target_x, C.SERVE_CONTACT_Y, C.OPPONENT_HIT_PLANE_Z)
        return start, self._serve(start)

    def serve(self, now: float, planned: tuple = None) -> tuple:
        if planned is not None:
            self.start_swing(planned[0][0] - 0.1, now)
            return planned
        start = (self.x, C.SERVE_CONTACT_Y, C.OPPONENT_HIT_PLANE_Z)
        self.start_swing(self.x - 0.1, now)
        return start, self._serve(start)
