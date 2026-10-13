#!/usr/bin/env python3
"""
tools/game_report.py -- a player's profile from their recorded games
(game_logs/<player>/, written by main.py with the real paddle).

    python tools/game_report.py                     # DEFAULT_PLAYER, every session
    python tools/game_report.py --player bri --last 3
    python tools/game_report.py --session 20261013-201500

It answers two questions:

  1. Are the game's calls right?  For every way of losing a point it shows how
     often it happened and what the recording says about it:
       NO SWING      was the paddle moving when the ball arrived? (a swing the
                     detector missed, or really no swing)
       LATE / EARLY  how far off the timing was -- a steady offset means the
                     latency calibration (L in the lobby) is off, not you
       WRONG STROKE  which source called it (camera / paddle), how sure, and
                     whether the other source disagreed
     plus every call you marked with X in the game ("that call was wrong").

  2. How do you play?  Hit rate by side and by ball speed, shot speed and
     spin, forehand/backhand mix and where your returns land -- the start of a
     model of your play style.

It also counts the game swings that are labelled reliably enough to train on
(calibrate_swing.py --with-games uses them).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import config as C  # noqa: E402
from hardware.swing import Calibration, gyro_dps, linear_g, load_csv  # noqa: E402

IN_WINDOW = ("hit", "late", "wrong_stroke", "serve_hit")
SHOT_FAULTS = ("NET", "OUT", "OWN SIDE", "SERVE FAULT")
OUTCOME_TEXT = {
    "hit": "hit the ball (right stroke)",
    "serve_hit": "struck the serve",
    "wrong_stroke": "WRONG STROKE called",
    "late": "LATE",
    "early": "too early (whiff, swing again)",
    "stray": "long after the ball",
    "no_ball": "no ball to hit",
    "awaiting_toss": "while waiting to toss",
    "toss_motion": "the toss itself",
    "latency_cal": "latency calibration",
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class Session:
    def __init__(self, path: str):
        self.path = path
        self.name = os.path.basename(path)
        with open(os.path.join(path, "session.json")) as f:
            self.meta = json.load(f)
        self.swings = _jsonl(os.path.join(path, "swings.jsonl"))
        self.events = _jsonl(os.path.join(path, "events.jsonl"))
        self._imu = None
        cal_path = os.path.join(path, "calibration.json")
        self.cal = Calibration.load(cal_path) if os.path.exists(cal_path) else Calibration()

    def imu(self):
        if self._imu is None:
            p = os.path.join(self.path, "imu.csv")
            self._imu = load_csv(p) if os.path.exists(p) else []
        return self._imu


def load_sessions(player: str, last: Optional[int], only: Optional[str]) -> List[Session]:
    root = os.path.join(HERE, C.GAME_LOG_DIR, player)
    paths = sorted(p for p in glob.glob(os.path.join(root, "*")) if os.path.exists(os.path.join(p, "session.json")))
    if only:
        paths = [p for p in paths if os.path.basename(p) == only]
    elif last:
        paths = paths[-last:]
    return [Session(p) for p in paths]


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def pct(n: int, d: int) -> str:
    return f"{100 * n / d:3.0f}%" if d else "  - "


def stats(vals: List[float], unit: str = "", fmt: str = "{:.2f}") -> str:
    if not vals:
        return "-"
    vals = sorted(vals)
    med = statistics.median(vals)
    lo, hi = vals[int(0.1 * (len(vals) - 1))], vals[int(0.9 * (len(vals) - 1))]
    return f"median {fmt.format(med)}{unit} (most between {fmt.format(lo)} and {fmt.format(hi)}{unit}, n={len(vals)})"


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def motion_near(session: Session, t: float, before: float = 0.6, after: float = 0.4):
    """Peak rotation (deg/s) and linear acceleration (g) of the paddle around time t."""
    cal = session.cal
    window = [s for s in session.imu() if t - before <= s.t <= t + after]
    if not window:
        return None
    return max(gyro_dps(s, cal) for s in window), max(linear_g(s, cal, cal.gravity) for s in window)


def confident_label(sw: dict) -> Optional[str]:
    """forehand/backhand when the swing's label is reliable: the side rule asked for it,
    the camera saw it surely, and the stroke was accepted. (The camera is independent
    of the paddle classifier, so training on these doesn't just echo the paddle.)"""
    ball, cam = sw.get("ball") or {}, sw.get("camera")
    req = ball.get("required")
    if sw.get("outcome") != "hit" or req not in ("forehand", "backhand") or not cam:
        return None
    return req if cam[0] == req and cam[1] >= C.STROKE_CAMERA_SURE else None


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------

def report(sessions: List[Session]) -> None:
    swings = [(s, sw) for s in sessions for sw in s.swings]
    events = [(s, e) for s in sessions for e in s.events]
    by_name: Dict[str, list] = defaultdict(list)
    for s, e in events:
        by_name[e["name"]].append((s, e))

    section("Sessions")
    for s in sessions:
        c = s.meta.get("counts", {})
        games = sum(1 for e in s.events if e["name"] == "game_start")
        print(f"  {s.name}  {s.meta.get('hand') or '?'}-handed, calibration {s.meta.get('calibration', '?')}, "
              f"{games} game(s), {c.get('swings', len(s.swings))} swings"
              + ("" if s.meta.get("ended") else "   (not closed cleanly)"))
    points = [e for _, e in by_name["point"]]
    won = sum(e["winner"] == "player" for e in points)
    print(f"  points: won {won}, lost {len(points) - won}")

    # -- swings ---------------------------------------------------------------
    section("What happened to your swings")
    oc = Counter(sw.get("outcome", "?") for _, sw in swings)
    for k, n in oc.most_common():
        print(f"  {n:4d}  {pct(n, len(swings))}  {OUTCOME_TEXT.get(k, k)}")
    results = {e["swing_id"]: e["result"] for _, e in by_name["shot_result"]}
    hits = [sw for _, sw in swings if sw.get("outcome") in ("hit", "serve_hit")]
    landed = sum(results.get(sw["id"]) == "in" for sw in hits)
    if hits:
        faults = Counter(results.get(sw["id"], "unfinished") for sw in hits if results.get(sw["id"]) != "in")
        print(f"  of {len(hits)} hits, {landed} landed in ({pct(landed, len(hits)).strip()})"
              + (": " + ", ".join(f"{k} {n}" for k, n in faults.most_common()) if faults else ""))

    # -- points lost ------------------------------------------------------------
    section("How you lost points -- and are those calls right?")
    lost = Counter(e["reason"] for e in points if e["winner"] == "cpu")
    disputed = Counter(e["reason"] for _, e in by_name["dispute"])
    for reason, n in lost.most_common():
        d = f"   you disputed {disputed[reason]}" if disputed[reason] else ""
        print(f"  {n:4d}  {reason}{d}")
    if not lost:
        print("  (none)")
    if by_name["dispute"]:
        print(f"  ({len(by_name['dispute'])} call(s) marked wrong with X in the game)")

    # NO SWING: was the paddle moving?
    missed = [(s, e) for s, e in by_name["missed_ball"] if not e.get("serve")]
    if missed:
        print("\n  NO SWING -- the paddle around the moment the ball arrived:")
        moved = still = unknown = 0
        for s, e in missed:
            m = motion_near(s, e["arrival_t"])
            if m is None:
                unknown += 1
            elif m[0] >= s.cal.gyro_threshold or m[1] >= s.cal.accel_threshold:
                moved += 1
            else:
                still += 1
        print(f"    {still:3d} paddle still -- really no swing")
        print(f"    {moved:3d} paddle moving past the swing thresholds but no swing registered"
              + ("   <-- likely missed by the detector (swing too short, or a backswing only)" if moved else ""))
        if unknown:
            print(f"    {unknown:3d} no paddle data")

    # Timing
    timed = [sw for _, sw in swings if sw.get("outcome") in IN_WINDOW + ("early",) and sw.get("ball")]
    if timed:
        dts = [sw["ball"]["dt"] for sw in timed if not sw["ball"].get("serve")]
        print("\n  Timing (swing peak minus the ball's arrival; + = late):")
        print(f"    rally swings  {stats(dts, ' s', '{:+.3f}')}")
        late = [sw["ball"]["dt"] - sw["ball"]["late_s"] for sw in timed if sw["outcome"] == "late"]
        if late:
            close = sum(x < 0.05 for x in late)
            print(f"    LATE calls: {len(late)}, {close} of them within 0.05 s of the window"
                  + ("   <-- borderline" if close > len(late) / 2 else ""))
        early = [sw for sw in timed if sw["outcome"] == "early"]
        if early:
            print(f"    early whiffs: {len(early)} (a second swing at the same ball is allowed)")
        if len(dts) >= 10 and abs(statistics.median(dts)) > 0.05:
            print(f"    Your swings land {statistics.median(dts):+.2f} s from the ball on average. If that feels "
                  "on time to you, redo the latency calibration (L in the lobby).")

    # WRONG STROKE
    wrong = [sw for _, sw in swings if sw.get("outcome") == "wrong_stroke"]
    if wrong:
        print("\n  WRONG STROKE calls:")
        src = Counter((sw.get("call") or {}).get("source", "?") for sw in wrong)
        print("    called by: " + ", ".join(f"{k} {n}" for k, n in src.most_common()))
        contested = 0
        for sw in wrong:
            req = sw["ball"]["required"]
            cam, imu = sw.get("camera"), sw.get("imu_stroke")
            if (cam and cam[0] == req) or imu == req:
                contested += 1
        print(f"    {contested} of {len(wrong)} had the other source seeing the stroke you needed -- "
              "those are the doubtful ones")

    # -- stroke readings ----------------------------------------------------------
    section("Stroke readings (forehand/backhand balls you swung at in time)")
    judged = [sw for _, sw in swings if sw.get("outcome") in ("hit", "wrong_stroke")
              and (sw.get("ball") or {}).get("required") in ("forehand", "backhand")]
    if judged:
        cam_seen = [sw for sw in judged if sw.get("camera") and sw["camera"][0]]
        cam_ok = sum(sw["camera"][0] == sw["ball"]["required"] for sw in cam_seen)
        imu_seen = [sw for sw in judged if sw.get("imu_stroke")]
        imu_ok = sum(sw["imu_stroke"] == sw["ball"]["required"] for sw in imu_seen)
        both = [sw for sw in judged if sw.get("camera") and sw["camera"][0] and sw.get("imu_stroke")]
        agree = sum(sw["camera"][0] == sw["imu_stroke"] for sw in both)
        print(f"  camera read the stroke the ball needed   {pct(cam_ok, len(cam_seen))}  "
              f"({len(judged) - len(cam_seen)} swings with no camera reading)")
        print(f"  paddle read the stroke the ball needed   {pct(imu_ok, len(imu_seen))}")
        print(f"  camera and paddle agreed with each other {pct(agree, len(both))}")
        print("  (A reading that differs from what the ball needed is either a misread or a real wrong stroke.)")
        for side in ("forehand", "backhand"):
            ss = [sw for sw in judged if sw["ball"]["required"] == side]
            c = [sw for sw in ss if sw.get("camera") and sw["camera"][0]]
            i = [sw for sw in ss if sw.get("imu_stroke")]
            print(f"    {side:9s}: camera {pct(sum(sw['camera'][0] == side for sw in c), len(c))}, "
                  f"paddle {pct(sum(sw['imu_stroke'] == side for sw in i), len(i))}   ({len(ss)} balls)")
    else:
        print("  (none yet)")

    # -- play style ----------------------------------------------------------------
    section("How you play")
    balls = [sw for _, sw in swings if sw.get("ball") and not sw["ball"].get("serve")
             and sw.get("outcome") in IN_WINDOW]
    misses_ball = [e for _, e in missed]
    for side in ("forehand", "backhand", "either"):
        tried = [sw for sw in balls if sw["ball"]["required"] == side]
        miss_n = sum(e["required"] == side for e in misses_ball)
        good = sum(results.get(sw["id"]) == "in" for sw in tried)
        total = len(tried) + miss_n
        if total:
            print(f"  {side:9s} balls: {total:3d}, returned in {pct(good, total)}")
    speeds = []
    for sw in balls:
        v = sw["ball"].get("vel")
        if v:
            speeds.append((sum(x * x for x in v) ** 0.5, results.get(sw["id"]) == "in"))
    if len(speeds) >= 6:
        speeds.sort()
        third = len(speeds) // 3
        for label, chunk in (("slowest third", speeds[:third]), ("middle third", speeds[third:2 * third]),
                             ("fastest third", speeds[2 * third:])):
            print(f"  {label} of balls ({chunk[0][0]:.1f}-{chunk[-1][0]:.1f} m/s): "
                  f"returned in {pct(sum(ok for _, ok in chunk), len(chunk))}")
    shots = [e for _, e in by_name["player_shot"] if not e.get("serve")]
    if shots:
        print(f"  shot speed   {stats([e['speed'] for e in shots], ' m/s')}")
        hit_sw = {sw["id"]: sw for _, sw in swings}
        tops = [hit_sw[e["swing_id"]]["topspin"] for e in shots if e.get("swing_id") in hit_sw]
        sides = [hit_sw[e["swing_id"]]["sidespin"] for e in shots if e.get("swing_id") in hit_sw]
        if tops:
            print(f"  spin         topspin {pct(sum(t > 0.2 for t in tops), len(tops))}, "
                  f"backspin {pct(sum(t < -0.2 for t in tops), len(tops))}, "
                  f"flat {pct(sum(abs(t) <= 0.2 for t in tops), len(tops))};  "
                  f"sidespin {pct(sum(abs(x) > 0.3 for x in sides), len(sides))} of shots")
        strokes = Counter(e.get("stroke") for e in shots)
        print("  strokes      " + ", ".join(f"{k} {n}" for k, n in strokes.most_common()))
    landings = [e["pos"] for _, e in by_name["shot_result"] if e["result"] == "in" and e.get("pos")]
    if landings:
        half_w = C.TABLE_WIDTH / 2
        lat = Counter("left" if p[0] < -half_w / 3 else "right" if p[0] > half_w / 3 else "middle" for p in landings)
        half_l = C.TABLE_LENGTH / 2
        depth = Counter("short" if p[2] < half_l / 3 else "deep" if p[2] > 2 * half_l / 3 else "mid" for p in landings)
        n = len(landings)
        print("  returns land " + ", ".join(f"{k} {pct(lat[k], n).strip()}" for k in ("left", "middle", "right"))
              + " (across the table);  " + ", ".join(f"{k} {pct(depth[k], n).strip()}" for k in ("short", "mid", "deep")))

    # -- training data --------------------------------------------------------------
    section("Training data")
    labelled = Counter(confident_label(sw) for _, sw in swings)
    print(f"  {labelled['forehand']} forehands and {labelled['backhand']} backhands labelled reliably "
          "(hit, side rule and a sure camera all agree).")
    print("  Add them to the paddle's forehand/backhand training with:")
    print(f"    python tools/calibrate_swing.py --player {sessions[0].meta.get('player', '<player>')} "
          "--from-logs --with-games")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--player", default=C.DEFAULT_PLAYER)
    ap.add_argument("--last", type=int, metavar="N", help="only the last N sessions")
    ap.add_argument("--session", help="only this session (its folder name)")
    args = ap.parse_args()
    sessions = load_sessions(args.player, args.last, args.session)
    if not sessions:
        root = os.path.join(C.GAME_LOG_DIR, args.player)
        print(f"No recorded games in {root}/. Play with the real paddle: python main.py --player {args.player}")
        return 1
    print(f"Play profile: {args.player}, {len(sessions)} session(s)")
    report(sessions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
