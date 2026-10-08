"""
server/protocol.py -- every message exchanged between Python and the browser.
Mirrored as a comment block at the top of web/main.js; keep them in sync.

Python -> browser
-----------------
state (~60 Hz):
    {"type": "state", "t": sim_time, "phase": "LOBBY|SERVE|RALLY|POINT_OVER|GAME_OVER|LATENCY_CAL",
     "paused": bool,
     "ball": {"pos": [x,y,z], "vel": [vx,vy,vz], "spin": [sx,sy,sz], "visible": bool},
     "opponent": {"x": float, "swing": "forehand"|"backhand"|null, "swing_t": seconds since swing start},
     "paddle": {"q": [x, y, z, w], "pitch": deg, "roll": deg, "yaw": deg},   # scene frame, vs. ready pose
     "stroke": {"phase": -1.3..1.3, "side": "forehand"|"backhand", "mode": "ready"|"forward"|"settle"} | null,
                                               # live stroke (real paddle): +1 drawn back, 0 contact, -1 follow-through
     "hold": bool,                             # hit-stop: the ball is waiting at the paddle for the swing to register
     "arm": {"ok": bool, "hand": [dx, dy], "elbow": [x, y], "wrist": [x, y], "zeroed": bool, "chosen_by": str} | null,
                                               # pose arm tracking, in torso lengths (hand = offset from ready)
     "required_stroke": "forehand"|"backhand"|"either"|null,
     "incoming": {"x": x_arrival, "t_to_arrival": s, "required": str, "window": [early_s, late_s]} | null,
     "contact": {"pos": [x,y,z], "t_to_contact": s, "serve": bool, "swung": bool} | null,
                                               # where the paddle meets the ball; drives auto-positioning
     "score": {"player", "cpu", "games_player", "games_cpu", "server", "games_needed"},
     "streak": {"current": n, "record": n},     # continuous hits; record also goes to MQTT
     "serve": {"server": "player"|"cpu", "state": null|"cpu"|"await_toss"|"tossed", "paddle_kind": str},
     "speed_setting": "slow"|"medium"|"fast",
     "status": {"paddle": str, "paddle_kind": "real"|"sim", "camera": str, "pose": bool,
                "stroke_check": bool, "calibration": "file"|"default", "pose_label": str|null,
                "mqtt": "connected"|"connecting"|"reconnecting"|"disabled"|...},
     "tag": {"id": int, "label": str, "progress": 0..1} | null,
     "latency": {"beats": [t...], "count": n, "offset_s": float} | null,
     "debug": {...} | null, "settings": {"hint": bool, "handedness": str, "debug": bool}}

event (discrete, triggers sounds/effects):
    {"type": "event", "name": "hit"|"bounce"|"net"|"miss"|"point"|"game_over"|"tag_progress"
                              |"tag_confirmed"|"swing"|"serve"|"message"|"record"
                              |"serve_prompt"|"toss"|"let"|"early_swing", ...fields}
    hit:        who, pos, speed, topspin, sidespin, stroke, confidence, required
    bounce:     pos, side
    net:        pos
    miss:       reason ("EARLY"|"LATE"|"WRONG STROKE"|"NO SWING"|"OUT"|"NET"), needed, judged
    point:      winner, reason, score
    game_over:  match_over, winner, score
    swing:      summary of the SwingEvent + trace [[t_rel, lin_g, gyro_dps], ...]
    record:     record (new best number of continuous hits)
    serve:      pos, by ("cpu")         serve_prompt: by ("player")
    toss:       height, pos             let: (a serve clipped the net -- replay it)
    miss reasons also include "SERVE FAULT" and "MISSED SERVE" (never "EARLY": a too-early
    swing is just a whiff -- event early_swing, by_s -- and the player can swing again)

camera_frame (~12 fps):
    {"type": "camera_frame", "jpeg_b64": "..."}

Browser -> Python
-----------------
    {"type": "key", "key": "p", "down": true, "shift": false}
"""

from __future__ import annotations

import json
from typing import Any


def _round(v: Any, nd: int = 4) -> Any:
    if isinstance(v, float):
        return round(v, nd)
    if isinstance(v, (list, tuple)):
        return [_round(x, nd) for x in v]
    if isinstance(v, dict):
        return {k: _round(x, nd) for k, x in v.items()}
    return v


def state_message(**fields) -> str:
    return json.dumps(_round({"type": "state", **fields}))


def event_message(name: str, **fields) -> str:
    return json.dumps(_round({"type": "event", "name": name, **fields}))


def camera_frame_message(jpeg_b64: str) -> str:
    return json.dumps({"type": "camera_frame", "jpeg_b64": jpeg_b64})


def parse_client_message(raw: str) -> dict:
    try:
        msg = json.loads(raw)
        return msg if isinstance(msg, dict) else {}
    except ValueError:
        return {}
