"""
vision/arm.py -- live arm tracking from the pose landmarks (wrist, elbow, shoulder).

arm_points() runs in the vision process on every frame. It returns both arms in
body coordinates: origin between the hips, measured in torso lengths (so it
doesn't matter how far the player stands from the camera), x to the right of
the SCREEN (the frame is mirrored, selfie-style), y up.

ArmTracker runs in the game. It decides which arm holds the paddle -- at first
the one on the player's handed side of the body, then confirmed by the real
paddle: at every swing the IMU reports, the paddle hand is the one moving
fastest. (MediaPipe's own left/right labels are swapped in a mirrored image, so
they're not relied on.) It also keeps the arm's neutral (ready) position,
captured with the paddle's zero (Z), and reports how far the hand has moved
from it, which the browser adds to the on-screen paddle's position.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, List, Optional, Sequence

LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER = 23, 24, 11, 12
ARMS = ((11, 13, 15), (12, 14, 16))       # (shoulder, elbow, wrist) for each arm
MIN_VISIBILITY = 0.5


def arm_points(landmarks) -> Optional[dict]:
    """Both arms in body coordinates, or None if there's no usable torso."""
    if not landmarks:
        return None
    hx = (landmarks[LEFT_HIP].x + landmarks[RIGHT_HIP].x) / 2
    hy = (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2
    sx = (landmarks[LEFT_SHOULDER].x + landmarks[RIGHT_SHOULDER].x) / 2
    sy = (landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) / 2
    torso = math.hypot(sx - hx, sy - hy)
    if torso < 1e-4:
        return None

    def body(i):
        p = landmarks[i]
        return [round((p.x - hx) / torso, 4), round(-(p.y - hy) / torso, 4)]

    arms = []
    for s, e, w in ARMS:
        vis = min(getattr(landmarks[k], "visibility", 1.0) or 0.0 for k in (s, e, w))
        arms.append({"shoulder": body(s), "elbow": body(e), "wrist": body(w), "vis": round(vis, 3)})
    return {"arms": arms}


def _sub(a, b):
    return [a[0] - b[0], a[1] - b[1]]


class ArmTracker:
    STALE_S = 0.35            # no fresh pose for this long -> not tracking
    VOTES_TO_DECIDE = 3       # swings before the paddle arm is chosen by motion

    def __init__(self, handedness: str = "right"):
        self.handedness = handedness
        self.history: Deque[tuple] = deque(maxlen=90)     # (t, arms) ~3 s at 30 fps
        self.votes = [0, 0]
        self.neutral: List[Optional[List[float]]] = [None, None]   # wrist - shoulder at ready, per arm
        self.chosen_by = "default"

    # --- input ---------------------------------------------------------
    def update(self, t: float, points: Optional[dict]) -> None:
        if points and points.get("arms"):
            self.history.append((t, points["arms"]))

    def on_swing(self, t_peak: float) -> None:
        """A real swing peaked at t_peak: the arm whose wrist moved fastest then
        is the paddle arm."""
        window = [(t, arms) for t, arms in self.history if abs(t - t_peak) <= 0.2]
        if len(window) < 3:
            return
        speeds = []
        for i in (0, 1):
            pts = [(t, arms[i]["wrist"]) for t, arms in window if arms[i]["vis"] >= MIN_VISIBILITY]
            if len(pts) < 2:
                speeds.append(0.0)
                continue
            dist = sum(math.dist(a[1], b[1]) for a, b in zip(pts, pts[1:]))
            speeds.append(dist / max(1e-3, pts[-1][0] - pts[0][0]))
        if max(speeds) > 0:
            self.votes[0 if speeds[0] >= speeds[1] else 1] += 1

    def zero(self) -> bool:
        """Capture the ready position of both arms (call with the paddle's zero)."""
        latest = self._latest()
        if latest is None:
            return False
        for i in (0, 1):
            a = latest[i]
            if a["vis"] >= MIN_VISIBILITY:
                self.neutral[i] = _sub(a["wrist"], a["shoulder"])
        return True

    # --- output --------------------------------------------------------
    def paddle_arm(self) -> Optional[int]:
        if sum(self.votes) >= self.VOTES_TO_DECIDE and self.votes[0] != self.votes[1]:
            self.chosen_by = "swings"
            return 0 if self.votes[0] > self.votes[1] else 1
        latest = self._latest()
        if latest is None:
            return None
        self.chosen_by = "default"
        # Default: the arm whose shoulder is on the player's handed side. The view is
        # mirrored, so a right-hander's right shoulder appears on the screen's right.
        right_side = 0 if latest[0]["shoulder"][0] > latest[1]["shoulder"][0] else 1
        return right_side if self.handedness == "right" else 1 - right_side

    def state(self, now: float) -> dict:
        """For the browser: the paddle arm's position, and the hand's offset from ready."""
        if not self.history or now - self.history[-1][0] > self.STALE_S:
            return {"ok": False}
        i = self.paddle_arm()
        if i is None:
            return {"ok": False}
        a = self.history[-1][1][i]
        if a["vis"] < MIN_VISIBILITY:
            return {"ok": False}
        rel_w, rel_e = _sub(a["wrist"], a["shoulder"]), _sub(a["elbow"], a["shoulder"])
        hand = _sub(rel_w, self.neutral[i]) if self.neutral[i] is not None else [0.0, 0.0]
        return {"ok": True, "hand": [round(v, 3) for v in hand],
                "elbow": [round(v, 3) for v in rel_e], "wrist": [round(v, 3) for v in rel_w],
                "zeroed": self.neutral[i] is not None, "chosen_by": self.chosen_by,
                "age_ms": round((now - self.history[-1][0]) * 1000)}

    def _latest(self) -> Optional[Sequence[dict]]:
        return self.history[-1][1] if self.history else None
