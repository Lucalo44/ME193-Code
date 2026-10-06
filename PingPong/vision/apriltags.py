"""
vision/apriltags.py -- AprilTag 36h11 detection (OpenCV ArUco module) and a
debounce that confirms a tag once it's been seen in TAG_CONFIRM_FRAMES of the
last TAG_HISTORY_FRAMES frames.

Detection MUST run on the unflipped camera frame: mirrored AprilTags do not
decode.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config as C

FAMILIES = {
    "36h11": cv2.aruco.DICT_APRILTAG_36h11,
    "25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "16h5": cv2.aruco.DICT_APRILTAG_16h5,
}


def dictionary():
    return cv2.aruco.getPredefinedDictionary(FAMILIES[C.TAG_FAMILY])


class TagDetector:
    def __init__(self):
        self.detector = cv2.aruco.ArucoDetector(dictionary(), cv2.aruco.DetectorParameters())

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        """Returns [(tag_id, corners 4x2 float array)] for tags in TAG_IDS."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return []
        return [(int(i), c.reshape(4, 2)) for c, i in zip(corners, ids.flatten()) if int(i) in C.TAG_IDS]


class TagDebouncer:
    def __init__(self, confirm: int = C.TAG_CONFIRM_FRAMES, history: int = C.TAG_HISTORY_FRAMES):
        self.confirm = confirm
        self.history: deque = deque(maxlen=history)

    def reset(self) -> None:
        self.history.clear()

    def update(self, ids_seen) -> Tuple[Optional[int], float, Optional[int]]:
        """Feed one frame's detected ids. Returns (leading_id, progress 0..1,
        confirmed_id). After a confirmation the history is cleared so the
        same tag doesn't fire again on the next frame."""
        self.history.append(set(ids_seen))
        counts: Dict[int, int] = {}
        for frame_ids in self.history:
            for i in frame_ids:
                counts[i] = counts.get(i, 0) + 1
        if not counts:
            return None, 0.0, None
        lead = max(counts, key=lambda i: counts[i])
        progress = min(1.0, counts[lead] / self.confirm)
        if counts[lead] >= self.confirm:
            self.history.clear()
            return lead, 1.0, lead
        return lead, progress, None
