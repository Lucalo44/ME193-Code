#!/usr/bin/env python3
"""
tools/make_cutouts.py -- turn photos of a person into paper cut-outs for the crowd.

    python tools/make_cutouts.py photo1.jpg photo2.png ...

For each photo: MediaPipe's pose model (the one the game already uses) finds the
person and masks them from the background; only the main person is kept. The
result gets a white paper border, like a cut-out, and is saved as
web/crowd/cutout_<n>.png (transparent background). The stands in the 3D scene
(web/scene/crowd.js) are filled with whatever is in web/crowd/.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
import numpy as np  # noqa: E402
from mediapipe.tasks.python import BaseOptions, vision  # noqa: E402

import config as C  # noqa: E402
from vision.pose import ensure_model  # noqa: E402

OUT_DIR = os.path.join(HERE, "web", "crowd")
HEIGHT = 512                      # output height in pixels (plenty for a seat in the stands)


# Skeleton segments (MediaPipe pose landmark ids) and how far around each one the
# person can reach, in shoulder widths: generous for head and torso (hair, clothes),
# tight for arms and hands -- so someone standing next to an outstretched hand isn't kept.
HEAD = [(0, 7), (0, 8), (7, 8)]
LIMBS = [(11, 13, 0.24), (13, 15, 0.2), (12, 14, 0.24), (14, 16, 0.2),
         (15, 17, 0.16), (15, 19, 0.16), (15, 21, 0.16), (16, 18, 0.16), (16, 20, 0.16), (16, 22, 0.16),
         (23, 25, 0.3), (25, 27, 0.25), (24, 26, 0.3), (26, 28, 0.25)]


def skeleton_region(landmarks, shape) -> np.ndarray:
    """Where the detected person's body can be: around their head, torso and limbs."""
    h, w = shape
    pts = {i: (lm.x * w, lm.y * h) for i, lm in enumerate(landmarks)}
    sw = max(8.0, float(np.hypot(pts[11][0] - pts[12][0], pts[11][1] - pts[12][1])))
    region = np.zeros((h, w), np.uint8)
    p = lambda i: (int(round(pts[i][0])), int(round(pts[i][1])))  # noqa: E731
    torso = np.array([p(11), p(12), p(24), p(23)], np.int32)
    cv2.fillPoly(region, [torso], 1)
    cv2.polylines(region, [torso], True, 1, int(0.5 * sw))
    for a, b in HEAD:
        cv2.line(region, p(a), p(b), 1, int(1.3 * sw))
    cv2.line(region, p(0), (p(0)[0], int(p(0)[1] - 0.45 * sw)), 1, int(1.2 * sw))   # top of the head
    for a, b, r in LIMBS:
        cv2.line(region, p(a), p(b), 1, max(2, int(2 * r * sw)))
    return region


def person_mask(rgb: np.ndarray) -> np.ndarray:
    """0..1 mask of the main person in the photo (only them, not people behind)."""
    opts = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=C.POSE_MODEL_PATH),
                                        running_mode=vision.RunningMode.IMAGE, num_poses=1,
                                        output_segmentation_masks=True)
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        if not res.segmentation_masks:
            raise ValueError("no person found in the photo")
        # Copy out while the landmarker is open: the mask's memory goes with it.
        mask = np.array(res.segmentation_masks[0].numpy_view()[:, :, 0], dtype=np.float32)
        return mask * skeleton_region(res.pose_landmarks[0], mask.shape)


def cutout(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"can't read {path}")
    # Small photos: upscale so the mask is smooth. Sizes are kept to multiples of 4 --
    # MediaPipe aborts reading the mask back for some other widths.
    longest = max(bgr.shape[:2])
    scale = 700 / longest if longest < 700 else min(1.0, 1200 / longest)   # big phone photos: shrink
    w, h = (int(round(bgr.shape[1] * scale / 4)) * 4, int(round(bgr.shape[0] * scale / 4)) * 4)
    if (w, h) != (bgr.shape[1], bgr.shape[0]):
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_CUBIC)
    mask = person_mask(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    mask = (cv2.GaussianBlur(mask, (0, 0), 2) > 0.5).astype(np.uint8)
    # Keep only the biggest blob (the main person), fill small holes.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        mask = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    k = max(3, int(0.012 * max(mask.shape)) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    # Paper border: the silhouette grown by a few percent, filled white.
    pad = max(6, int(0.03 * max(mask.shape)))
    bgr = cv2.copyMakeBorder(bgr, pad, 0, pad, pad, cv2.BORDER_REPLICATE)
    mask = cv2.copyMakeBorder(mask, pad, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
    border = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1)))
    out = np.zeros((*mask.shape, 4), np.uint8)
    out[border > 0] = (250, 250, 247, 255)                 # off-white card
    out[mask > 0, :3] = bgr[mask > 0]
    out[mask > 0, 3] = 255
    # Crop to the cut-out and size it.
    ys, xs = np.nonzero(out[:, :, 3])
    out = out[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = out.shape[:2]
    return cv2.resize(out, (max(1, round(w * HEIGHT / h)), HEIGHT), interpolation=cv2.INTER_AREA)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", nargs="+")
    ap.add_argument("--keep", action="store_true", help="add to the existing cut-outs instead of replacing them")
    args = ap.parse_args()
    ensure_model()
    os.makedirs(OUT_DIR, exist_ok=True)
    if not args.keep:
        for old in glob.glob(os.path.join(OUT_DIR, "cutout_*.png")):
            os.remove(old)
    start = len(glob.glob(os.path.join(OUT_DIR, "cutout_*.png")))
    names = []
    for photo in args.photos:
        try:
            img = cutout(photo)
        except (ValueError, SystemExit) as exc:
            print(f"skipped {photo}: {exc}")
            continue
        name = f"cutout_{start + len(names)}.png"
        cv2.imwrite(os.path.join(OUT_DIR, name), img)
        names.append(name)
        print(f"wrote web/crowd/{name}  (from {photo})")
    all_names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(OUT_DIR, "cutout_*.png")))
    with open(os.path.join(OUT_DIR, "index.json"), "w") as f:
        f.write('{"cutouts": [' + ", ".join(f'"{n}"' for n in all_names) + "]}\n")
    print(f"web/crowd/index.json lists {len(all_names)} cut-out(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
