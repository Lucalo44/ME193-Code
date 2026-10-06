#!/usr/bin/env python3
"""
tools/make_tags.py -- render the three start/speed AprilTags (36h11) for printing.

Writes:
    tags/tag_<id>_<speed>.png   one printable page-sized image per tag, ID and speed label underneath
    tags/start_tags.pdf         all three on separate pages (via matplotlib)
    web/tags/tag_<id>.png       small copies shown on the lobby screen

Print them at least ~8 cm wide and keep the white border -- the detector
needs it.

    python tools/make_tags.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import config as C  # noqa: E402
from vision.apriltags import dictionary  # noqa: E402


def render(tag_id: int, label: str, size: int = 800) -> np.ndarray:
    marker = cv2.aruco.generateImageMarker(dictionary(), tag_id, size)
    border = size // 8
    page = cv2.copyMakeBorder(marker, border, border * 3, border, border, cv2.BORDER_CONSTANT, value=255)
    text = f"ID {tag_id}  -  {label.upper()}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = size / 400
    (tw, th), _ = cv2.getTextSize(text, font, scale, 3)
    cv2.putText(page, text, ((page.shape[1] - tw) // 2, size + border + border + th // 2),
                font, scale, 0, 3, cv2.LINE_AA)
    return page


def main() -> int:
    out_dir = os.path.join(HERE, "tags")
    web_dir = os.path.join(HERE, "web", "tags")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(web_dir, exist_ok=True)
    pages = []
    for tag_id, label in sorted(C.TAG_IDS.items()):
        page = render(tag_id, label)
        path = os.path.join(out_dir, f"tag_{tag_id}_{label}.png")
        cv2.imwrite(path, page)
        pages.append((page, tag_id, label))
        small = cv2.aruco.generateImageMarker(dictionary(), tag_id, 160)
        small = cv2.copyMakeBorder(small, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=255)
        cv2.imwrite(os.path.join(web_dir, f"tag_{tag_id}.png"), small)
        print(f"wrote {path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        pdf_path = os.path.join(out_dir, "start_tags.pdf")
        with PdfPages(pdf_path) as pdf:
            for page, tag_id, label in pages:
                fig = plt.figure(figsize=(8.5, 11))
                ax = fig.add_axes([0.1, 0.15, 0.8, 0.75])
                ax.imshow(page, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
                ax.axis("off")
                pdf.savefig(fig)
                plt.close(fig)
        print(f"wrote {pdf_path}")
    except ImportError:
        print("matplotlib not installed -- skipped the PDF (PNGs are fine to print).")

    # Self-check: every tag must decode from its own image.
    from vision.apriltags import TagDetector
    det = TagDetector()
    for page, tag_id, _ in pages:
        found = [i for i, _ in det.detect(cv2.cvtColor(page, cv2.COLOR_GRAY2BGR))]
        print(f"tag {tag_id}: {'OK' if found == [tag_id] else f'DETECTION FAILED ({found})'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
