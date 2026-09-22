"""Generate a printable AprilTag image.

Usage:
    python generate_apriltag.py [tag_id] [output_path]

Defaults to tag ID 0, family 36h11 (the most common family - 36 data bits,
minimum Hamming distance 11 between codes, so misreads are rare). Print the
saved PNG at a size where the black border is a few centimeters across, and
tape it flat - AprilTag detection is very sensitive to a tag that's curled
or wrinkled, since it relies on the border being a straight-edged quad.
"""

import sys
from pathlib import Path

import cv2

FAMILY = cv2.aruco.DICT_APRILTAG_36h11
TAG_PIXELS = 600  # output image size; print it, don't shrink it on screen


def main():
    tag_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(f"apriltag_{tag_id}.png")

    tag_dict = cv2.aruco.getPredefinedDictionary(FAMILY)
    tag_img = cv2.aruco.generateImageMarker(tag_dict, tag_id, TAG_PIXELS)

    cv2.imwrite(str(output_path), tag_img)
    print(f"Saved tag {tag_id} to {output_path}")


if __name__ == "__main__":
    main()
