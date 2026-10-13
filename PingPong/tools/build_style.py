#!/usr/bin/env python3
"""
tools/build_style.py -- build a player's style profile from their recorded
games (game_logs/<player>/) and save it to styles/<player>.json.

    python tools/build_style.py                 # DEFAULT_PLAYER
    python tools/build_style.py --player bri

The game rebuilds the profile itself whenever a style opponent is picked
(tag 3 / key 4 for luca), so this is for looking at it -- or for making the
styles/ file to play against on a machine without the recordings.
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import config as C  # noqa: E402
from game.style import build_profile, save_profile, summarize  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--player", default=C.DEFAULT_PLAYER)
    args = ap.parse_args()
    prof = build_profile(args.player)
    if prof is None:
        print(f"No recorded shots in {os.path.join(C.GAME_LOG_DIR, args.player)}/ -- play some games first "
              f"(python main.py --player {args.player}).")
        return 1
    path = save_profile(prof)
    print("\n".join(summarize(prof)))
    print(f"Saved {os.path.relpath(path, HERE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
