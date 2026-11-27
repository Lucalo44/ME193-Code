"""Rhythms on a 16th-note grid, for the stick figures on the LED matrix.

A rhythm is a string with one character per 16th note: 'x' = a hit (the pose changes and
the figure flashes), '.' = hold. "x..." is quarter notes, "x.x." eighths, "xxxx" 16ths.

How a pattern lines up:
  * a pattern whose length is a whole number of bars (16, 32... in 4/4; 12, 24 in 3/4) is a
    phrase, counted from the start of its section, so a 2-bar tango phrase stays intact;
  * anything else repeats inside every bar ("x.x." in any meter; a 16-step 4/4 pattern in a
    3/4 bar is cut after 12 steps);
  * "" means no hits after the first one (a freeze or a bow holds still).
"""

# Each lane's default rhythm per move, used when nothing more specific is given.
CAR_RHYTHM = {"shimmy": "x.x.", "spin": "xxxx", "step": "x...", "wiggle": "x.x.x.x.x.x.xxxx",
              "freeze": "", "bow": ""}
LEGO_RHYTHM = {"wave": "x.x.", "clap": "x.x.", "pump": "x.x.", "point": "x...", "windmill": "xxxx",
               "freeze": "", "bow": ""}
DEFAULT = {"car": CAR_RHYTHM, "lego": LEGO_RHYTHM}

# The choreographer (an LLM, for styles outside the style library) names a feel; each feel
# is a (car, lego) rhythm pair, so legs and arms can run on different rhythms.
FEELS = {
    "smooth":     ("x.......", "x..."),              # half-note steps, quarter-note arms
    "bouncy":     ("x.x.", "x.x."),                  # eighths
    "sharp":      ("x...x...x.x.x...", "x..x..x."),  # hits and stops
    "syncopated": ("x..x..x.", "..x...x."),          # off the beat
    "driving":    ("x.x.", "xxxx"),                  # 16th-note arms over eighth-note feet
}


def valid(pattern):
    return isinstance(pattern, str) and len(pattern) <= 64 and set(pattern) <= {"x", "."}


def is_hit(pattern, step, meter=4):
    """Is 16th number `step` (counted from the section start) a hit?"""
    if not pattern:
        return False
    bar = meter * 4
    if len(pattern) % bar == 0:
        return pattern[step % len(pattern)] == "x"
    return pattern[(step % bar) % len(pattern)] == "x"


def for_meter(pattern, meter):
    """3/4 can't hold an 8-step cell evenly: stretch it to the bar."""
    if meter == 3 and len(pattern) == 8:
        return pattern + "...."
    return pattern
