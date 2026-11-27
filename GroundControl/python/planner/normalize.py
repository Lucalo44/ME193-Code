"""Input normalizer (Part 2, A2): fixes what speech-to-text gets wrong, before the reflexes,
the fast path and the planner ever see the command.

Conservative by design. Every rule needs a cue right after the word it changes, so it only
fires where the reading is unambiguous:

  * "spin to times", "for bars", "won time"  -> two / four / one, but only right before a
    unit (times, bars, beats, degrees, seconds...). "go to the left" and "dance for 4 bars"
    are untouched.
  * "right hi on the matrix" -> "write hi on the matrix": "right" opening a command that
    puts something on the display, and not followed by a turn amount.
  * "Twice"/"thrice" and numbers as words are left alone: the planner was trained on them.

The changes are returned too, so the dashboard and the command log can show them.

Streaming partials (A6): a rule looks one word ahead, so a word can change when the next
one arrives ("spin to" -> "spin two times"). The prefill just re-reads from that word,
which costs a few tokens; earlier words never change.
"""
import re

UNITS = r"(times?|x\b|bars?|beats?|degrees?|seconds?|secs?|steps?|loops?|rounds?|spins?|turns?)"
_HOMOPHONE = [
    (re.compile(rf"\b(to|too)(?=\s+{UNITS})", re.I), "two"),
    (re.compile(rf"\b(for|fore)(?=\s+{UNITS})", re.I), "four"),
    (re.compile(rf"\b(won)(?=\s+{UNITS})", re.I), "one"),
    (re.compile(rf"\b(ate)(?=\s+{UNITS})", re.I), "eight"),
]
# "right hi on the matrix": STT writes "write" as "right". Only at the start of a command that
# targets the display, and never when a turn amount follows ("right 90 degrees").
_RIGHT_WRITE = re.compile(r"^(\s*(please\s+)?)right\b(?!\s+(\d+|ninety|forty|a quarter|turn|now))"
                          r"(?=.*\bon\s+the\s+(led\s+)?(matrix|screen|display|leds?)\b)", re.I)


def normalize(text):
    """-> (normalized text, [(before, after), ...])."""
    out, changes = text or "", []
    for rx, word in _HOMOPHONE:
        def sub(m, word=word):
            changes.append((m.group(1), word))
            return word
        out = rx.sub(sub, out)
    m = _RIGHT_WRITE.match(out)
    if m:
        changes.append(("right", "write"))
        out = out[:m.end(1)] + "write" + out[m.end(1) + len("right"):]
    return out, changes


# ---- turn convention: positive = left (counter-clockwise); no direction given = left.
# "turn 180 degrees" has no direction, and the model sometimes wrote -180. The heading ends
# up the same, but the rule makes plans predictable (docs/MODEL.md).
_DIRECTION = re.compile(r"\b(left|right|clockwise|counter-?clockwise|anti-?clockwise|ccw|cw)\b", re.I)


def turn_default_left(plan, text):
    """Make every turn positive when the command names no direction at all."""
    if not plan or _DIRECTION.search(text or ""):
        return plan
    for st in plan.get("steps") or []:
        if st.get("skill") == "turn_by" and isinstance((st.get("args") or {}).get("degrees"), (int, float)):
            st["args"]["degrees"] = abs(st["args"]["degrees"])
    return plan
