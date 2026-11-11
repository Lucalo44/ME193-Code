"""Set the safe's 3-step code. Each step is a number 1-5, shown by hand (h) or said (v).

    python3 set_password.py 1h 3v 5h     # hand 1, say "three", hand 5 (mode before or after)
    python3 set_password.py 4 2 5        # bare digits = all hand
    python3 set_password.py              # prompts (hidden input)

Which steps are spoken is part of the secret: the safe takes either at every step and
only checks the whole sequence at the end, so the screen never hints what's expected.
Only a salted PBKDF2 hash goes into password.json (which is git-ignored).
"""
import getpass
import hashlib
import hmac
import json
import os
import re
import sys

import config as C

MODES = {"h": "hand", "v": "voice"}


def _key(steps):
    return "".join(f"{m}{d}" for m, d in steps)          # e.g. "h1v2h5"


def _hash(text, salt):
    return hashlib.pbkdf2_hmac("sha256", text.encode(), salt, 200_000).hex()


def save(steps):
    salt = os.urandom(16)
    C.PASSWORD_FILE.write_text(json.dumps({"format": 2, "salt": salt.hex(), "hash": _hash(_key(steps), salt)}))


def check(steps):
    """steps: [("h" | "v", digit), ...] in the order they were entered."""
    stored = json.loads(C.PASSWORD_FILE.read_text())
    salt = bytes.fromhex(stored["salt"])
    if stored.get("format", 1) == 1:                       # old hand-only file: digits only
        if any(m != "h" for m, _ in steps):
            return False
        text = "".join(str(d) for _, d in steps)
    else:
        text = _key(steps)
    return hmac.compare_digest(_hash(text, salt), stored["hash"])


def parse(text):
    """Accepts the mode before or after each digit: "h1 v3 h5", "1h 3v 5h", "hand 1, voice 3".
    A bare digit is a hand step."""
    toks = re.findall(r"[a-z]+|[0-9]", text.lower())
    suffix = bool(toks) and toks[0].isdigit()           # "1h 3v 5h" style
    steps = []
    for i, t in enumerate(toks):
        if not t.isdigit():
            continue
        word = None
        if suffix and i + 1 < len(toks) and not toks[i + 1].isdigit():
            word = toks[i + 1]
        elif not suffix and i > 0 and not toks[i - 1].isdigit():
            word = toks[i - 1]
        mode = (word or "h")[0]
        if mode not in MODES:
            raise ValueError(f"'{word}' isn't h/hand or v/voice")
        steps.append((mode, int(t)))
    if len(steps) != C.CODE_LENGTH or not all(1 <= d <= 5 for _, d in steps):
        raise ValueError(f"need {C.CODE_LENGTH} steps, each 1-5, like: h1 v2 h5")
    return steps


def describe(steps):
    return ", ".join(f"{MODES[m]} {d}" for m, d in steps)


def main():
    text = " ".join(sys.argv[1:]) or getpass.getpass(f"{C.CODE_LENGTH}-step code (e.g. h1 v2 h5): ")
    try:
        steps = parse(text)
    except ValueError as e:
        raise SystemExit(e)
    save(steps)
    print(f"Saved to {C.PASSWORD_FILE.name}: {len(steps)} steps "
          f"({sum(m == 'v' for m, _ in steps)} spoken).")


if __name__ == "__main__":
    main()
