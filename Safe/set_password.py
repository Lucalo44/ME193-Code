"""Set the safe's 3-digit hand code (each digit 1-5).

    python3 set_password.py          # prompts for the code
    python3 set_password.py 4 2 5

Only a salted PBKDF2 hash goes into password.json (which is git-ignored). With 125 possible
codes that won't stop a determined person with the file, but it keeps the code out of plain
sight in the repo.
"""
import getpass
import hashlib
import hmac
import json
import os
import sys

import config as C


def _hash(digits, salt):
    return hashlib.pbkdf2_hmac("sha256", "".join(map(str, digits)).encode(), salt, 200_000).hex()


def save(digits):
    salt = os.urandom(16)
    C.PASSWORD_FILE.write_text(json.dumps({"salt": salt.hex(), "hash": _hash(digits, salt)}))


def check(digits):
    stored = json.loads(C.PASSWORD_FILE.read_text())
    return hmac.compare_digest(_hash(digits, bytes.fromhex(stored["salt"])), stored["hash"])


def parse(text):
    digits = [int(ch) for ch in text if ch.isdigit()]
    if len(digits) != C.CODE_LENGTH or not all(1 <= d <= 5 for d in digits):
        raise ValueError(f"need {C.CODE_LENGTH} digits, each 1-5")
    return digits


def main():
    text = " ".join(sys.argv[1:]) or getpass.getpass(f"{C.CODE_LENGTH}-digit code (1-5 each): ")
    try:
        digits = parse(text)
    except ValueError as e:
        raise SystemExit(e)
    save(digits)
    print(f"Saved to {C.PASSWORD_FILE.name}.")


if __name__ == "__main__":
    main()
