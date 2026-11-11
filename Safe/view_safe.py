"""Laptop companion: the safe's UI on the laptop.

    python3 view_safe.py                 # waits; pops up when the safe is armed for a code attempt
    python3 view_safe.py --always        # keep the window open the whole time
    python3 view_safe.py --host 10.0.0.5 # board address (default: ArduinoUNOq/tools/board_secrets.py)

Shows the same view as run_safe.py's window on the laptop: the camera with the box, the
live prediction and hold bar, the mic meter, and the virtual safe panel. SPACE (or clicking
PRESS) presses the safe's button remotely, q quits. A browser at http://<board>:8765 works too.
If the board is only connected over USB, run `adb forward tcp:8765 tcp:8765` and use
--host localhost.
"""
import argparse
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

import config as C
import ui

PORT = 8765
CLOSE_AFTER_S = 4            # keep the window up this long after an attempt ends (shows the result)


def board_hosts():
    """HOSTS from ArduinoUNOq/tools/board_secrets.py (the same list deploy.py tries)."""
    tools = C.HERE.parent / "ArduinoUNOq" / "tools"
    try:
        sys.path.insert(0, str(tools))
        from board_secrets import HOSTS
        return list(HOSTS)
    except Exception:
        return []
    finally:
        sys.path.remove(str(tools))


class Reader(threading.Thread):
    """Pulls JPEGs out of the multipart stream in the background; keeps only the newest."""

    def __init__(self, resp):
        super().__init__(daemon=True)
        self.resp, self.frame, self.t_frame, self.alive = resp, None, 0.0, True

    def run(self):
        buf = b""
        try:
            while True:
                chunk = self.resp.read1(65536) if hasattr(self.resp, "read1") else self.resp.read(4096)
                if not chunk:
                    break
                buf += chunk
                start = buf.find(b"\xff\xd8")                 # JPEG start / end markers
                end = buf.find(b"\xff\xd9", start + 2)
                while start != -1 and end != -1:
                    img = cv2.imdecode(np.frombuffer(buf[start:end + 2], np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        self.frame, self.t_frame = img, time.time()
                    buf = buf[end + 2:]
                    start = buf.find(b"\xff\xd8")
                    end = buf.find(b"\xff\xd9", start + 2)
                if len(buf) > 4_000_000:
                    buf = b""
        except OSError:
            pass
        self.alive = False


def get_state(url):
    try:
        return json.loads(urllib.request.urlopen(url + "/state", timeout=1.5).read())
    except (OSError, ValueError):
        return None


def find_safe(hosts, port):
    for host in hosts:
        url = f"http://{host}:{port}"
        if get_state(url) is not None:
            return url
        print(f"  {url}: no answer")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", help="board address (default: HOSTS from board_secrets.py)")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--always", action="store_true",
                    help="keep the window open the whole time (default: pop up only while the safe is armed)")
    a = ap.parse_args()

    hosts = [a.host] if a.host else board_hosts() + ["localhost"]
    url = find_safe(hosts, a.port)
    if url is None:
        raise SystemExit("Couldn't reach the safe. Is the safe app running on the board? "
                         "(python3 tools/deploy.py safe)")
    print(f"Connected to {url}")

    def press():
        try:
            urllib.request.urlopen(url + "/press", timeout=3).read()
            print("button pressed")
        except OSError as e:
            print("press failed:", e)

    win = "Hand Safe (UNO Q)"
    reader, shown, t_left, last_state = None, False, None, None
    if not a.always:
        print("Waiting for the safe to arm (press its button twice: lock, then enter code). Ctrl+C quits.")
    try:
        while True:
            st = get_state(url) or {}
            state = st.get("state")
            if state != last_state and state:
                print(f"safe: {state} - {st.get('status', '')}")
                last_state = state
            want = a.always or state == "ARMED"
            if want:
                t_left = None
            elif shown and t_left is None:
                t_left = time.time()             # attempt over: keep the result up for a moment

            if want and not shown:               # open the stream + window
                try:
                    reader = Reader(urllib.request.urlopen(url + "/stream", timeout=3))
                    reader.start()
                except OSError as e:
                    print("stream failed:", e)
                    time.sleep(1)
                    continue
                cv2.namedWindow(win)
                cv2.setMouseCallback(win, lambda ev, x, y, *_: threading.Thread(target=press, daemon=True).start()
                                     if ev == cv2.EVENT_LBUTTONDOWN and ui.in_button(x, y, C.FRAME_W) else None)
                shown = True
            if shown and t_left is not None and time.time() - t_left > CLOSE_AFTER_S:
                reader.resp.close()
                cv2.destroyWindow(win)
                cv2.waitKey(1)
                shown, reader, t_left = False, None, None

            if not shown:
                time.sleep(0.3)
                continue
            # draw for ~0.3 s between state polls
            t_end = time.time() + 0.3
            while time.time() < t_end:
                if reader.frame is not None:
                    img = reader.frame.copy()
                    if time.time() - reader.t_frame > 2:
                        cv2.putText(img, "no new frames...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    img = np.zeros((C.FRAME_H, C.FRAME_W + ui.PANEL_W, 3), np.uint8)
                    cv2.putText(img, "connecting...", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow(win, img)
                k = cv2.waitKey(30) & 0xFF
                if k == ord("q"):
                    raise KeyboardInterrupt
                if k == ord(" "):
                    threading.Thread(target=press, daemon=True).start()
            if not reader.alive:
                print("Stream ended (safe app restarted?). Reconnecting...")
                cv2.destroyWindow(win)
                shown, reader = False, None
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
