"""LED matrix content: static text, scrolling text, icons, animations.

Python renders every frame (the sketch only paints what it's sent), so the
dashboard mirror shows exactly what the matrix shows. A single animator
thread owns the matrix; setting new content replaces the running animation.
"""
import threading
import time

from hal import COLS, ROWS
from skills.font import icon_frame, text_columns

SCROLL_S = 0.06          # seconds per column
TEXT_ROW = 0             # top row of the 7-high glyphs


def text_frame(cols, offset=0):
    """Frame showing `cols` starting at matrix column `offset` (may be negative)."""
    f = [0] * (ROWS * COLS)
    for c in range(COLS):
        src = c - offset
        if 0 <= src < len(cols):
            for r in range(7):
                if cols[src][r]:
                    f[(TEXT_ROW + r) * COLS + c] = 7
    return f


def digit_frame(ch):
    cols = text_columns(ch)
    return text_frame(cols, (COLS - len(cols)) // 2)


class Matrix:
    def __init__(self, hal):
        self.hal = hal
        self._gen = None
        self._cv = threading.Condition()
        self._token = 0
        self.content = {"kind": "clear"}
        threading.Thread(target=self._run, daemon=True, name="matrix").start()

    # ---- public API ----
    def show_frame(self, frame, content=None):
        self._set(iter([(frame, None)]), content or {"kind": "frame"})

    def clear(self):
        self.show_frame([0] * (ROWS * COLS), {"kind": "clear"})

    def icon(self, name):
        self.show_frame(icon_frame(name), {"kind": "icon", "name": name})

    def text(self, text):
        """Static if it fits, else scrolls forever. Returns seconds for one pass."""
        cols = text_columns(text)
        if len(cols) <= COLS:
            self.show_frame(text_frame(cols, (COLS - len(cols)) // 2), {"kind": "text", "text": text})
            return 0.0

        def scroll():
            while True:
                for off in range(COLS, -len(cols) - 1, -1):
                    yield text_frame(cols, off), SCROLL_S
        self._set(scroll(), {"kind": "text", "text": text})
        return (len(cols) + COLS) * SCROLL_S

    def thinking(self):
        """Three dots chasing along the bottom row, plus the think bubble."""
        base = icon_frame("think")

        def anim():
            i = 0
            while True:
                f = list(base)
                for k in range(3):
                    f[(ROWS - 1) * COLS + 8 + k] = 7 if k == i % 3 else 1
                yield f, 0.25
                i += 1
        self._set(anim(), {"kind": "thinking"})

    def warming(self):
        """An hourglass whose sand runs down while the model prefills its prompt
        (warm-up takes minutes; the smiley only appears once it is ready)."""
        outline = ["...#######...", "....#...#....", ".....#.#.....", "......#......",
                   "......#......", ".....#.#.....", "....#...#....", "...#######..."]
        # sand cells per stage: (row, col); top chamber empties into the bottom one
        stages = [[(1, 5), (1, 6), (1, 7), (2, 6)],
                  [(1, 5), (1, 6), (1, 7), (3, 6), (6, 6)],
                  [(1, 6), (4, 6), (6, 5), (6, 6), (6, 7)],
                  [(6, 5), (6, 6), (6, 7), (5, 6)]]
        frames = []
        for cells in stages:
            f = [7 if ch == "#" else 0 for row in outline for ch in row]
            for r, c in cells:
                f[r * COLS + c] = 4
            frames.append(f)

        def anim():
            while True:
                for f in frames:
                    yield f, 0.6
                yield frames[-1], 0.6
        self._set(anim(), {"kind": "warming"})

    # ---- internals ----
    def _set(self, gen, content):
        with self._cv:
            self._token += 1
            self._gen = gen
            self.content = content
            self._cv.notify()

    def _run(self):
        while True:
            with self._cv:
                while self._gen is None:
                    self._cv.wait()
                gen, token = self._gen, self._token
            try:
                frame, delay = next(gen)
            except StopIteration:
                with self._cv:
                    if self._token == token:
                        self._gen = None
                continue
            try:
                self.hal.draw(frame)
            except Exception as e:  # noqa: BLE001
                print(f"[matrix] draw failed: {e!r}", flush=True)
            if delay is None:
                with self._cv:
                    if self._token == token:
                        self._gen = None
                continue
            with self._cv:
                self._cv.wait_for(lambda: self._token != token, timeout=delay)
