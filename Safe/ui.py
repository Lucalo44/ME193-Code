"""Draws the virtual safe (right-hand panel of the run_safe.py window) with plain OpenCV, so
the same code runs on the laptop and on the UNO Q without a GUI toolkit."""
import time

import cv2
import numpy as np

import config as C

PANEL_W, PANEL_H = 380, C.FRAME_H
BUTTON = (90, 420, 290, 468)                 # x0, y0, x1, y1 inside the panel
N_LEDS = 9
FONT = cv2.FONT_HERSHEY_SIMPLEX

STEEL, DOOR, DARK = (70, 66, 62), (120, 116, 110), (28, 28, 28)
OFF, AMBER, BLUE = (55, 55, 55), (0, 170, 255), (255, 190, 80)
GREEN, RED, WHITE = (60, 210, 60), (40, 40, 230), (235, 235, 235)


def _led_colors(mode, digits, t):
    if mode == "armed" or mode == "progress":
        lit = digits * N_LEDS // C.CODE_LENGTH
        return [BLUE if i < lit else AMBER for i in range(N_LEDS)]
    if mode == "granted":
        return [GREEN] * N_LEDS
    if mode == "denied":
        return [RED if int(t * 6) % 2 else OFF] * N_LEDS
    if mode == "lockout":
        k = 0.35 + 0.35 * np.sin(t * 2)
        return [tuple(int(c * k) for c in RED)] * N_LEDS
    return [OFF] * N_LEDS


def _center_text(img, text, y, scale, color, thick=1):
    (w, _), _ = cv2.getTextSize(text, FONT, scale, thick)
    cv2.putText(img, text, ((img.shape[1] - w) // 2, y), FONT, scale, color, thick, cv2.LINE_AA)


DIAL, DIAL_R, RING_R = (190, 160), 48, 58


def _draw_hold_ring(p, hold):
    """Ring around the dial that fills over HOLD_S while a digit is held, with the digit and a
    countdown in the middle. Flashes green on confirm; asks for the hand to be removed after."""
    cv2.ellipse(p, DIAL, (RING_R, RING_R), 0, 0, 360, (55, 55, 55), 7, cv2.LINE_AA)
    if hold["flash"] is not None:
        cv2.ellipse(p, DIAL, (RING_R, RING_R), 0, 0, 360, GREEN, 7, cv2.LINE_AA)
        _text_at(p, str(hold["flash"]), DIAL[1] + 14, 1.6, GREEN, 3)
        _text_at(p, hold.get("flash_text", "OK"), DIAL[1] + 36, 0.5, GREEN, 1)
    elif hold["release"]:
        _text_at(p, "remove", DIAL[1] - 2, 0.5, AMBER, 1)
        _text_at(p, "hand", DIAL[1] + 16, 0.5, AMBER, 1)
    elif hold["label"]:
        prog = hold["progress"]
        cv2.ellipse(p, DIAL, (RING_R, RING_R), -90, 0, 360 * prog, BLUE, 7, cv2.LINE_AA)
        _text_at(p, hold["label"], DIAL[1] + 14, 1.6, WHITE, 3)
        _text_at(p, f"{hold['remaining']:.1f}s", DIAL[1] + 36, 0.5, (200, 200, 200), 1)
    else:
        _text_at(p, "hold", DIAL[1] - 2, 0.5, (200, 200, 200), 1)
        _text_at(p, f"{C.HOLD_S:.0f}s", DIAL[1] + 16, 0.5, (200, 200, 200), 1)


def _text_at(img, text, y, scale, color, thick=1, cx=DIAL[0]):
    (w, _), _ = cv2.getTextSize(text, FONT, scale, thick)
    cv2.putText(img, text, (cx - w // 2, y), FONT, scale, color, thick, cv2.LINE_AA)


def draw_panel(safe, status, owner_mode, owner_ready, hold=None):
    t = time.time()
    p = np.full((PANEL_H, PANEL_W, 3), 18, np.uint8)
    _center_text(p, "HAND SAFE", 30, 0.8, WHITE, 2)

    # body + door
    cv2.rectangle(p, (30, 45), (350, 275), STEEL, -1)
    if safe.locked:
        cv2.rectangle(p, (45, 58), (335, 262), DOOR, -1)
        cv2.rectangle(p, (45, 58), (335, 262), (90, 86, 80), 2)
        cx, cy = DIAL
        cv2.circle(p, DIAL, DIAL_R, (80, 78, 74), -1)
        cv2.circle(p, DIAL, DIAL_R, (150, 146, 140), 2)
        for k in range(12):                              # dial ticks
            a = k * np.pi / 6
            cv2.line(p, (int(cx + 40 * np.cos(a)), int(cy + 40 * np.sin(a))),
                     (int(cx + 47 * np.cos(a)), int(cy + 47 * np.sin(a))), WHITE, 1)
        if hold is not None:
            _draw_hold_ring(p, hold)
        cv2.rectangle(p, (295, 130), (308, 190), (60, 58, 54), -1)    # handle
    else:
        cv2.rectangle(p, (45, 58), (335, 262), DARK, -1)
        for i in range(3):                               # a modest stack of gold
            cv2.rectangle(p, (120 + i * 45, 225), (160 + i * 45, 255), (40, 190, 230), -1)
        cv2.rectangle(p, (142, 195), (182, 225), (40, 190, 230), -1)
        cv2.rectangle(p, (187, 195), (227, 225), (40, 190, 230), -1)
        cv2.fillPoly(p, [np.array([(45, 58), (8, 38), (8, 282), (45, 262)])], DOOR)   # swung open

    # LED strip
    for i, col in enumerate(_led_colors(safe.leds, safe.digits, t)):
        x = 190 + (i - (N_LEDS - 1) / 2) * 34
        cv2.circle(p, (int(x), 297), 10, col, -1, cv2.LINE_AA)
        cv2.circle(p, (int(x), 297), 10, (20, 20, 20), 1, cv2.LINE_AA)

    # code slots
    for i in range(C.CODE_LENGTH):
        x0 = 190 - (C.CODE_LENGTH * 60) // 2 + i * 60 + 5
        cv2.rectangle(p, (x0, 318), (x0 + 50, 358), (90, 90, 90), 2)
        if i < safe.digits:
            cv2.putText(p, "*", (x0 + 14, 352), FONT, 1.3, WHITE, 2, cv2.LINE_AA)

    _center_text(p, status, 383, 0.6, WHITE, 1)
    if owner_mode:
        owner_txt, owner_col = "OWNER CHECK: ON  (o)", GREEN
    elif owner_ready:
        owner_txt, owner_col = "OWNER CHECK: off  (o)", (150, 150, 150)
    else:
        owner_txt, owner_col = "OWNER CHECK: no model", (110, 110, 110)
    _center_text(p, owner_txt, 406, 0.45, owner_col, 1)

    x0, y0, x1, y1 = BUTTON
    lit = safe.leds in ("armed", "progress")
    cv2.rectangle(p, (x0, y0), (x1, y1), (30, 30, 160) if lit else (40, 40, 200), -1)
    cv2.rectangle(p, (x0, y0), (x1, y1), (200, 200, 255), 2)
    _center_text(p, "PRESS  (space)", y0 + 31, 0.6, WHITE, 2)
    return p


def in_button(x, y, panel_x):
    x0, y0, x1, y1 = BUTTON
    return panel_x + x0 <= x <= panel_x + x1 and y0 <= y <= y1
