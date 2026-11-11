"""Regenerates the write-up figures in notion/media/ from the real data and models.

    cd Safe && python3 notion/make_figures.py      (needs matplotlib: pip install matplotlib)
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import cv2                                     # noqa: E402
import matplotlib                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                # noqa: E402
import numpy as np                             # noqa: E402
import torch                                   # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from torch.utils.data import DataLoader        # noqa: E402

import audio                                   # noqa: E402
import config as C                             # noqa: E402
import models                                  # noqa: E402
import run_safe                                # noqa: E402
from dataset import FolderDataset, load_split, sessions_of  # noqa: E402
from train import evaluate                     # noqa: E402

OUT = HERE / "media"
OUT.mkdir(exist_ok=True)

# reference palette (light mode)
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BLUES = LinearSegmentedColormap.from_list("blues", ["#f2f6fc", "#9fc1ec", BLUE, "#0f3d75"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "font.size": 11, "axes.titleweight": "bold", "axes.titlesize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def box(ax, x, y, w, h, text, fill="#ffffff", edge=GRID, color=INK, weight="normal", size=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fill, ec=edge, lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=color, fontsize=size, weight=weight)


def arrow(ax, a, b, color=MUTED):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, lw=1.5, color=color))


# ---------------------------------------------------------------- 1. pipeline
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.set_title("From button press to bolt", loc="left")
    hand = [("USB webcam\n30 fps", 0.1), ("crop the box\n96 px RGB", 1.75), ("GestureNet\nnone / 1-5", 3.4),
            ("hold logic\n3 s of votes", 5.05)]
    voice = [("webcam mic\n16 kHz", 0.1), ("trigger cuts\na 1 s clip", 1.75), ("log-mel\n40 x 98", 3.4),
             ("VoiceNet\nnone / one-five", 5.05)]
    for (t, x) in hand:
        box(ax, x, 2.55, 1.45, 0.9, t, edge=BLUE)
    for (t, x) in voice:
        box(ax, x, 0.65, 1.45, 0.9, t, edge=ORANGE)
    for i in range(3):
        arrow(ax, (hand[i][1] + 1.45, 3.0), (hand[i + 1][1], 3.0))
        arrow(ax, (voice[i][1] + 1.45, 1.1), (voice[i + 1][1], 1.1))
    box(ax, 7.0, 1.6, 1.6, 1.0, "code check\n3 steps vs. hash", fill="#eef4fc", edge=BLUE, weight="bold")
    arrow(ax, (6.5, 3.0), (7.4, 2.6))
    arrow(ax, (6.5, 1.1), (7.4, 1.6))
    box(ax, 9.2, 2.55, 1.7, 0.9, "servo bolt\nD9", edge=INK2)
    box(ax, 9.2, 0.65, 1.7, 0.9, "LED matrix\n8 x 13", edge=INK2)
    arrow(ax, (8.6, 2.3), (9.2, 3.0))
    arrow(ax, (8.6, 1.9), (9.2, 1.1))
    ax.text(10.05, 3.6, "via Bridge -> STM32", ha="center", color=INK2, fontsize=9)
    box(ax, 7.0, 3.35, 1.6, 0.6, "button D2", edge=INK2, size=9)
    arrow(ax, (7.8, 3.35), (7.8, 2.6))
    ax.text(0.1, 3.75, "hand", color=BLUE, weight="bold")
    ax.text(0.1, 0.2, "voice", color=ORANGE, weight="bold")
    save(fig, "fig-pipeline.png")


# ---------------------------------------------------------------- 2. training data
def fig_data():
    labels = C.CLASSES["gesture"]
    fig, axes = plt.subplots(3, 6, figsize=(11, 5.6))
    for j, lab in enumerate(labels):
        sess = list(sessions_of("gesture", lab).values())
        picks = [sess[-1][len(sess[-1]) // 3], sess[-2][len(sess[-2]) // 2], sess[0][len(sess[0]) // 2]]
        for i, f in enumerate(picks):
            ax = axes[i, j]
            ax.imshow(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(lab, fontsize=12)
    fig.subplots_adjust(left=0.01, right=0.99, wspace=0.06, hspace=0.08)
    fig.suptitle("Training crops: ~900 per label, 4 sessions each, every label in every setup",
                 x=0.01, ha="left", fontweight="bold")
    save(fig, "fig-data.png")


# ---------------------------------------------------------------- 3. the CNN
def fig_cnn():
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    ax.set_title("GestureNet: four conv blocks, then average and classify (294,774 weights)", loc="left")
    stages = [("input", 3, 96), ("block 1", 16, 48), ("block 2", 32, 24), ("block 3", 64, 12),
              ("block 4", 128, 6), ("avg pool", 128, 1), ("6 scores", 6, 1)]
    slot = 10.6 / len(stages)
    for i, (name, ch, hw) in enumerate(stages):
        h = 0.35 + 2.2 * hw / 96
        w = 0.2 + 0.75 * ch / 128
        x = 0.2 + i * slot + (slot - w) / 2
        y = 1.55 - h / 2 + 0.3
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                    fc=BLUE if 0 < i < 5 else "#ffffff", ec=BLUE, lw=1.5,
                                    alpha=0.25 + 0.6 * i / 6 if 0 < i < 5 else 1))
        ax.text(x + w / 2, 0.35, name, ha="center", fontsize=10, weight="bold")
        ax.text(x + w / 2, 0.05, f"{ch} x {hw} x {hw}" if hw > 1 else f"{ch}", ha="center", fontsize=9, color=INK2)
        if i < len(stages) - 1:
            nxt = 0.2 + (i + 1) * slot + (slot - (0.2 + 0.75 * stages[i + 1][1] / 128)) / 2
            arrow(ax, (x + w + 0.05, 1.85), (nxt - 0.05, 1.85))
    ax.text(0.2, 3.3, "each block = 3x3 conv -> batch norm -> ReLU, twice, then 2x2 max pool "
            "(halves the size, doubles the channels)", fontsize=9.5, color=INK2)
    save(fig, "fig-cnn.png")


# ---------------------------------------------------------------- 4. confusion
def fig_confusion():
    labels = C.CLASSES["gesture"]
    _, va, _ = load_split("gesture")
    m = models.load("gesture")
    acc, conf, _, _ = evaluate(m, DataLoader(FolderDataset(va, "gesture", False), 64), "cpu", len(labels))
    conf = conf.numpy()
    frac = conf / conf.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.imshow(frac, cmap=BLUES, vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if conf[i, j]:
                ax.text(j, i, conf[i, j], ha="center", va="center", fontsize=10,
                        color="#ffffff" if frac[i, j] > 0.5 else INK)
    ax.set_xticks(range(6), labels)
    ax.set_yticks(range(6), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actually shown")
    ax.set_title(f"Held-out frames: {acc:.1%} correct", loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    save(fig, "fig-confusion.png")


# ---------------------------------------------------------------- 5. hold logic
def fig_hold():
    rng = np.random.default_rng(3)
    r = run_safe.CodeReader()
    t, ts, credit, events = 0.0, [], {d: [] for d in "12345"}, []
    for lab, dur in [("none", 0.5), ("3", 5.0), ("none", 1.0)]:
        for _ in range(int(dur * 10)):
            t += 0.1
            l, c = lab, 0.9
            u = rng.random()
            if lab != "none" and u < 0.25:
                l = str(rng.integers(1, 6))
            elif lab != "none" and u < 0.4:
                c = 0.35
            before = dict(r.credit)
            d = r.update(l, c, None, t)
            ts.append(t)
            snap = before if d else r.credit
            if d:
                snap = dict(before)
                snap[str(d)] = C.HOLD_S
                events.append(t)
            for k in credit:
                credit[k].append(snap.get(k, 0.0))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.axhline(C.HOLD_S, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(6.5, C.HOLD_S + 0.08, "confirm at 3 s of credit", color=INK2, fontsize=9, ha="right")
    ax.plot(ts, credit["3"], color=BLUE, lw=2, label="digit actually shown (3)")
    others = np.max([credit[k] for k in "1245"], axis=0)
    ax.plot(ts, others, color=ORANGE, lw=2, label="best misread digit")
    for e in events:
        ax.plot([e], [C.HOLD_S], "o", ms=9, color=BLUE, mec=SURFACE, mew=2)
        ax.annotate(f"3 confirmed at {e - 0.5:.1f} s into the hold", (e, C.HOLD_S), xytext=(e + 0.1, 2.2),
                    color=INK, fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.set_xlabel("seconds")
    ax.set_ylabel("credit (s)")
    ax.set_ylim(0, 3.6)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 0.78))
    ax.text(5.15, 0.25, "credit cleared:\nremove hand", color=INK2, fontsize=9)
    ax.set_title("Hold logic: a flickery hand (25% misreads, 15% unsure) still gets through", loc="left")
    save(fig, "fig-hold.png")


# ---------------------------------------------------------------- 6. voice spectrograms
def fig_voice():
    words = ["1", "2", "3", "4", "5"]
    names = ["one", "two", "three", "four", "five"]
    fig, axes = plt.subplots(1, 5, figsize=(11, 2.8), sharey=True)
    for ax, w, n in zip(axes, words, names):
        fs = list(sessions_of("voice", w).values())[0]
        x = audio.read_wav(fs[len(fs) // 2])
        db = audio.log_mel(audio.fit_clip(x))
        ax.imshow(db, origin="lower", aspect="auto", cmap=BLUES, vmin=db.max() - 70, vmax=db.max(),
                  extent=[0, 1, 0, C.N_MELS])
        ax.set_title(f'"{n}"')
        ax.set_xlabel("s")
        ax.set_xticks([0, 0.5, 1])
    axes[0].set_ylabel("mel band")
    fig.suptitle("What VoiceNet sees: 1 s log-mel spectrograms (40 bands x 98 frames)",
                 x=0.02, ha="left", fontweight="bold", y=1.04)
    save(fig, "fig-voice.png")


# ---------------------------------------------------------------- 7. laptop vs board
def fig_board(board_gesture_ms=95.0, board_voice_ms=161.0):
    g, v = models.load("gesture"), models.load("voice")
    torch.set_num_threads(2)
    x = torch.zeros(1, 3, 96, 96)
    with torch.no_grad():
        for _ in range(5):
            g(x)
        t = time.time()
        for _ in range(50):
            g(x)
        lap_g = (time.time() - t) / 50 * 1000
        t = time.time()
        for _ in range(10):
            v(torch.from_numpy(audio.voice_array(np.zeros(16000, np.float32)))[None])
        lap_v = (time.time() - t) / 10 * 1000
    fig, ax = plt.subplots(figsize=(8, 2.9))
    rows = ["GestureNet\n(one frame)", "VoiceNet\n(one clip, incl. spectrogram)"]
    y = np.arange(2)
    ax.barh(y - 0.19, [lap_g, lap_v], height=0.34, color=BLUE, label="MacBook Pro (M-series)")
    ax.barh(y + 0.19, [board_gesture_ms, board_voice_ms], height=0.34, color=ORANGE,
            label="UNO Q (4x Cortex-A53)")
    for yy, val in zip(list(y - 0.19) + list(y + 0.19), [lap_g, lap_v, board_gesture_ms, board_voice_ms]):
        ax.text(val + 2, yy, f"{val:.0f} ms" if val >= 10 else f"{val:.1f} ms", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, rows)
    ax.invert_yaxis()
    ax.set_xlabel("milliseconds per inference (lower is faster)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_xlim(0, max(board_voice_ms, board_gesture_ms) * 1.18)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Same weights, same code: the board is ~50x slower per inference", loc="left")
    save(fig, "fig-board-vs-laptop.png")


if __name__ == "__main__":
    fig_pipeline()
    fig_data()
    fig_cnn()
    fig_confusion()
    fig_hold()
    fig_voice()
    fig_board()
