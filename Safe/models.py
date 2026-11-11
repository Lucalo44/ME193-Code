"""The two PyTorch nets.

GestureNet: RGB hand crop -> none / 1 / 2 / 3 / 4 / 5 fingers.
PalmNet:    [equalized gray, crease map] -> not_me / me.
VoiceNet:   log-mel spectrogram of a 1 s clip -> none / "one" .. "five".
SpeakerNet: same input -> not_me / me.
All four are the same small conv stack so they run at camera rate on a laptop CPU or the UNO Q.
"""
import torch
import torch.nn as nn

import config as C


def _block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    def __init__(self, in_ch, n_classes, widths=(16, 32, 64, 128), dropout=0.3):
        super().__init__()
        layers, c = [], in_ch
        for w in widths:
            layers.append(_block(c, w))
            c = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(c, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build(task):
    n = len(C.CLASSES[task])
    if task == "gesture":
        return SmallCNN(3, n, widths=(16, 32, 64, 128))
    if task in C.AUDIO_TASKS:
        # 1 x 40 mel bands x 98 frames; four poolings -> 2 x 6 before the average
        return SmallCNN(1, n, widths=(16, 32, 64, 128), dropout=0.3 if task == "voice" else 0.4)
    # one extra stage: the palm input is bigger and the creases are fine detail
    return SmallCNN(2, n, widths=(16, 32, 64, 128, 128), dropout=0.4)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def model_path(task):
    return C.MODEL_DIR / f"{task}.pt"


def load(task, device="cpu"):
    """Returns the trained model in eval mode, or None if it hasn't been trained yet."""
    path = model_path(task)
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location=device)
    model = build(task)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()
