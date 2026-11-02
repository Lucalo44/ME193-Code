"""
StepNet: a small temporal convolutional network that turns the fused sensor
stream into a step density.

    input  (batch, channels, T)   normalised fuse.build() output, 50 Hz
    output (batch, T)             >= 0, "how much of a step happens at this sample"

The step count over any stretch of time is the sum of the density over it, so
counting needs no peak picking and no thresholds, and windows can be stitched
together without double counting.

Layout: 1x1 conv in -> residual blocks with dilated convs (dilation 1, 2, 4, 8,
16: each block sees twice as far as the last) -> 1x1 conv -> softplus.
No normalisation layers: GroupNorm rescales every crop and erases how hard the
leg moves (standing still would look like walking), and BatchNorm's training
statistics don't match inference on whole takes. Five residual blocks train
fine without either, since fuse.fit_norm already scales the inputs.
With kernel 5 the receptive field is 1 + 4 * 31 = 125 samples (2.5 s),
enough to see a couple of strides around every sample.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import config as C


class Block(nn.Module):
    def __init__(self, ch, kernel, dilation, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, kernel, dilation=dilation, padding=dilation * (kernel - 1) // 2)
        self.mix = nn.Conv1d(ch, ch, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.mix(self.drop(F.gelu(self.conv(x))))


class StepNet(nn.Module):
    def __init__(self, in_ch, hidden=C.HIDDEN, kernel=C.KERNEL, dilations=tuple(C.DILATIONS)):
        super().__init__()
        self.cfg = {"in_ch": in_ch, "hidden": hidden, "kernel": kernel, "dilations": list(dilations)}
        self.inp = nn.Conv1d(in_ch, hidden, 1)
        self.blocks = nn.ModuleList(Block(hidden, kernel, d) for d in dilations)
        self.out = nn.Conv1d(hidden, 1, 1)
        nn.init.constant_(self.out.bias, -4.0)       # start near zero steps, not 0.7 per sample

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return F.softplus(self.out(h)).squeeze(1)

    @property
    def receptive_field(self):
        return 1 + (self.cfg["kernel"] - 1) * sum(self.cfg["dilations"])


def save(model, path, norm, channels, extra=None):
    torch.save({"state": model.state_dict(), "cfg": model.cfg, "norm": norm, "channels": channels,
                "fs": C.FS, **(extra or {})}, path)


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = StepNet(**ck["cfg"])
    model.load_state_dict(ck["state"])
    model.eval()
    return model, ck


@torch.no_grad()
def predict_density(model, Z):
    """Z: (T, C) normalised input -> (T,) numpy density. Whole sequence in one pass."""
    x = torch.from_numpy(np.ascontiguousarray(Z.T))[None]
    return model(x)[0].numpy()
