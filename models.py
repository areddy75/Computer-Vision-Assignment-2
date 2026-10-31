"""Model definitions and preprocessing shared by the notebook and predict.py.

The code here is identical to the notebook (Assignment2_SSL.ipynb, section 3) so that
checkpoints saved there load here.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

STORE = 72  # images are cached at 72x72 grayscale
IMG = 64    # network input


def conv_bn(i, o, s):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class ResBlock(nn.Module):
    def __init__(self, i, o, s):
        super().__init__()
        self.a = conv_bn(i, o, s)
        self.b = nn.Sequential(nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o))
        self.sc = nn.Identity() if (i == o and s == 1) else nn.Sequential(nn.Conv2d(i, o, 1, s, bias=False), nn.BatchNorm2d(o))

    def forward(self, x):
        return F.relu(self.b(self.a(x)) + self.sc(x))


class Backbone(nn.Module):
    """Small ResNet for 64x64 grayscale input: stride-2 stem, 4 residual stages (32-256 ch), GAP -> 256-d."""
    def __init__(self, w=32):
        super().__init__()
        self.net = nn.Sequential(conv_bn(1, w, 2), ResBlock(w, w, 1), ResBlock(w, 2 * w, 2),
                                 ResBlock(2 * w, 4 * w, 2), ResBlock(4 * w, 8 * w, 2),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.dim = 8 * w

    def forward(self, x):
        return self.net(x)


class Net(nn.Module):
    """backbone + linear classifier + SimCLR projector (+ optional rotation head)."""
    def __init__(self, n_cls=16, proj_dim=128, rot_head=False):
        super().__init__()
        self.backbone = Backbone(); d = self.backbone.dim
        self.head = nn.Linear(d, n_cls)
        self.proj = nn.Sequential(nn.Linear(d, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, proj_dim))
        self.rot = nn.Linear(d, 4) if rot_head else None

    def forward(self, x):
        return self.head(self.backbone(x))


def load_img(path):
    """Grayscale, resize short side to 72, centre-crop 72x72 -> uint8 tensor (1,72,72)."""
    im = Image.open(path).convert("L")
    w, h = im.size; s = STORE / min(w, h)
    im = im.resize((max(STORE, round(w * s)), max(STORE, round(h * s))), Image.BICUBIC)
    w, h = im.size; l, t = (w - STORE) // 2, (h - STORE) // 2
    im = im.crop((l, t, l + STORE, t + STORE))
    return torch.from_numpy(np.asarray(im, dtype=np.uint8).copy())[None]


def preprocess(xb, mean, std):
    """uint8 (n,1,72,72) -> normalised float (n,1,64,64), same as the notebook's 'eval' transform."""
    x = F.interpolate(xb.float() / 255, size=(IMG, IMG), mode="bilinear", antialias=True, align_corners=False)
    return (x - mean) / std
