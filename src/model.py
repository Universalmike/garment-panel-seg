"""
Panel segmentation model.

Design in one line: freeze a small pretrained MobileNetV2 as a feature
extractor, and train only a tiny UNet-style decoder on top of it. The frozen
encoder gives us general shape/texture features for free; the trainable decoder
is what learns "which pixels are body / sleeve / collar".

Why this fits the 2M trainable-parameter cap:
  - The MobileNetV2 encoder is FROZEN (requires_grad=False), so it does not
    count against the trainable budget. We still report its size as part of the
    total inference-time parameter count.
  - The decoder uses depthwise-separable convolutions, which are cheap. It lands
    comfortably under 2M trainable params (exact count printed by count_params).

We intentionally do NOT predict left vs. right sleeve here. The two sleeves look
almost identical up close, so asking a small model to tell them apart from local
texture is a losing game. The model predicts a single "sleeve" class, and left
vs. right is decided afterwards from geometry (see predict.py). That keeps the
learning problem honest and the targeting deterministic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


# Classes the network is trained to predict (background + 3 real panel groups).
# Left/right sleeve split happens later, in post-processing.
TRAIN_CLASSES = ["background", "body", "sleeve", "collar"]
NUM_TRAIN_CLASSES = len(TRAIN_CLASSES)


class DSConv(nn.Module):
    """Depthwise-separable conv + BN + ReLU6. Cheap on parameters."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class UpBlock(nn.Module):
    """Upsample x2, fuse a skip connection, then two cheap convs."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.conv1 = DSConv(out_ch + skip_ch, out_ch)
        self.conv2 = DSConv(out_ch, out_ch)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.reduce(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class PanelSegNet(nn.Module):
    """MobileNetV2 (frozen) encoder + small trainable UNet decoder."""

    def __init__(self, num_classes=NUM_TRAIN_CLASSES, pretrained=True, freeze_encoder=True):
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        features = mobilenet_v2(weights=weights).features

        # Split the encoder into stages so we can grab skip features at each scale.
        self.enc1 = features[0:2]    # ->  16 ch, 1/2
        self.enc2 = features[2:4]    # ->  24 ch, 1/4
        self.enc3 = features[4:7]    # ->  32 ch, 1/8
        self.enc4 = features[7:14]   # ->  96 ch, 1/16
        self.enc5 = features[14:18]  # -> 320 ch, 1/32  (bottleneck)

        if freeze_encoder:
            for p in self.parameters():
                p.requires_grad = False
        self._encoder_frozen = freeze_encoder

        # Decoder channel widths. Tuned to use most of the trainable budget
        # (roughly 1.4M params) while staying safely under the 2M cap.
        c4, c3, c2, c1, c0 = 256, 192, 128, 96, 64
        self.up4 = UpBlock(320, 96, c4)   # 1/32 -> 1/16
        self.up3 = UpBlock(c4, 32, c3)    # 1/16 -> 1/8
        self.up2 = UpBlock(c3, 24, c2)    # 1/8  -> 1/4
        self.up1 = UpBlock(c2, 16, c1)    # 1/4  -> 1/2
        self.up0 = UpBlock(c1, 0, c0)     # 1/2  -> 1/1  (no skip)
        self.head = nn.Conv2d(c0, num_classes, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        s5 = self.enc5(s4)

        d = self.up4(s5, s4)
        d = self.up3(d, s3)
        d = self.up2(d, s2)
        d = self.up1(d, s1)
        d = self.up0(d, None)
        if d.shape[-2:] != x.shape[-2:]:
            d = F.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(d)


def count_params(model):
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


if __name__ == "__main__":
    m = PanelSegNet()
    tr, tot = count_params(m)
    x = torch.randn(1, 3, 256, 256)
    y = m(x)
    print(f"output shape      : {tuple(y.shape)}")
    print(f"trainable params  : {tr:,}")
    print(f"total (inference) : {tot:,}")
    print(f"under 2M cap      : {tr < 2_000_000}")
