# model_simple.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_ch, out_ch, stride=1, dilation=1):
    return nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=dilation, dilation=dilation, bias=False)

class PreActResBlock(nn.Module):
    """Pre-activation residual block with InstanceNorm (light 'Improved' touch)."""
    def __init__(self, ch):
        super().__init__()
        self.in1 = nn.InstanceNorm2d(ch, affine=True, eps=1e-5)
        self.in2 = nn.InstanceNorm2d(ch, affine=True, eps=1e-5)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.c1 = conv3x3(ch, ch)
        self.c2 = conv3x3(ch, ch)

    def forward(self, x):
        y = self.c1(self.act(self.in1(x)))
        y = self.c2(self.act(self.in2(y)))
        return x + y

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = conv3x3(in_ch, out_ch, stride=2)
        self.block = PreActResBlock(out_ch)

    def forward(self, x):
        return self.block(self.conv(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = conv3x3(in_ch, out_ch)
        self.block = PreActResBlock(out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return self.block(x)

class UNetImproved2D(nn.Module):
    """Minimal Improved U-Net (no deep supervision yet, we’ll add later)."""
    def __init__(self, in_ch=1, n_classes=4, base=32):
        super().__init__()
        c1, c2, c3, c4 = base, base*2, base*4, base*8

        self.stem = nn.Sequential(conv3x3(in_ch, c1), PreActResBlock(c1))
        self.d1 = Down(c1, c2)
        self.d2 = Down(c2, c3)
        self.d3 = Down(c3, c4)

        self.bott = nn.Sequential(PreActResBlock(c4), PreActResBlock(c4))

        self.u1 = Up(c4 + c3, c3)
        self.u2 = Up(c3 + c2, c2)
        self.u3 = Up(c2 + c1, c1)

        self.head = nn.Conv2d(c1, n_classes, 1)

    def forward(self, x):
        s1 = self.stem(x)
        s2 = self.d1(s1)
        s3 = self.d2(s2)
        s4 = self.d3(s3)

        b = self.bott(s4)

        d3 = self.u1(b, s3)
        d2 = self.u2(d3, s2)
        d1 = self.u3(d2, s1)

        logits = self.head(d1)
        return logits
