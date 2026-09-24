from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


NET_CONFIG_rec = {
    "blocks2": [[3, 16, 32, 1, False]],
    "blocks3": [[3, 32, 64, 1, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, (2, 1), False], [3, 128, 128, 1, False]],
    "blocks5": [
        [3, 128, 256, (1, 2), False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
    ],
    "blocks6": [
        [5, 256, 512, (2, 1), True],
        [5, 512, 512, 1, True],
        [5, 512, 512, (2, 1), False],
        [5, 512, 512, 1, False],
    ],
}


def make_divisible(v, divisor=16, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class LearnableAffineBlock(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value], dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor([bias_value], dtype=torch.float32))

    def forward(self, x):
        return self.scale * x + self.bias


class ConvBNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class Act(nn.Module):
    def __init__(self, act="hswish"):
        super().__init__()
        self.act = nn.Hardswish() if act == "hswish" else nn.ReLU()
        self.lab = LearnableAffineBlock()

    def forward(self, x):
        return self.lab(self.act(x))


class LearnableRepLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, num_conv_branches=1):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.identity = (
            nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None
        )
        self.conv_kxk = nn.ModuleList(
            [ConvBNLayer(in_channels, out_channels, kernel_size, stride, groups=groups) for _ in range(num_conv_branches)]
        )
        self.conv_1x1 = ConvBNLayer(in_channels, out_channels, 1, stride, groups=groups) if kernel_size > 1 else None
        self.lab = LearnableAffineBlock()
        self.act = Act()

    def forward(self, x):
        out = 0
        if self.identity is not None:
            out = out + self.identity(x)
        if self.conv_1x1 is not None:
            out = out + self.conv_1x1(x)
        for conv in self.conv_kxk:
            out = out + conv(x)
        out = self.lab(out)
        # Paddle: skip Hardswish only when stride is exactly int 2 (det). Rec uses (2, 1)/(1, 2).
        if self.stride != 2:
            out = self.act(out)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channel, channel // reduction, 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channel // reduction, channel, 1)
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        y = self.hardsigmoid(self.conv2(self.relu(self.conv1(self.avg_pool(x)))))
        return x * y


class LCNetV3Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dw_size, use_se=False, conv_kxk_num=4):
        super().__init__()
        self.use_se = use_se
        self.dw_conv = LearnableRepLayer(in_channels, in_channels, dw_size, stride, groups=in_channels, num_conv_branches=conv_kxk_num)
        self.se = SELayer(in_channels) if use_se else None
        self.pw_conv = LearnableRepLayer(in_channels, out_channels, 1, 1, num_conv_branches=conv_kxk_num)

    def forward(self, x):
        x = self.dw_conv(x)
        if self.se is not None:
            x = self.se(x)
        return self.pw_conv(x)


class PPLCNetV3(nn.Module):
    def __init__(self, in_channels=3, scale=1.0, conv_kxk_num=4, det=False, **kwargs):
        super().__init__()
        self.det = det
        cfg = NET_CONFIG_rec
        self.conv1 = ConvBNLayer(in_channels, make_divisible(16 * scale), 3, 2)

        def make_stage(key):
            return nn.Sequential(
                *[
                    LCNetV3Block(
                        make_divisible(in_c * scale),
                        make_divisible(out_c * scale),
                        s,
                        k,
                        se,
                        conv_kxk_num,
                    )
                    for k, in_c, out_c, s, se in cfg[key]
                ]
            )

        self.blocks2 = make_stage("blocks2")
        self.blocks3 = make_stage("blocks3")
        self.blocks4 = make_stage("blocks4")
        self.blocks5 = make_stage("blocks5")
        self.blocks6 = make_stage("blocks6")
        self.out_channels = make_divisible(512 * scale)

    def forward(self, x):
        x = self.blocks6(self.blocks5(self.blocks4(self.blocks3(self.blocks2(self.conv1(x))))))
        if self.training:
            x = F.adaptive_avg_pool2d(x, (1, 40))
        else:
            x = F.avg_pool2d(x, (3, 2))
        return x
