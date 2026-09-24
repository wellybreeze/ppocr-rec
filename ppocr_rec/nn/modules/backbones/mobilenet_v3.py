from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, if_act=True, act=None):
        super().__init__()
        self.if_act = if_act
        self.act = act
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.if_act:
            if self.act == "relu":
                x = F.relu(x)
            elif self.act == "hardswish":
                x = F.hardswish(x)
        return x


class SEModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, 1)

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = F.relu(self.conv1(outputs))
        outputs = F.hardsigmoid(self.conv2(outputs))
        return inputs * outputs


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, stride, use_se, act=None):
        super().__init__()
        stride_hw = stride if isinstance(stride, tuple) else (stride, stride)
        self.if_shortcut = stride_hw == (1, 1) and in_channels == out_channels
        self.if_se = use_se
        self.expand_conv = ConvBNLayer(in_channels, mid_channels, 1, 1, 0, if_act=True, act=act)
        self.bottleneck_conv = ConvBNLayer(
            mid_channels,
            mid_channels,
            kernel_size,
            stride,
            int((kernel_size - 1) // 2),
            groups=mid_channels,
            if_act=True,
            act=act,
        )
        if self.if_se:
            self.mid_se = SEModule(mid_channels)
        self.linear_conv = ConvBNLayer(mid_channels, out_channels, 1, 1, 0, if_act=False)

    def forward(self, inputs):
        x = self.expand_conv(inputs)
        x = self.bottleneck_conv(x)
        if self.if_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            x = inputs + x
        return x


class MobileNetV3(nn.Module):
    """PaddleOCR rec MobileNetV3 (large_stride default [1,2,2,2])."""

    def __init__(
        self,
        in_channels=3,
        model_name="small",
        scale=0.5,
        large_stride=None,
        small_stride=None,
        disable_se=False,
        **kwargs,
    ):
        super().__init__()
        self.disable_se = disable_se
        if small_stride is None:
            small_stride = [2, 2, 2, 2]
        if large_stride is None:
            large_stride = [1, 2, 2, 2]
        if model_name == "large":
            cfg = [
                [3, 16, 16, False, "relu", large_stride[0]],
                [3, 64, 24, False, "relu", (large_stride[1], 1)],
                [3, 72, 24, False, "relu", 1],
                [5, 72, 40, True, "relu", (large_stride[2], 1)],
                [5, 120, 40, True, "relu", 1],
                [5, 120, 40, True, "relu", 1],
                [3, 240, 80, False, "hardswish", 1],
                [3, 200, 80, False, "hardswish", 1],
                [3, 184, 80, False, "hardswish", 1],
                [3, 184, 80, False, "hardswish", 1],
                [3, 480, 112, True, "hardswish", 1],
                [3, 672, 112, True, "hardswish", 1],
                [5, 672, 160, True, "hardswish", (large_stride[3], 1)],
                [5, 960, 160, True, "hardswish", 1],
                [5, 960, 160, True, "hardswish", 1],
            ]
            cls_ch_squeeze = 960
        elif model_name == "small":
            cfg = [
                [3, 16, 16, True, "relu", (small_stride[0], 1)],
                [3, 72, 24, False, "relu", (small_stride[1], 1)],
                [3, 88, 24, False, "relu", 1],
                [5, 96, 40, True, "hardswish", (small_stride[2], 1)],
                [5, 240, 40, True, "hardswish", 1],
                [5, 240, 40, True, "hardswish", 1],
                [5, 120, 48, True, "hardswish", 1],
                [5, 144, 48, True, "hardswish", 1],
                [5, 288, 96, True, "hardswish", (small_stride[3], 1)],
                [5, 576, 96, True, "hardswish", 1],
                [5, 576, 96, True, "hardswish", 1],
            ]
            cls_ch_squeeze = 576
        else:
            raise NotImplementedError(model_name)

        inplanes = 16
        self.conv1 = ConvBNLayer(
            in_channels,
            make_divisible(inplanes * scale),
            3,
            2,
            1,
            if_act=True,
            act="hardswish",
        )
        inplanes = make_divisible(inplanes * scale)
        blocks = []
        for k, exp, c, se, nl, s in cfg:
            se = se and not self.disable_se
            blocks.append(
                ResidualUnit(
                    inplanes,
                    make_divisible(scale * exp),
                    make_divisible(scale * c),
                    k,
                    s,
                    se,
                    nl,
                )
            )
            inplanes = make_divisible(scale * c)
        self.blocks = nn.Sequential(*blocks)
        self.conv2 = ConvBNLayer(
            inplanes,
            make_divisible(scale * cls_ch_squeeze),
            1,
            1,
            0,
            if_act=True,
            act="hardswish",
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.out_channels = make_divisible(scale * cls_ch_squeeze)

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.pool(x)
        return x
