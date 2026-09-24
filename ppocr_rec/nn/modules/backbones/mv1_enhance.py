from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class ConvBNLayer(nn.Module):
    def __init__(self, num_channels, filter_size, num_filters, stride, padding, num_groups=1, act="hard_swish"):
        super().__init__()
        self.conv = nn.Conv2d(num_channels, num_filters, filter_size, stride, padding, groups=num_groups, bias=False)
        self.bn = nn.BatchNorm2d(num_filters)
        self.act = act

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.act == "hard_swish":
            x = F.hardswish(x)
        elif self.act == "relu":
            x = F.relu(x)
        return x


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channel, channel // reduction, 1)
        self.conv2 = nn.Conv2d(channel // reduction, channel, 1)

    def forward(self, inputs):
        outputs = F.hardsigmoid(self.conv2(F.relu(self.conv1(self.avg_pool(inputs)))))
        return inputs * outputs


class DepthwiseSeparable(nn.Module):
    def __init__(self, num_channels, num_filters1, num_filters2, num_groups, stride, scale, dw_size=3, padding=1, use_se=False):
        super().__init__()
        self.use_se = use_se
        self.dw = ConvBNLayer(num_channels, dw_size, int(num_filters1 * scale), stride, padding, num_groups=int(num_groups * scale))
        self.se = SEModule(int(num_filters1 * scale)) if use_se else None
        self.pw = ConvBNLayer(int(num_filters1 * scale), 1, int(num_filters2 * scale), 1, 0)

    def forward(self, x):
        x = self.dw(x)
        if self.se is not None:
            x = self.se(x)
        return self.pw(x)


class MobileNetV1Enhance(nn.Module):
    def __init__(self, in_channels=3, scale=0.5, last_conv_stride=1, last_pool_type="max", last_pool_kernel_size=(3, 2), **kwargs):
        super().__init__()
        self.conv1 = ConvBNLayer(in_channels, 3, int(32 * scale), 2, 1)
        blocks = [
            DepthwiseSeparable(int(32 * scale), 32, 64, 32, 1, scale),
            DepthwiseSeparable(int(64 * scale), 64, 128, 64, 1, scale),
            DepthwiseSeparable(int(128 * scale), 128, 128, 128, 1, scale),
            DepthwiseSeparable(int(128 * scale), 128, 256, 128, (2, 1), scale),
            DepthwiseSeparable(int(256 * scale), 256, 256, 256, 1, scale),
            DepthwiseSeparable(int(256 * scale), 256, 512, 256, (2, 1), scale),
        ]
        for _ in range(5):
            blocks.append(DepthwiseSeparable(int(512 * scale), 512, 512, 512, 1, scale, dw_size=5, padding=2))
        blocks.append(DepthwiseSeparable(int(512 * scale), 512, 1024, 512, (2, 1), scale, dw_size=5, padding=2, use_se=True))
        blocks.append(
            DepthwiseSeparable(
                int(1024 * scale), 1024, 1024, 1024, last_conv_stride if not isinstance(last_conv_stride, list) else tuple(last_conv_stride),
                scale, dw_size=5, padding=2, use_se=True,
            )
        )
        self.block_list = nn.Sequential(*blocks)
        ksz = last_pool_kernel_size if not isinstance(last_pool_kernel_size, list) else tuple(last_pool_kernel_size)
        if last_pool_type == "avg":
            self.pool = nn.AvgPool2d(kernel_size=ksz, stride=ksz, padding=0)
        else:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.out_channels = int(1024 * scale)

    def forward(self, x):
        return self.pool(self.block_list(self.conv1(x)))
