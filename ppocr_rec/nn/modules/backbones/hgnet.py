from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, use_act=True):
        super().__init__()
        self.use_act = use_act
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU() if use_act else nn.Identity()

    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.act(x) if self.use_act else x


class ESEModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.conv(self.avg_pool(x)))


class HG_Block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, layer_num, identity=False):
        super().__init__()
        self.identity = identity
        layers = [ConvBNAct(in_channels, mid_channels, 3, 1)]
        for _ in range(layer_num - 1):
            layers.append(ConvBNAct(mid_channels, mid_channels, 3, 1))
        self.layers = nn.ModuleList(layers)
        total_channels = in_channels + layer_num * mid_channels
        self.aggregation_conv = ConvBNAct(total_channels, out_channels, 1, 1)
        self.att = ESEModule(out_channels)

    def forward(self, x):
        identity = x
        output = [x]
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = self.att(self.aggregation_conv(torch.cat(output, 1)))
        if self.identity:
            x = x + identity
        return x


class HG_Stage(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, block_num, layer_num, downsample=True, stride=(2, 1)):
        super().__init__()
        self.do_downsample = downsample
        if downsample:
            if not isinstance(stride, int):
                stride = tuple(stride)
            self.downsample = ConvBNAct(in_channels, in_channels, 3, stride, groups=in_channels, use_act=False)
        blocks = [HG_Block(in_channels, mid_channels, out_channels, layer_num, False)]
        for _ in range(block_num - 1):
            blocks.append(HG_Block(out_channels, mid_channels, out_channels, layer_num, True))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        if self.do_downsample:
            x = self.downsample(x)
        return self.blocks(x)


class PPHGNet(nn.Module):
    def __init__(self, stem_channels, stage_config, layer_num, in_channels=3, det=False, **kwargs):
        super().__init__()
        self.det = det
        chs = [in_channels] + list(stem_channels)
        self.stem = nn.Sequential(
            *[ConvBNAct(chs[i], chs[i + 1], 3, 2 if i == 0 else 1) for i in range(len(chs) - 1)]
        )
        self.stages = nn.ModuleList()
        for k in stage_config:
            in_c, mid_c, out_c, block_num, downsample, stride = stage_config[k]
            self.stages.append(HG_Stage(in_c, mid_c, out_c, block_num, layer_num, downsample, stride))
        self.out_channels = stage_config["stage4"][2]

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        if self.training:
            x = F.adaptive_avg_pool2d(x, (1, 40))
        else:
            x = F.avg_pool2d(x, (3, 2))
        return x


def PPHGNet_small(in_channels=3, det=False, **kwargs):
    stage_config = {
        "stage1": [128, 128, 256, 1, True, [2, 1]],
        "stage2": [256, 160, 512, 1, True, [1, 2]],
        "stage3": [512, 192, 768, 2, True, [2, 1]],
        "stage4": [768, 224, 1024, 1, True, [2, 1]],
    }
    return PPHGNet(stem_channels=[64, 64, 128], stage_config=stage_config, layer_num=6, in_channels=in_channels, det=det)
