from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


NET_CONFIG_REC = {
    "tiny": {
        "stem": (24, 48),
        "stem_type": "simple",
        "blocks2": [[3, 48, 48, 1, True]],
        "blocks3": [[3, 48, 48, 1, False]],
        "blocks4": [
            [3, 48, 96, (2, 1), False],
            [3, 96, 96, 1, True],
            [3, 96, 96, 1, False],
        ],
        "blocks5": [
            [3, 96, 160, (2, 1), False],
            [3, 160, 160, 1, True],
            [3, 160, 160, 1, False],
            [3, 160, 160, 1, False],
        ],
        "blocks6": [],
    },
    "small": {
        "stem": (48, 96),
        "stem_type": "branch",
        "blocks2": [[3, 96, 96, 1, True]],
        "blocks3": [[3, 96, 96, 1, False], [3, 96, 96, 1, False]],
        "blocks4": [
            [3, 96, 192, (2, 1), False],
            [3, 192, 192, 1, True],
            [3, 192, 192, 1, False],
            [3, 192, 192, 1, True],
            [3, 192, 192, 1, False],
            [3, 192, 192, 1, True],
            [3, 192, 192, 1, False],
        ],
        "blocks5": [
            [3, 192, 384, (2, 1), False],
            [3, 384, 384, 1, True],
            [3, 384, 384, 1, False],
        ],
        "blocks6": [],
    },
    "medium": {
        "stem": (64, 128),
        "stem_type": "branch",
        "blocks2": [[3, 128, 128, 1, True]],
        "blocks3": [
            [3, 128, 256, 1, False],
            [3, 256, 256, 1, False],
            [3, 256, 256, 1, True],
        ],
        "blocks4": [
            [3, 256, 512, (2, 1), False],
            [3, 512, 512, 1, True],
            [3, 512, 512, 1, False],
            [3, 512, 512, 1, True],
            [3, 512, 512, 1, False],
            [3, 512, 512, 1, True],
            [3, 512, 512, 1, False],
        ],
        "blocks5": [
            [3, 512, 768, (2, 1), False],
            [3, 768, 768, 1, True],
            [3, 768, 768, 1, False],
        ],
        "blocks6": [],
    },
}


class Conv2D_BN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1):
        super().__init__()
        self.add_module(
            "conv",
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
        )
        self.add_module("bn", nn.BatchNorm2d(out_channels))


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, groups=1, use_act=True):
        super().__init__()
        self.use_act = use_act
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU() if use_act else nn.Identity()

    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.act(x) if self.use_act else x


class StemBlock(nn.Module):
    """Multi-branch stem, total stride 4 (stem1 s=2 + stem3 s=2)."""

    def __init__(self, in_channels=3, mid_channels=48, out_channels=96):
        super().__init__()
        self.stem1 = ConvBNAct(in_channels, mid_channels, 3, 2, use_act=True)
        self.stem2a = ConvBNAct(mid_channels, mid_channels // 2, 2, 1, padding=0, use_act=True)
        self.stem2b = ConvBNAct(mid_channels // 2, mid_channels, 2, 1, padding=0, use_act=True)
        self.stem3 = ConvBNAct(mid_channels * 2, mid_channels, 3, 2, use_act=True)
        self.stem4 = ConvBNAct(mid_channels, out_channels, 1, 1, padding=0, use_act=True)

    def _same_k2(self, x):
        return F.pad(x, [0, 1, 0, 1])

    def forward(self, x):
        x = self.stem1(x)
        x2 = self.stem2b(self._same_k2(self.stem2a(self._same_k2(x))))
        x1 = F.max_pool2d(self._same_k2(x), kernel_size=2, stride=1)
        if x1.shape[2:] != x2.shape[2:]:
            x2 = F.interpolate(x2, size=x1.shape[2:], mode="nearest")
        x = self.stem4(self.stem3(torch.cat([x1, x2], 1)))
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel // reduction, 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channel // reduction, channel, 1)
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        y = x.mean(dim=(2, 3), keepdim=True)
        y = self.hardsigmoid(self.conv2(self.relu(self.conv1(y))))
        return x * y


class RepDWConv(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = Conv2D_BN(channels, channels, kernel_size, 1, padding, groups=channels)
        self.conv1 = nn.Conv2d(channels, channels, 1, 1, 0, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.bn(self.conv(x) + self.conv1(x) + x)


class LCNetV4Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dw_size, use_se=False, expand_ratio=2):
        super().__init__()
        self.has_residual = in_channels == out_channels and stride == 1
        self.use_rep_dw = stride == 1 and in_channels == out_channels
        mixers = []
        if self.use_rep_dw:
            mixers.append(("rep_dw", RepDWConv(in_channels, dw_size)))
        else:
            padding = (dw_size - 1) // 2
            mixers.append(("dw_conv", Conv2D_BN(in_channels, in_channels, dw_size, stride, padding, groups=in_channels)))
        if use_se:
            mixers.append(("se", SELayer(in_channels)))
        self.token_mixer = nn.Sequential()
        for name, mod in mixers:
            self.token_mixer.add_module(name, mod)
        hidden = int(in_channels * expand_ratio)
        self.channel_mixer = nn.Sequential()
        self.channel_mixer.add_module("expand", Conv2D_BN(in_channels, hidden, 1, 1, 0))
        self.channel_mixer.add_module("act", nn.GELU())
        self.channel_mixer.add_module("compress", Conv2D_BN(hidden, out_channels, 1, 1, 0))

    def forward(self, x):
        x = self.token_mixer(x)
        if self.has_residual:
            return x + self.channel_mixer(x)
        return self.channel_mixer(x)


class PPLCNetV4(nn.Module):
    def __init__(self, in_channels=3, model_size="small", det=False, **kwargs):
        super().__init__()
        if det:
            raise NotImplementedError("PPLCNetV4 detection backbone is not included")
        cfg = NET_CONFIG_REC[model_size]
        stem_mid, stem_out = cfg["stem"]
        if cfg["stem_type"] == "branch":
            self.conv1 = StemBlock(in_channels, stem_mid, stem_out)
        else:
            self.conv1 = nn.Sequential(
                Conv2D_BN(in_channels, stem_mid, 3, 2, 1),
                nn.GELU(),
                Conv2D_BN(stem_mid, stem_out, 3, 2, 1),
            )

        def make_stage(stage_name):
            return nn.Sequential(
                *[LCNetV4Block(in_c, out_c, s, k, se, expand_ratio=2) for k, in_c, out_c, s, se in cfg.get(stage_name, [])]
            )

        self.blocks2 = make_stage("blocks2")
        self.blocks3 = make_stage("blocks3")
        self.blocks4 = make_stage("blocks4")
        self.blocks5 = make_stage("blocks5")
        self.blocks6 = make_stage("blocks6")
        self.out_channels = None
        for sname in ("blocks6", "blocks5", "blocks4", "blocks3", "blocks2"):
            if cfg.get(sname):
                self.out_channels = cfg[sname][-1][2]
                break

    def forward(self, x):
        x = self.blocks6(self.blocks5(self.blocks4(self.blocks3(self.blocks2(self.conv1(x))))))
        if self.training:
            x = F.adaptive_avg_pool2d(x, (1, 40))
        else:
            x = F.avg_pool2d(x, (3, 2))
        return x
