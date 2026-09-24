from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _same_k2(x: torch.Tensor) -> torch.Tensor:
    """Paddle Conv2D / MaxPool2D padding='SAME' for kernel=2, stride=1."""
    return F.pad(x, [0, 1, 0, 1])


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=1,
        use_act=True,
        use_lab=False,
        **kwargs,
    ):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab
        self.same_pad = isinstance(padding, str) and str(padding).upper() == "SAME"
        conv_padding = 0 if self.same_pad else (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, conv_padding, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        if use_act:
            self.act = nn.ReLU()
            if use_lab:
                self.lab = nn.Identity()

    def forward(self, x):
        if self.same_pad:
            x = _same_k2(x)
        x = self.bn(self.conv(x))
        if self.use_act:
            x = self.act(x)
            if self.use_lab:
                x = self.lab(x)
        return x


class LightConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, use_lab=False, **kwargs):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels, 1, 1, padding=0, use_act=False, use_lab=use_lab)
        self.conv2 = ConvBNAct(
            out_channels, out_channels, kernel_size, 1, groups=out_channels, use_act=True, use_lab=use_lab
        )

    def forward(self, x):
        return self.conv2(self.conv1(x))


class StemBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, use_lab=False, text_rec=False, **kwargs):
        super().__init__()
        self.stem1 = ConvBNAct(in_channels, mid_channels, 3, 2, use_lab=use_lab)
        self.stem2a = ConvBNAct(mid_channels, mid_channels // 2, 2, 1, padding="SAME", use_lab=use_lab)
        self.stem2b = ConvBNAct(mid_channels // 2, mid_channels, 2, 1, padding="SAME", use_lab=use_lab)
        self.stem3 = ConvBNAct(mid_channels * 2, mid_channels, 3, 1 if text_rec else 2, use_lab=use_lab)
        self.stem4 = ConvBNAct(mid_channels, out_channels, 1, 1, padding=0, use_lab=use_lab)

    def forward(self, x):
        x = self.stem1(x)
        x2 = self.stem2b(self.stem2a(x))
        x1 = F.max_pool2d(_same_k2(x), kernel_size=2, stride=1)
        if x1.shape[2:] != x2.shape[2:]:
            x2 = F.interpolate(x2, size=x1.shape[2:], mode="nearest")
        return self.stem4(self.stem3(torch.cat([x1, x2], 1)))


class HGV2_Block(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        kernel_size=3,
        layer_num=6,
        identity=False,
        light_block=True,
        use_lab=False,
        **kwargs,
    ):
        super().__init__()
        self.identity = identity
        block_cls = LightConvBNAct if light_block else ConvBNAct
        layers = []
        for i in range(layer_num):
            layers.append(
                block_cls(
                    in_channels=in_channels if i == 0 else mid_channels,
                    out_channels=mid_channels,
                    kernel_size=kernel_size,
                    use_lab=use_lab,
                )
            )
        self.layers = nn.ModuleList(layers)
        total_channels = in_channels + layer_num * mid_channels
        self.aggregation_squeeze_conv = ConvBNAct(total_channels, out_channels // 2, 1, 1, padding=0, use_lab=use_lab)
        self.aggregation_excitation_conv = ConvBNAct(out_channels // 2, out_channels, 1, 1, padding=0, use_lab=use_lab)

    def forward(self, x):
        identity = x
        output = [x]
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = self.aggregation_excitation_conv(self.aggregation_squeeze_conv(torch.cat(output, 1)))
        if self.identity:
            x = x + identity
        return x


class HGV2_Stage(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        block_num,
        layer_num=6,
        is_downsample=True,
        light_block=True,
        kernel_size=3,
        use_lab=False,
        stride=2,
        **kwargs,
    ):
        super().__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            if not isinstance(stride, int):
                stride = tuple(stride)
            self.downsample = ConvBNAct(
                in_channels, in_channels, 3, stride, groups=in_channels, use_act=False, use_lab=use_lab
            )
        blocks = []
        for i in range(block_num):
            blocks.append(
                HGV2_Block(
                    in_channels=in_channels if i == 0 else out_channels,
                    mid_channels=mid_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    layer_num=layer_num,
                    identity=i != 0,
                    light_block=light_block,
                    use_lab=use_lab,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        if self.is_downsample:
            x = self.downsample(x)
        return self.blocks(x)


class PPHGNetV2(nn.Module):
    def __init__(
        self,
        stage_config,
        stem_channels=(3, 32, 48),
        use_lab=False,
        det=False,
        text_rec=False,
        **kwargs,
    ):
        super().__init__()
        self.det = det
        self.text_rec = text_rec
        self.stem = StemBlock(
            in_channels=stem_channels[0],
            mid_channels=stem_channels[1],
            out_channels=stem_channels[2],
            use_lab=use_lab,
            text_rec=text_rec,
        )
        self.stages = nn.ModuleList()
        for k in stage_config:
            in_c, mid_c, out_c, block_num, is_down, light_block, ksz, layer_num, stride = stage_config[k]
            self.stages.append(
                HGV2_Stage(
                    in_c, mid_c, out_c, block_num, layer_num, is_down, light_block, ksz, use_lab, stride
                )
            )
        self.out_channels = stage_config["stage4"][2]

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        if self.det:
            return x
        if self.text_rec:
            if self.training:
                x = F.adaptive_avg_pool2d(x, (1, 40))
            else:
                x = F.avg_pool2d(x, (3, 2))
        return x


def PPHGNetV2_B4(in_channels=3, det=False, text_rec=True, **kwargs):
    stage_config_rec = {
        "stage1": [48, 48, 128, 1, True, False, 3, 6, [2, 1]],
        "stage2": [128, 96, 512, 1, True, False, 3, 6, [1, 2]],
        "stage3": [512, 192, 1024, 3, True, True, 5, 6, [2, 1]],
        "stage4": [1024, 384, 2048, 1, True, True, 5, 6, [2, 1]],
    }
    stage_config_det = {
        "stage1": [48, 48, 128, 1, False, False, 3, 6, 2],
        "stage2": [128, 96, 512, 1, True, False, 3, 6, 2],
        "stage3": [512, 192, 1024, 3, True, True, 5, 6, 2],
        "stage4": [1024, 384, 2048, 1, True, True, 5, 6, 2],
    }
    return PPHGNetV2(
        stem_channels=(in_channels, 32, 48),
        stage_config=stage_config_det if det else stage_config_rec,
        use_lab=False,
        det=det,
        text_rec=text_rec,
        **kwargs,
    )
