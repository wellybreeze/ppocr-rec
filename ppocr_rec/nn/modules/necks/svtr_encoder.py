from __future__ import annotations

import torch
import torch.nn as nn

from ppocr_rec.nn.modules.necks.svtr_block import Block, ConvBNLayer


class EncoderWithSVTR(nn.Module):
    def __init__(
        self,
        in_channels,
        dims=64,
        depth=2,
        hidden_dims=120,
        use_guide=False,
        num_heads=8,
        qkv_bias=True,
        mlp_ratio=2.0,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path=0.0,
        kernel_size=(3, 3),
        qk_scale=None,
        **kwargs,
    ):
        super().__init__()
        self.use_guide = use_guide
        ks = kernel_size if not isinstance(kernel_size, list) else tuple(kernel_size)
        self.conv1 = ConvBNLayer(in_channels, in_channels // 8, ks, padding=(ks[0] // 2, ks[1] // 2), act=nn.SiLU)
        self.conv2 = ConvBNLayer(in_channels // 8, hidden_dims, 1, act=nn.SiLU)
        self.svtr_block = nn.ModuleList(
            [
                Block(
                    dim=hidden_dims,
                    num_heads=num_heads,
                    mixer="Global",
                    HW=None,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path,
                    epsilon=1e-5,
                    prenorm=False,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dims, eps=1e-6)
        self.conv3 = ConvBNLayer(hidden_dims, in_channels, 1, act=nn.SiLU)
        self.conv4 = ConvBNLayer(2 * in_channels, in_channels // 8, ks, padding=(ks[0] // 2, ks[1] // 2), act=nn.SiLU)
        self.conv1x1 = ConvBNLayer(in_channels // 8, dims, 1, act=nn.SiLU)
        self.out_channels = dims

    def forward(self, x):
        z = x.detach() if self.use_guide else x
        h = z
        z = self.conv2(self.conv1(z))
        b, c, hh, ww = z.shape
        z = z.flatten(2).transpose(1, 2)
        for blk in self.svtr_block:
            z = blk(z)
        z = self.norm(z).reshape(b, hh, ww, c).permute(0, 3, 1, 2)
        z = self.conv3(z)
        z = self.conv1x1(self.conv4(torch.cat([h, z], dim=1)))
        return z


class EncoderWithLightSVTR(nn.Module):
    def __init__(
        self,
        in_channels,
        dims=64,
        depth=1,
        num_heads=8,
        qkv_bias=True,
        mlp_ratio=4.0,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path=0.0,
        qk_scale=None,
        local_kernel=7,
        use_guide=False,
        **kwargs,
    ):
        super().__init__()
        self.use_guide = use_guide
        self.conv_reduce = ConvBNLayer(in_channels, dims, 1, act=nn.SiLU)
        self.local_conv = nn.Sequential(
            nn.Conv2d(dims, dims, (1, local_kernel), padding=(0, local_kernel // 2), groups=dims, bias=False),
            nn.BatchNorm2d(dims),
            nn.SiLU(),
        )
        self.svtr_block = nn.ModuleList(
            [
                Block(
                    dim=dims,
                    num_heads=num_heads,
                    mixer="Global",
                    HW=None,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path,
                    epsilon=1e-5,
                    prenorm=False,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dims, eps=1e-6)
        self.skip_conv = ConvBNLayer(in_channels, dims, 1, act=nn.SiLU)
        self.out_channels = dims

    def forward(self, x):
        if self.use_guide:
            x = x.detach()
        skip = self.skip_conv(x)
        z = self.conv_reduce(x)
        z = z + self.local_conv(z)
        b, c, h, w = z.shape
        z = z.flatten(2).transpose(1, 2)
        for blk in self.svtr_block:
            z = blk(z)
        z = self.norm(z).reshape(b, h, w, c).permute(0, 3, 1, 2)
        return z + skip
