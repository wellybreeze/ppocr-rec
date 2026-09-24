from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ppocr_rec.nn.modules.necks.svtr_block import Block, ConvBNLayer


class PatchEmbed(nn.Module):
    def __init__(self, img_size=(32, 100), in_channels=3, embed_dim=768, sub_num=2):
        super().__init__()
        self.img_size = img_size
        self.num_patches = (img_size[1] // (2**sub_num)) * (img_size[0] // (2**sub_num))
        if sub_num == 2:
            self.proj = nn.Sequential(
                ConvBNLayer(in_channels, embed_dim // 2, 3, 2, 1, act=nn.GELU),
                ConvBNLayer(embed_dim // 2, embed_dim, 3, 2, 1, act=nn.GELU),
            )
        else:
            self.proj = nn.Sequential(
                ConvBNLayer(in_channels, embed_dim // 4, 3, 2, 1, act=nn.GELU),
                ConvBNLayer(embed_dim // 4, embed_dim // 2, 3, 2, 1, act=nn.GELU),
                ConvBNLayer(embed_dim // 2, embed_dim, 3, 2, 1, act=nn.GELU),
            )

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class SubSample(nn.Module):
    def __init__(self, in_channels, out_channels, types="Conv", stride=(2, 1)):
        super().__init__()
        self.types = types
        if types == "Pool":
            self.avgpool = nn.AvgPool2d(kernel_size=(3, 5), stride=stride, padding=(1, 2))
            self.maxpool = nn.MaxPool2d(kernel_size=(3, 5), stride=stride, padding=(1, 2))
            self.proj = nn.Linear(in_channels, out_channels)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        if self.types == "Pool":
            x = 0.5 * (self.avgpool(x) + self.maxpool(x))
            out = self.proj(x.flatten(2).transpose(1, 2))
        else:
            x = self.conv(x)
            out = x.flatten(2).transpose(1, 2)
        return self.norm(out)


class SVTRNet(nn.Module):
    def __init__(
        self,
        img_size=(32, 100),
        in_channels=3,
        embed_dim=(64, 128, 256),
        depth=(3, 6, 3),
        num_heads=(2, 4, 8),
        mixer=None,
        local_mixer=((7, 11), (7, 11), (7, 11)),
        patch_merging="Conv",
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        last_drop=0.1,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        out_channels=192,
        out_char_num=25,
        last_stage=True,
        sub_num=2,
        prenorm=True,
        **kwargs,
    ):
        super().__init__()
        img_size = tuple(img_size)
        embed_dim = list(embed_dim)
        depth = list(depth)
        num_heads = list(num_heads)
        if mixer is None:
            mixer = ["Local"] * 6 + ["Global"] * 6
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.prenorm = prenorm
        self.patch_merging = patch_merging if patch_merging in {"Conv", "Pool"} else None
        self.patch_embed = PatchEmbed(img_size, in_channels, embed_dim[0], sub_num)
        self.HW = [img_size[0] // (2**sub_num), img_size[1] // (2**sub_num)]
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim[0]))
        self.pos_drop = nn.Dropout(drop_rate)
        dpr = np.linspace(0, drop_path_rate, sum(depth))

        def make_blocks(dim, heads, mix, hw, local_k, dpr_slice, n):
            return nn.ModuleList(
                [
                    Block(
                        dim=dim,
                        num_heads=heads,
                        mixer=mix[i],
                        HW=hw,
                        local_mixer=local_k,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=float(dpr_slice[i]),
                        prenorm=prenorm,
                    )
                    for i in range(n)
                ]
            )

        self.blocks1 = make_blocks(embed_dim[0], num_heads[0], mixer[0 : depth[0]], self.HW, local_mixer[0], dpr[0 : depth[0]], depth[0])
        hw = [self.HW[0] // 2, self.HW[1]] if self.patch_merging else self.HW
        if self.patch_merging:
            self.sub_sample1 = SubSample(embed_dim[0], embed_dim[1], types=self.patch_merging)
        self.blocks2 = make_blocks(
            embed_dim[1], num_heads[1], mixer[depth[0] : depth[0] + depth[1]], hw, local_mixer[1], dpr[depth[0] : depth[0] + depth[1]], depth[1]
        )
        hw2 = [self.HW[0] // 4, self.HW[1]] if self.patch_merging else self.HW
        if self.patch_merging:
            self.sub_sample2 = SubSample(embed_dim[1], embed_dim[2], types=self.patch_merging)
        self.blocks3 = make_blocks(
            embed_dim[2], num_heads[2], mixer[depth[0] + depth[1] :], hw2, local_mixer[2], dpr[depth[0] + depth[1] :], depth[2]
        )
        self.last_stage = last_stage
        if last_stage:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, out_char_num))
            self.last_conv = nn.Conv2d(embed_dim[2], out_channels, 1, bias=False)
            self.hardswish = nn.Hardswish()
            self.dropout = nn.Dropout(last_drop)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks1:
            x = blk(x)
        if self.patch_merging is not None:
            x = self.sub_sample1(x.transpose(1, 2).reshape(-1, self.embed_dim[0], self.HW[0], self.HW[1]))
        for blk in self.blocks2:
            x = blk(x)
        if self.patch_merging is not None:
            x = self.sub_sample2(x.transpose(1, 2).reshape(-1, self.embed_dim[1], self.HW[0] // 2, self.HW[1]))
        for blk in self.blocks3:
            x = blk(x)
        if self.last_stage:
            h = self.HW[0] // 4 if self.patch_merging else self.HW[0]
            x = x.transpose(1, 2).reshape(-1, self.embed_dim[2], h, self.HW[1])
            x = self.dropout(self.hardswish(self.last_conv(self.avg_pool(x))))
        return x
