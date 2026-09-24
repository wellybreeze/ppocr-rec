from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCHead(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        fc_decay=0.0004,
        mid_channels=None,
        return_feats=False,
        use_guide=False,
        **kwargs,
    ):
        super().__init__()
        self.use_guide = use_guide
        self.mid_channels = mid_channels
        self.return_feats = return_feats
        self.out_channels = out_channels
        if use_guide:
            self.guide_layer = nn.Sequential(
                nn.Conv1d(in_channels, in_channels, 5, padding=2, groups=in_channels, bias=False),
                nn.BatchNorm1d(in_channels),
                nn.Hardswish(),
                nn.Conv1d(in_channels, in_channels, 1, bias=False),
                nn.BatchNorm1d(in_channels),
                nn.Hardswish(),
            )
        if mid_channels is None:
            self.fc = nn.Linear(in_channels, out_channels)
        else:
            self.fc1 = nn.Linear(in_channels, mid_channels)
            self.fc2 = nn.Linear(mid_channels, out_channels)

    def forward(self, x, targets=None):
        if self.use_guide:
            x = self.guide_layer(x.transpose(1, 2)).transpose(1, 2)
        if self.mid_channels is None:
            predicts = self.fc(x)
        else:
            x = self.fc1(x)
            predicts = self.fc2(x)
        result = (x, predicts) if self.return_feats else predicts
        if not self.training:
            predicts = F.softmax(predicts, dim=2)
            result = predicts
        return result
