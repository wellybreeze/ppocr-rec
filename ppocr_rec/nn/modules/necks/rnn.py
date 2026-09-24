from __future__ import annotations

import torch
import torch.nn as nn


class Im2Seq(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x):
        b, c, h, w = x.shape
        if h != 1:
            x = nn.functional.adaptive_avg_pool2d(x, (1, w))
        x = x.squeeze(2).permute(0, 2, 1)  # B, W, C
        return x


class EncoderWithRNN(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super().__init__()
        self.out_channels = hidden_size * 2
        self.lstm = nn.LSTM(in_channels, hidden_size, num_layers=2, bidirectional=True, batch_first=True)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x


class EncoderWithFC(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super().__init__()
        self.out_channels = hidden_size
        self.fc = nn.Linear(in_channels, hidden_size)

    def forward(self, x):
        return self.fc(x)


class SequenceEncoder(nn.Module):
    def __init__(self, in_channels, encoder_type, hidden_size=48, **kwargs):
        super().__init__()
        self.encoder_reshape = Im2Seq(in_channels)
        self.encoder_type = encoder_type
        self.out_channels = in_channels
        if encoder_type == "reshape":
            self.only_reshape = True
            self.encoder = None
        elif encoder_type == "rnn":
            self.only_reshape = False
            self.encoder = EncoderWithRNN(in_channels, hidden_size)
            self.out_channels = self.encoder.out_channels
        elif encoder_type == "fc":
            self.only_reshape = False
            self.encoder = EncoderWithFC(in_channels, hidden_size)
            self.out_channels = self.encoder.out_channels
        elif encoder_type == "svtr":
            from ppocr_rec.nn.modules.necks.svtr_encoder import EncoderWithSVTR

            self.only_reshape = False
            self.encoder = EncoderWithSVTR(in_channels, **kwargs)
            self.out_channels = self.encoder.out_channels
        elif encoder_type == "lightsvtr":
            from ppocr_rec.nn.modules.necks.svtr_encoder import EncoderWithLightSVTR

            self.only_reshape = False
            self.encoder = EncoderWithLightSVTR(in_channels, **kwargs)
            self.out_channels = self.encoder.out_channels
        else:
            raise NotImplementedError(encoder_type)

    def forward(self, x):
        if self.encoder_type not in ("svtr", "lightsvtr"):
            x = self.encoder_reshape(x)
            if self.encoder is not None:
                x = self.encoder(x)
            return x
        x = self.encoder(x)
        x = self.encoder_reshape(x)
        return x
