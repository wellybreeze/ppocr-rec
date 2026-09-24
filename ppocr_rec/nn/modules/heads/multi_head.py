from __future__ import annotations

import torch
import torch.nn as nn

from ppocr_rec.nn.modules.heads.ctc_head import CTCHead
from ppocr_rec.nn.modules.heads.nrtr_head import Transformer
from ppocr_rec.nn.modules.necks.rnn import Im2Seq, SequenceEncoder


class FCTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, only_transpose=False):
        super().__init__()
        self.only_transpose = only_transpose
        if not only_transpose:
            self.fc = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return x if self.only_transpose else self.fc(x)


class MultiHead(nn.Module):
    def __init__(self, in_channels, out_channels_list, head_list=None, **kwargs):
        super().__init__()
        from copy import deepcopy

        self.needs_targets = True
        self.head_list = deepcopy(head_list)
        self.gtc_kind = "sar"
        assert self.head_list and len(self.head_list) >= 2
        for idx, head_name in enumerate(self.head_list):
            name = list(head_name)[0]
            if name == "SARHead":
                from ppocr_rec.nn.modules.heads.sar_head import SARHead

                sar_args = dict(self.head_list[idx][name] or {})
                self.sar_head = SARHead(
                    in_channels=in_channels,
                    out_channels=out_channels_list["SARLabelDecode"],
                    **sar_args,
                )
                self.gtc_kind = "sar"
            elif name == "NRTRHead":
                gtc_args = dict(self.head_list[idx][name] or {})
                max_text_length = gtc_args.get("max_text_length", 25)
                nrtr_dim = gtc_args.get("nrtr_dim", 256)
                num_decoder_layers = gtc_args.get("num_decoder_layers", 4)
                self.before_gtc = nn.Sequential(nn.Flatten(2), FCTranspose(in_channels, nrtr_dim))
                # Name matches Paddle `head.gtc_head.*` for weight conversion.
                self.gtc_head = Transformer(
                    d_model=nrtr_dim,
                    nhead=max(nrtr_dim // 32, 1),
                    num_encoder_layers=-1,
                    num_decoder_layers=num_decoder_layers,
                    max_len=max_text_length,
                    dim_feedforward=nrtr_dim * 4,
                    out_channels=out_channels_list["NRTRLabelDecode"],
                )
                self.gtc_kind = "nrtr"
            elif name == "CTCHead":
                neck_args = dict(self.head_list[idx][name].get("Neck") or {})
                encoder_type = neck_args.pop("name")
                self.ctc_encoder = SequenceEncoder(in_channels=in_channels, encoder_type=encoder_type, **neck_args)
                head_args = dict(self.head_list[idx][name].get("Head") or {})
                self.ctc_head = CTCHead(
                    in_channels=self.ctc_encoder.out_channels,
                    out_channels=out_channels_list["CTCLabelDecode"],
                    **head_args,
                )
            else:
                raise NotImplementedError(name)

    def forward(self, x, targets=None):
        ctc_encoder = self.ctc_encoder(x)
        ctc_out = self.ctc_head(ctc_encoder, targets)
        if not self.training:
            return ctc_out
        head_out = {"ctc": ctc_out, "ctc_neck": ctc_encoder}
        if self.gtc_kind == "sar":
            head_out["sar"] = self.sar_head(x, targets[1:] if targets is not None else None)
        else:
            gtc_in = self.before_gtc(x)
            gtc_targets = [targets[1], targets[2]] if targets is not None else None
            head_out["gtc"] = self.gtc_head(gtc_in, gtc_targets)
        return head_out
