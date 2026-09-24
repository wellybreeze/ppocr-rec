from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, dropout, dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)
        self.dim = dim

    def forward(self, x):
        x = x * math.sqrt(self.dim)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, attn_drop, resid_drop, with_cross_attn=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=attn_drop, batch_first=True)
        self.with_cross_attn = with_cross_attn
        if with_cross_attn:
            self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=attn_drop, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(resid_drop)

    def forward(self, tgt, memory=None, self_mask=None):
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=self_mask)
        tgt = self.norm1(tgt + self.dropout(tgt2))
        if self.with_cross_attn and memory is not None:
            tgt2, _ = self.cross_attn(tgt, memory, memory)
            tgt = self.norm2(tgt + self.dropout(tgt2))
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        return self.norm3(tgt + self.dropout(tgt2))


class Transformer(nn.Module):
    """NRTR-style decoder used as GTC head during training. Eval is unused (CTC only)."""

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=-1,
        beam_size=-1,
        num_decoder_layers=4,
        max_len=25,
        dim_feedforward=1024,
        out_channels=0,
        **kwargs,
    ):
        super().__init__()
        self.out_channels = out_channels + 1
        self.max_len = max_len
        self.embedding = nn.Embedding(self.out_channels, d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(0.1, d_model)
        self.encoder = None
        self.decoder = nn.ModuleList(
            [TransformerBlock(d_model, nhead, dim_feedforward, 0.0, 0.1, True) for _ in range(num_decoder_layers)]
        )
        self.tgt_word_prj = nn.Linear(d_model, self.out_channels, bias=False)

    @staticmethod
    def generate_square_subsequent_mask(sz, device):
        return torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)

    def forward_train(self, src, tgt):
        tgt = tgt[:, :-1]
        tgt = self.positional_encoding(self.embedding(tgt.long()))
        mask = self.generate_square_subsequent_mask(tgt.size(1), tgt.device)
        memory = src
        for layer in self.decoder:
            tgt = layer(tgt, memory, self_mask=mask)
        return self.tgt_word_prj(tgt)

    def forward(self, src, targets=None):
        if self.training:
            max_len = int(targets[1].max().item()) if targets[1] is not None else src.size(1)
            tgt = targets[0][:, : 2 + max_len]
            return self.forward_train(src, tgt)
        # greedy (rarely used; MultiHead eval returns CTC)
        bs = src.size(0)
        dec = torch.full((bs, 1), 2, dtype=torch.long, device=src.device)
        logits = []
        for _ in range(self.max_len):
            emb = self.positional_encoding(self.embedding(dec))
            mask = self.generate_square_subsequent_mask(emb.size(1), src.device)
            tgt = emb
            for layer in self.decoder:
                tgt = layer(tgt, src, self_mask=mask)
            logit = self.tgt_word_prj(tgt[:, -1])
            logits.append(logit)
            nxt = logit.argmax(-1, keepdim=True)
            dec = torch.cat([dec, nxt], 1)
            if (nxt == 3).all():
                break
        return torch.stack(logits, 1)
