from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAREncoder(nn.Module):
    def __init__(self, enc_bi_rnn=False, enc_drop_rnn=0.1, enc_gru=False, d_model=512, d_enc=512, mask=True, **kwargs):
        super().__init__()
        self.mask = mask
        rnn_cls = nn.GRU if enc_gru else nn.LSTM
        self.rnn_encoder = rnn_cls(
            d_model, d_enc, num_layers=2, batch_first=True, dropout=enc_drop_rnn, bidirectional=enc_bi_rnn
        )
        encoder_rnn_out_size = d_enc * (int(enc_bi_rnn) + 1)
        self.linear = nn.Linear(encoder_rnn_out_size, encoder_rnn_out_size)

    def forward(self, feat, img_metas=None):
        h_feat = feat.shape[2]
        feat_v = F.max_pool2d(feat, kernel_size=(h_feat, 1), stride=1).squeeze(2).permute(0, 2, 1)
        holistic_feat = self.rnn_encoder(feat_v)[0]
        valid_hf = holistic_feat[:, -1, :]
        return self.linear(valid_hf)


class ParallelSARDecoder(nn.Module):
    def __init__(
        self,
        out_channels,
        enc_bi_rnn=False,
        dec_bi_rnn=False,
        dec_drop_rnn=0.0,
        dec_gru=False,
        d_model=512,
        d_enc=512,
        d_k=512,
        pred_dropout=0.1,
        max_text_length=30,
        pred_concat=True,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = out_channels
        self.start_idx = out_channels - 2
        self.padding_idx = out_channels - 1
        self.max_seq_len = max_text_length
        self.pred_concat = pred_concat
        encoder_rnn_out_size = d_enc * (int(enc_bi_rnn) + 1)
        decoder_rnn_out_size = encoder_rnn_out_size * (int(dec_bi_rnn) + 1)
        self.conv1x1_1 = nn.Linear(decoder_rnn_out_size, d_k)
        self.conv3x3_1 = nn.Conv2d(d_model, d_k, 3, 1, 1)
        self.conv1x1_2 = nn.Linear(d_k, 1)
        rnn_cls = nn.GRU if dec_gru else nn.LSTM
        self.rnn_decoder = rnn_cls(
            encoder_rnn_out_size,
            encoder_rnn_out_size,
            num_layers=2,
            batch_first=True,
            dropout=dec_drop_rnn,
            bidirectional=dec_bi_rnn,
        )
        self.embedding = nn.Embedding(self.num_classes, encoder_rnn_out_size, padding_idx=self.padding_idx)
        self.pred_dropout = nn.Dropout(pred_dropout)
        pred_num_classes = self.num_classes - 1
        fc_in = decoder_rnn_out_size + d_model + encoder_rnn_out_size if pred_concat else d_model
        self.prediction = nn.Linear(fc_in, pred_num_classes)

    def _2d_attention(self, decoder_input, feat, holistic_feat, train_mode=True):
        y = self.rnn_decoder(decoder_input)[0]
        attn_query = self.conv1x1_1(y).unsqueeze(-1).unsqueeze(-1)
        attn_key = self.conv3x3_1(feat).unsqueeze(1)
        attn_weight = torch.tanh(attn_key + attn_query).permute(0, 1, 3, 4, 2)
        attn_weight = self.conv1x1_2(attn_weight)
        bsz, t, h, w, _ = attn_weight.shape
        attn_weight = attn_weight.reshape(bsz, t, -1).softmax(-1).reshape(bsz, t, 1, h, w)
        attn_feat = (feat.unsqueeze(1) * attn_weight).sum(dim=(3, 4))
        if self.pred_concat:
            hf = holistic_feat.expand(bsz, y.size(1), -1)
            y = self.prediction(torch.cat([y, attn_feat, hf], dim=2))
        else:
            y = self.prediction(attn_feat)
        if train_mode:
            y = self.pred_dropout(y)
        return y

    def forward(self, feat, out_enc, label=None, train_mode=True):
        if train_mode:
            lab_embedding = self.embedding(label.long())
            out_enc = out_enc.unsqueeze(1)
            in_dec = torch.cat([out_enc, lab_embedding], dim=1)
            out_dec = self._2d_attention(in_dec, feat, out_enc, True)
            return out_dec[:, 1:, :]
        bsz = feat.size(0)
        start = self.embedding(torch.full((bsz,), self.start_idx, dtype=torch.long, device=feat.device)).unsqueeze(1)
        start = start.expand(bsz, self.max_seq_len, -1)
        out_enc = out_enc.unsqueeze(1)
        decoder_input = torch.cat([out_enc, start], dim=1)
        outputs = []
        for i in range(1, self.max_seq_len + 1):
            decoder_output = self._2d_attention(decoder_input, feat, out_enc, False)
            char_output = decoder_output[:, i, :].softmax(-1)
            outputs.append(char_output)
            max_idx = char_output.argmax(1)
            if i < self.max_seq_len:
                decoder_input = decoder_input.clone()
                decoder_input[:, i + 1, :] = self.embedding(max_idx)
        return torch.stack(outputs, 1)


class SARHead(nn.Module):
    def __init__(self, in_channels, out_channels, enc_dim=512, max_text_length=30, **kwargs):
        super().__init__()
        self.encoder = SAREncoder(d_model=in_channels, d_enc=enc_dim, **{k: kwargs[k] for k in kwargs if k.startswith("enc_")})
        dec_keys = ("enc_bi_rnn", "dec_bi_rnn", "dec_drop_rnn", "dec_gru", "d_k", "pred_dropout", "pred_concat")
        dec_args = {k: kwargs[k] for k in dec_keys if k in kwargs}
        self.decoder = ParallelSARDecoder(
            out_channels=out_channels, d_model=in_channels, d_enc=enc_dim, max_text_length=max_text_length, **dec_args
        )

    def forward(self, feat, targets=None):
        holistic_feat = self.encoder(feat, targets)
        if self.training:
            label = targets[0]
            return self.decoder(feat, holistic_feat, label, train_mode=True)
        return self.decoder(feat, holistic_feat, None, train_mode=False)
