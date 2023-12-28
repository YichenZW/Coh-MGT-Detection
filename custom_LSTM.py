import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import math

class CustomRNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        batch_first=True,
        max_seq_length=60,
        attention_maxscore=None,
    ):
        super(CustomRNN, self).__init__()
        self.bidirect = False
        self.num_layers = 1
        self.num_heads = 1
        self.batch_first = batch_first
        self.with_weight = False
        self.max_seq_length = max_seq_length
        self.attention_maxscore = attention_maxscore
        self.rnn = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=batch_first,
            bidirectional=self.bidirect,
            num_layers=self.num_layers,
        )
        self.pooling = nn.AdaptiveMaxPool2d((1, input_size))

    def forward(self, inputs, seq_lengths, sen_mask, method = "AttLSTM"):  
        # input.size = (batch_size, max_seq_length, node_num)
        # method can be "Pool", "LSTM", or 'AttLSTM"
        if method == "LSTM":
            packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(
                inputs,
                seq_lengths.to("cpu"),
                batch_first=self.batch_first,
                enforce_sorted=False,
            )
            res, (hn, cn) = self.rnn(input=packed_inputs)
            padded_res, _ = nn.utils.rnn.pad_packed_sequence(
                res, batch_first=self.batch_first, total_length=self.max_seq_length
            )  
            return hn.squeeze(0), padded_res
        elif method == "AttLSTM":
            sen_mask = torch.tensor(
                np.hstack([[[1]] * inputs.size()[0], sen_mask.cpu()])
            ).cuda()
            att_inputs, att_inputs_weight = attention(
                inputs,
                inputs,
                inputs,
                sen_mask,
                attention_maxscore=self.attention_maxscore,
            )
            packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(
                att_inputs,
                seq_lengths.to("cpu"),
                batch_first=self.batch_first,
                enforce_sorted=False,
            )
            res, (hn, cn) = self.rnn(input=packed_inputs)
            padded_res, _ = nn.utils.rnn.pad_packed_sequence(
                res, batch_first=self.batch_first, total_length=self.max_seq_length
            )  
            return hn.squeeze(0), padded_res
        else:
            out = self.pooling(inputs)
            return out.squeeze(1), None


def attention(query, key, value, mask=None, dropout=None, attention_maxscore=1000):
    """Compute scaled dot product attention"""
    d_k = query.size(-1)
    query = f.normalize(query, p=2, dim=-1)
    key = f.normalize(key, p=2, dim=-1)
    scores = (
        torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) * attention_maxscore
    )
    p_attn = None
    if mask is not None:
        for s, m in zip(scores, mask):
            s = s.masked_fill(m == 0, -1e9)
            p = s.softmax(dim=-1)
            if p_attn is None:
                p_attn = p
            else:
                p_attn = torch.cat([p_attn, p], dim=0)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
