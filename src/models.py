"""
Model architectures for Fake News Detection.

Baselines:
  1. BiLSTMBaseline  — BiLSTM + Sequential Attention
  2. BiGRUBaseline   — BiGRU  + Sequential Attention

Proposed:
  3. HybridCNNBiGRUAttention — CNN + BiGRU + Sequential Attention
     (supports ablation via use_cnn / use_attention flags)

All models expect input shape (batch, seq_len, embedding_dim) where
seq_len = number of sentence embeddings per article.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequentialAttention(nn.Module):
    """Learnable attention over RNN time-steps."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, rnn_out: torch.Tensor):
        # rnn_out: (batch, seq_len, hidden_dim)
        scores = self.v(torch.tanh(self.W(rnn_out)))  # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)         # (batch, seq_len, 1)
        context = (weights * rnn_out).sum(dim=1)        # (batch, hidden_dim)
        return context, weights.squeeze(-1)              # (batch,), (batch, seq_len)


# ------------------------------------------------------------------
# Baseline 1 — BiLSTM + Sequential Attention
# ------------------------------------------------------------------
class BiLSTMBaseline(nn.Module):
    def __init__(self, embedding_dim=384, hidden_dim=128, num_layers=2,
                 num_classes=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = SequentialAttention(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        ctx, attn_w = self.attention(out)
        return self.head(self.dropout(ctx)), attn_w


# ------------------------------------------------------------------
# Baseline 2 — BiGRU + Sequential Attention
# ------------------------------------------------------------------
class BiGRUBaseline(nn.Module):
    def __init__(self, embedding_dim=384, hidden_dim=128, num_layers=2,
                 num_classes=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            embedding_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = SequentialAttention(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        ctx, attn_w = self.attention(out)
        return self.head(self.dropout(ctx)), attn_w


# ------------------------------------------------------------------
# Proposed — LLM + CNN + BiGRU + Sequential Attention
# ------------------------------------------------------------------
class HybridCNNBiGRUAttention(nn.Module):
    """
    Proposed hybrid model.
    Set use_cnn=False or use_attention=False for ablation studies.
    """

    def __init__(self, embedding_dim=384, hidden_dim=128, num_filters=64,
                 kernel_sizes=(3, 5, 7), num_layers=2, num_classes=2,
                 dropout=0.3, use_cnn=True, use_attention=True):
        super().__init__()
        self.use_cnn = use_cnn
        self.use_attention = use_attention

        if use_cnn:
            self.convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(embedding_dim, num_filters, k, padding=(k - 1) // 2),
                    nn.BatchNorm1d(num_filters),
                    nn.ReLU(),
                )
                for k in kernel_sizes
            ])
            gru_input = num_filters * len(kernel_sizes)
        else:
            gru_input = embedding_dim

        self.gru = nn.GRU(
            gru_input, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        if use_attention:
            self.attention = SequentialAttention(hidden_dim * 2)

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        if self.use_cnn:
            xt = x.permute(0, 2, 1)                           # (B, D, S)
            conv_outs = [conv(xt) for conv in self.convs]      # each (B, F, S)
            x = torch.cat(conv_outs, dim=1).permute(0, 2, 1)  # (B, S, F*3)

        gru_out, _ = self.gru(x)                               # (B, S, H*2)

        if self.use_attention:
            ctx, attn_w = self.attention(gru_out)
        else:
            ctx = gru_out.mean(dim=1)
            attn_w = torch.ones(gru_out.size(0), gru_out.size(1),
                                device=gru_out.device) / gru_out.size(1)

        return self.head(self.dropout(ctx)), attn_w
