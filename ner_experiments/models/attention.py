import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseNERModel

class BiLSTM_Attention(BaseNERModel):
    def __init__(self, config, num_classes, word_vocab, pad_idx):
        super().__init__(config, word_vocab, pad_idx)

        self.num_classes = num_classes
        self.pad_idx = pad_idx
        
        # BiLSTM Layer
        self.lstm = nn.LSTM(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            bidirectional=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention Mechanism
        self.attention = nn.Linear(2 * config.hidden_dim, 1, bias=False)
        
        # Output Layer
        self.fc = nn.Linear(2 * config.hidden_dim, num_classes)
        
        # LayerNorm for stability
        self.layer_norm = nn.LayerNorm(2 * config.hidden_dim)

    def forward(self, x, labels=None):
        # Embedding
        x = self.embedding(x)
        x = self.dropout(x)

        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, 2*hidden_dim)
        lstm_out = self.layer_norm(lstm_out)

        # Compute attention scores
        attn_scores = self.attention(lstm_out).squeeze(-1)  # (batch, seq_len)

        mask = (x != self.pad_idx).float()
        mask = mask[:, :, 0]

        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Normalize attention weights
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)

        # Apply attention
        attended_lstm_out = attn_weights * lstm_out  # (batch, seq_len, 2*hidden_dim)

        # Token-wise Classification
        logits = self.fc(attended_lstm_out)  # (batch, seq_len, num_classes)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
            loss = loss_fn(logits.view(-1, self.num_classes), labels.view(-1))
            return loss

        return logits, attn_weights.squeeze()