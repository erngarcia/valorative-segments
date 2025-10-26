import torch
import torch.nn as nn
from torchcrf import CRF
from .base_model import BaseNERModel

class BiLSTM_CRF(BaseNERModel):
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
        
        # Fully connected layer
        self.fc = nn.Linear(2 * config.hidden_dim, num_classes)
        
        # CRF Layer
        self.crf = CRF(num_classes, batch_first=True)
    
    def forward(self, x, labels=None):
        # Embedding + BiLSTM
        x_embed = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        lstm_out, _ = self.lstm(x_embed)  # (batch_size, seq_len, hidden_dim*2)
        logits = self.fc(lstm_out)  # (batch_size, seq_len, num_classes)
        
        # CRF Mode
        mask = (x != self.pad_idx).bool()
        
        if labels is not None:
            loss = -self.crf(logits, labels, mask=mask, reduction="mean")
            return loss
        else:
            return self.crf.decode(logits) 
