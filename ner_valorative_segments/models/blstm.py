import torch.nn as nn
from .base_model import BaseNERModel

class BiLSTM(BaseNERModel):
    def __init__(self, config, num_classes, word_vocab, pad_idx):
        super().__init__(config,word_vocab,pad_idx)
        self.lstm = nn.LSTM(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            bidirectional=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True,
        )

        self.classifier = nn.Linear(2*config.hidden_dim, num_classes)
        self.num_classes = num_classes
        self.pad_idx = pad_idx

    def forward(self, x, labels=None):
        x = self.embedding(x)
        x = self.dropout(x)
        lstm_out, _ = self.lstm(x)
        logits = self.classifier(lstm_out)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
            loss = loss_fn(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        return logits