import torch.nn as nn
from abc import ABC, abstractmethod

class BaseNERModel(nn.Module, ABC):
    def __init__(self, config,word_vocab, pad_idx):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            len(word_vocab), 
            config.embed_dim, 
            padding_idx=pad_idx)

        self.dropout = nn.Dropout(config.dropout)
        
    @abstractmethod
    def forward(self, x, labels=None):
        pass

    @classmethod
    def from_config(cls, config):
        return cls(config)