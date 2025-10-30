"""
Dataset class for valorative segment tagging.
"""

import json
from collections import defaultdict
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class NERDataset(Dataset):
    """PyTorch dataset for token-level valorative tagging (Appraisal Theory)."""

    def __init__(self, file_path, config):
        """Load and preprocess the dataset from a JSON file."""
        self.config = config
        self.data, self.word_vocab, self.label_vocab = self._load_data(file_path)

    def _load_data(self, file_path):
        """Load dataset and return token/label pairs with vocabularies."""
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        return self.preprocess_data(json_data)

    def _build_vocabs(self):
        """Initialize word and label vocabularies."""
        word_vocab = {"<PAD>": 0, "<UNK>": 1}
        label_vocab = {"<PAD>": 0}
        word_vocab = defaultdict(lambda: len(word_vocab), word_vocab)
        label_vocab = defaultdict(lambda: len(label_vocab), label_vocab)
        return word_vocab, label_vocab

    def preprocess_data(self, json_data):
        """Convert annotated text into indexed token and label sequences."""
        word_vocab, label_vocab = self._build_vocabs()
        data = []

        for example in json_data:
            tokens = example["text"].split()
            labels = ["O"] * len(tokens)

            for entity in example["entities"]:
                entity_tokens = entity["fragment"].split()
                label = entity["label"]

                for i in range(len(tokens)):
                    if tokens[i:i + len(entity_tokens)] == entity_tokens:
                        labels[i] = f"B-{label}"
                        for j in range(1, len(entity_tokens)):
                            labels[i + j] = f"I-{label}"
                        break

            token_ids = [word_vocab[token] for token in tokens]
            label_ids = [label_vocab[label] for label in labels]
            data.append((token_ids, label_ids))

        return data, word_vocab, label_vocab

    def __len__(self):
        """Return number of samples."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return token and label IDs for one sample."""
        return self.data[idx]

    def train_test_split(self, test_size=0.2, random_state=42):
        """Split data into training and test subsets."""
        return train_test_split(self.data, test_size=test_size, random_state=random_state)
