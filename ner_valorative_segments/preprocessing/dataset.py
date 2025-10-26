from torch.utils.data import Dataset
from collections import defaultdict
from sklearn.model_selection import train_test_split
import json


class NERDataset(Dataset):
    def __init__(self, file_path, config):
        self.config = config
        self.data, self.word_vocab, self.label_vocab = self._load_data(file_path)
        
    def _load_data(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        data = self.preprocess_data(json_data)

        return data
        
    def _build_vocabs(self):
        word_vocab = defaultdict(lambda: len(word_vocab))
        label_vocab = defaultdict(lambda: len(label_vocab))
        
        label_vocab["<PAD>"] = 0 
        word_vocab["<PAD>"] = 0
        word_vocab["<UNK>"] = 1
  
        # Add your vocabulary building logic
        return word_vocab, label_vocab
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.word_vocab, self.label_vocab
    
    # Preprocess function
    def preprocess_data(self, json_data):
        word_vocab, label_vocab = self._build_vocabs()
        data = []
        for example in json_data:
            text = example["text"]
            entities = example["entities"]
            
            # Tokenize the text
            tokens = text.split()
            labels = ["O"] * len(tokens)  # Initialize all tokens as "O"
            
            # Assign labels based on entities
            for entity in entities:
                entity_text = entity["fragment"].split()
                label = entity["label"]
                start, end = entity["start"], entity["end"]
                
                # Find the token indices for the entity
                entity_tokens = entity_text
                for i in range(len(tokens)):
                    if tokens[i:i+len(entity_tokens)] == entity_tokens:
                        labels[i] = f"B-{label}"
                        for j in range(1, len(entity_tokens)):
                            labels[i + j] = f"I-{label}"
                        break
            
            # Convert tokens and labels to indices
            token_ids = [word_vocab[token] for token in tokens]
            label_ids = [label_vocab[label] for label in labels]
            # Add to data
            data.append((token_ids, label_ids))
        return data, word_vocab, label_vocab
    

    def train_test_split(self):
        train_data, test_data = train_test_split(self.data, test_size=0.2, random_state=42)
        return train_data, test_data