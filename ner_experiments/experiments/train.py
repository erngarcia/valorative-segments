import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from datetime import datetime
import os
import pandas as pd
import numpy as np
from models import BiLSTM, BiLSTM_CRF, BiLSTM_Attention



class NERTrainer:
    def __init__(self, model, train_data, val_data, config, device, word_vocab, label_vocab):
        device=device
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=label_vocab["O"])
        self.device = device
        self.epochs = config.max_epochs
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.train_loader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=self.collate_fn)
        self.val_loader = DataLoader(val_data, batch_size=8, shuffle=False, collate_fn=self.collate_fn)
        self.num_classes = len(label_vocab)
        self.experiment_id = datetime.now().strftime("%Y%m%d-%H%M%S")


    def collate_fn(self, batch):
        tokens, labels = zip(*batch)  # Ensure batch is in expected format

        tokens = pad_sequence(
            [torch.tensor(seq) for seq in tokens], 
            batch_first=True, 
            padding_value=self.word_vocab["<PAD>"]
        )
        labels = pad_sequence(
            [torch.tensor(seq) for seq in labels], 
            batch_first=True, 
            padding_value=self.label_vocab["O"]
        )

        return tokens, labels

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for tokens, labels in self.train_loader:
            tokens, labels = tokens.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            
            loss = self.model(tokens, labels)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for tokens, labels in self.val_loader:
                tokens, labels = tokens.to(self.device), labels.to(self.device)

                if isinstance(self.model, BiLSTM_CRF):
                    preds = self.model(tokens)  # list of predicted sequences
                    loss = self.model(tokens, labels)

                    # Pad CRF output to match shape
                    preds = [torch.tensor(p, device=self.device) for p in preds]
                    preds = torch.nn.utils.rnn.pad_sequence(preds, batch_first=True, padding_value=self.label_vocab["O"])
                else:
                    output = self.model(tokens)  # Can be logits or (logits, attention)
                    if isinstance(output, tuple):
                        logits = output[0]  # e.g., from BiLSTM_Attention
                    else:
                        logits = output  # e.g., from BiLSTM

                    preds = torch.argmax(logits, dim=-1)
                    loss = self.criterion(logits.view(-1, self.num_classes), labels.view(-1))

                total_loss += loss.item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Compute metrics (excluding PAD)
        precision, recall, f1, accuracy = self.compute_metrics(all_preds, all_labels, self.label_vocab)

        print(f"Validation - Loss: {total_loss / len(self.val_loader):.4f}, "
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")

        return {
            "loss": total_loss / len(self.val_loader),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy
        }

   

    def compute_metrics(self, preds, labels, label_vocab):
        """
        Computes precision, recall, F1-score, and accuracy for NER predictions.

        Args:
            preds (list of numpy arrays): Model predictions.
            labels (list of numpy arrays): Ground truth labels.
            label_vocab (dict): Mapping from label names to indices.

        Returns:
            tuple: (precision, recall, f1, accuracy)
        """
        # Flatten predictions and labels, ignoring padding tokens
        true_labels = []
        pred_labels = []

        pad_idx = label_vocab.get("<PAD>", None)  # Ensure we handle padding
        for pred_seq, label_seq in zip(preds, labels):
            for pred, label in zip(pred_seq, label_seq):
                if pad_idx is None or label != pad_idx:  # Ignore padding tokens
                    true_labels.append(label)
                    pred_labels.append(pred)

        # Convert to numpy arrays
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)

        # Compute metrics
        precision = precision_score(true_labels, pred_labels, average="macro", zero_division=0)
        recall = recall_score(true_labels, pred_labels, average="macro", zero_division=0)
        f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
        accuracy = accuracy_score(true_labels, pred_labels)

        return precision, recall, f1, accuracy



    def log_experiment(self, metrics):
        log_path = "artifacts/experiments_log.csv"
        log_exists = os.path.exists(log_path)

        log_entry = {
            "experiment_id": self.experiment_id,
            "model": self.model.__class__.__name__,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1"],
            "accuracy": metrics["accuracy"],
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "batch_size": self.train_loader.batch_size,
            "epochs": self.epochs
        }

        df = pd.DataFrame([log_entry])
        if log_exists:
            df.to_csv(log_path, mode="a", header=False, index=False)
        else:
            df.to_csv(log_path, index=False)

    def run(self):
        best_f1 = 0
        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            val_metrics = self.validate()
            self.log_experiment(val_metrics)

            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {train_loss}")
            print(f"Validation - Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, "
                  f"F1 Score: {val_metrics['f1']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")

            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                # torch.save(self.model.state_dict(), f"artifacts/best_model_{self.experiment_id}.pt")
