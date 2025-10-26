import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix
from datetime import datetime
import os
import pandas as pd
import numpy as np
from models import BiLSTM, BiLSTM_CRF, BiLSTM_Attention
from collections import Counter


class NERTrainer:
    def __init__(self, model, train_data, val_data, config, device, word_vocab, label_vocab):
        device = device
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
        self.device = device
        self.epochs = config.max_epochs
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.train_loader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=self.collate_fn)
        self.val_loader = DataLoader(val_data, batch_size=8, shuffle=False, collate_fn=self.collate_fn)
        self.num_classes = len(label_vocab)
        self.experiment_id = datetime.now().strftime("%Y%m%d-%H%M%S")

        class_weights = self.compute_class_weights(label_vocab)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), ignore_index=label_vocab["O"])

    def compute_class_weights(self, label_vocab):
        #biasing the model towards meaningful labels 
        label_counts = {
            "O": 21408,
            "B-no_attitude": 55,
            "B-judgement": 387,
            "I-judgement": 3814,
            "B-appreciation": 108,
            "I-appreciation": 886,
            "B-affect": 54,
            "I-affect": 248
        }

        total = sum(label_counts.values())
        weights = []
        for label, idx in sorted(label_vocab.items(), key=lambda x: x[1]):
            count = label_counts.get(label, 1)
            weight = total / (count + 1e-6)
            weights.append(weight)

        weights = torch.tensor(weights, dtype=torch.float)
        weights = weights / weights.max()
        return weights

    def collate_fn(self, batch):
        tokens, labels = zip(*batch)
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
    
    def reverse_lookup_token(self, idx):
        for token, token_idx in self.word_vocab.items():
            if token_idx == idx:
                return token
        return "<UNK>"

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

    def validate(self, return_preds=False):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_tokens = []
        all_token_strs = []


        with torch.no_grad():
            for tokens, labels in self.val_loader:
                tokens, labels = tokens.to(self.device), labels.to(self.device)
                all_tokens.extend(tokens.cpu().numpy())

                if isinstance(self.model, BiLSTM_CRF):
                    preds = self.model(tokens)
                    loss = self.model(tokens, labels)
                    preds = [torch.tensor(p, device=self.device) for p in preds]
                    preds = torch.nn.utils.rnn.pad_sequence(preds, batch_first=True, padding_value=self.label_vocab["O"])
                else:
                    output = self.model(tokens)
                    if isinstance(output, tuple):
                        logits = output[0]
                    else:
                        logits = output
                    preds = torch.argmax(logits, dim=-1)
                    loss = self.criterion(logits.view(-1, self.num_classes), labels.view(-1))

                total_loss += loss.item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                token_ids = tokens.cpu().numpy()
                for seq in token_ids:
                    token_str_seq = [
                        self.reverse_lookup_token(t) 
                        for t in seq 
                        if t != self.word_vocab["<PAD>"]
                    ]
                    all_token_strs.append(token_str_seq)

        precision, recall, f1, accuracy, cm = self.compute_metrics(all_preds, all_labels, self.label_vocab)

        print(f"Validation - Loss: {total_loss / len(self.val_loader):.4f}, "
              f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
        result = {
        "loss": total_loss / len(self.val_loader),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "confusion_matrix": cm
        }

        if return_preds:
            result.update({
                "predictions": all_preds,
                "labels": all_labels,
                "token_strs": all_token_strs
            })
        return result

    def compute_metrics(self, preds, labels, label_vocab):
        true_labels = []
        pred_labels = []
        pad_idx = label_vocab.get("<PAD>")

        for pred_seq, label_seq in zip(preds, labels):
            for pred, label in zip(pred_seq, label_seq):
                if label != pad_idx:
                    true_labels.append(label)
                    pred_labels.append(pred)

        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)

        # Build idx-to-label map and filter out PAD
        idx_to_label = {idx: label for label, idx in label_vocab.items()}
        valid_indices = [i for i in sorted(idx_to_label) if i != pad_idx]
        class_names = [idx_to_label[i] for i in valid_indices]

        # Print classification report for valid labels only
        report = classification_report(
            true_labels,
            pred_labels,
            labels=valid_indices,
            target_names=class_names,
            zero_division=0
        )
        print(report)

        precision = precision_score(true_labels, pred_labels, labels=valid_indices, average="macro", zero_division=0)
        recall = recall_score(true_labels, pred_labels, labels=valid_indices, average="macro", zero_division=0)
        f1 = f1_score(true_labels, pred_labels, labels=valid_indices, average="macro", zero_division=0)
        accuracy = accuracy_score(true_labels, pred_labels)
        cm = confusion_matrix(true_labels, pred_labels, labels=valid_indices)

        return precision, recall, f1, accuracy, cm


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
        patience = 5
        epochs_no_improve = 0
        best_metrics = None
        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            val_metrics = self.validate()
            self.log_experiment(val_metrics)
            

            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {train_loss}")
            print(f"Validation - Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, "
                  f"F1 Score: {val_metrics['f1']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")

            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                best_metrics = val_metrics
                detailed = self.validate(return_preds=True)
                token_strs = detailed["token_strs"]  # original token strings
                best_predictions = {
                        "tokens": detailed["token_strs"],
                        "predictions": detailed["predictions"],
                        "labels": detailed["labels"],
                        "confusion_matrix": detailed["confusion_matrix"]
                    }
                # Map indices to labels
                idx_to_label = {idx: label for label, idx in self.label_vocab.items()}
                idx_to_word = {idx: word for word, idx in self.word_vocab.items()}

                pred_str = [[idx_to_label.get(p, "O") for p in seq] for seq in best_predictions["predictions"]]
                label_str = [[idx_to_label.get(t, "O") for t in seq] for seq in best_predictions["labels"]]
                token_str = [[idx_to_word.get(t, "<UNK>") for t in seq] for seq in best_predictions["tokens"]]

                best_predictions["predictions_str"] = pred_str
                best_predictions["labels_str"] = label_str
                best_predictions["tokens_str"] = token_str

                errors = self.log_errors(
                    tokens=token_strs,  # this is now full string tokens
                    predictions=pred_str,
                    labels=label_str,
                    n_fp=20,
                    n_fn=20
                )

                best_predictions["errors"] = errors
                epochs_no_improve = 0
                                
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        return {
                "best_f1": best_f1,
                "best_metrics": best_metrics,
                "confusion_matrix": best_predictions["confusion_matrix"],
                "errors": best_predictions["errors"]
            }

    def log_errors(self, tokens, predictions, labels, n_fp=100, n_fn=100):
        fp, fn = [], []

        for tok_seq, pred_seq, true_seq in zip(tokens, predictions, labels):
            for i, (tok, p, t) in enumerate(zip(tok_seq, pred_seq, true_seq)):
                if p != t:
                    context = " ".join(tok_seq)  # full sentence context
                    if p != 'O' and t == 'O':
                        fp.append((tok, p, t, context))  # False Positive
                    elif p == 'O' and t != 'O':
                        fn.append((tok, p, t, context))  # False Negative
                    else:
                        # Other kinds of mismatches (label vs label)
                        fp.append((tok, p, t, context))  # or log in a separate list if needed


        return {
            "false_positives": fp[:n_fp],
            "false_negatives": fn[:n_fn]
        }