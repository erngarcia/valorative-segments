"""
Run a single valorative segment tagging experiment.

Loads configuration, builds the dataset, initializes models (BiLSTM, BiLSTM+CRF,
BiLSTM+Attention), and trains them using NERTrainer. Returns the best-performing
model and its metrics.
"""

import os
import sys
import yaml
import torch

# Ensure project root is on path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models import BiLSTM, BiLSTM_CRF, BiLSTM_Attention
from preprocessing.dataset import NERDataset
from experiments.train import NERTrainer
from experiments.configs.configs import Config


def load_config(config_path):
    """Load a YAML configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def run_experiment(params, experiment_number):
    """Train and evaluate multiple NER architectures on the given dataset.

    Args:
        params (dict): Dictionary of hyperparameters (e.g., lr, batch_size).
        experiment_number (int): Index of the experiment in the sweep.

    Returns:
        tuple: (
            best_f1, best_metrics, best_confusion_matrix, best_model_name,
            best_errors, best_model, dataset, word_vocab, label_vocab
        )
    """
    # Load base config and merge overrides
    config_path = os.path.join(os.path.dirname(__file__), "configs", "base.yaml")
    config_dict = load_config(config_path)
    if params:
        config_dict.update(params)
    config = Config(config_dict)

    # Load dataset and vocabularies
    dataset = NERDataset(config.data_path, config)
    vocab_size = len(dataset.word_vocab)
    num_classes = len(dataset.label_vocab)

    # Set dynamic parameters
    config.vocab_size = vocab_size
    config.word_pad_idx = dataset.word_vocab["<PAD>"]
    config.label_pad_idx = dataset.label_vocab["<PAD>"]

    # Split dataset
    train_data, test_data = dataset.train_test_split()

    # Define models
    models = {
        "BiLSTM": BiLSTM(config, num_classes, dataset.word_vocab, config.word_pad_idx),
        "BiLSTM_CRF": BiLSTM_CRF(config, num_classes, dataset.word_vocab, config.word_pad_idx),
        "BiLSTM_Attention": BiLSTM_Attention(config, num_classes, dataset.word_vocab, config.word_pad_idx),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_f1, best_model_name, best_trainer, best_errors = 0, None, None, None

    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n🧠 Training {name} model...\n")
        model.to(device)

        trainer = NERTrainer(
            model=model,
            train_data=train_data,
            val_data=test_data,
            config=config,
            word_vocab=dataset.word_vocab,
            label_vocab=dataset.label_vocab,
            device=device,
        )
        results = trainer.run()

        if results["best_f1"] > best_f1:
            best_f1 = results["best_f1"]
            best_model_name = name
            best_matrix = results["confusion_matrix"]
            best_metrics = results["best_metrics"]
            best_errors = results["errors"]
            best_trainer = trainer

    best_model = best_trainer.model if best_trainer else None

    return (
        best_f1,
        best_metrics,
        best_matrix,
        best_model_name,
        best_errors,
        best_model,
        dataset,
        dataset.word_vocab,
        dataset.label_vocab,
    )
