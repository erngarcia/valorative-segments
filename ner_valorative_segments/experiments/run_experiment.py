import torch
import yaml
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models import BiLSTM, BiLSTM_CRF, BiLSTM_Attention
from preprocessing.dataset import NERDataset
from experiments.train import NERTrainer
from experiments.configs.configs import Config



def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def run_experiment(params, experiment_number):
    # Load the YAML config
    config_path = os.path.join(os.path.dirname(__file__), "configs", "base.yaml")
    config_dict = load_config(config_path)

    if params:
        config_dict.update(params)

    config = Config(config_dict)

    # Load the dataset
    dataset = NERDataset(config.data_path, config)
    vocab_size = len(dataset.word_vocab)
    num_classes = len(dataset.label_vocab)

    # Set required parameters
    config.vocab_size = vocab_size
    config.word_pad_idx = dataset.word_vocab["<PAD>"]
    config.label_pad_idx = dataset.label_vocab["<PAD>"]

    # Split dataset
    train_data, test_data = dataset.train_test_split()

    # Define models
    models = {
        "BiLSTM": BiLSTM(config, num_classes, dataset.word_vocab, config.word_pad_idx, config.label_pad_idx),
        "BiLSTM_CRF": BiLSTM_CRF(config, num_classes, dataset.word_vocab, config.word_pad_idx, config.label_pad_idx),
        "BiLSTM_Attention": BiLSTM_Attention(config, num_classes, dataset.word_vocab, config.word_pad_idx, config.label_pad_idx),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    best_f1 = 0
    best_model_name = None
    best_trainer = None
    best_errors = None

    for name, model in models.items():
        print(model.to(device))
        print(f'training {name} model')
        # Training
        trainer = NERTrainer(
            model=model,
            train_data=train_data,
            val_data=test_data,
            config=config,
            word_vocab=dataset.word_vocab,
            label_vocab=dataset.label_vocab,
            device=device
        )
        results = trainer.run()

        if results['best_f1'] > best_f1:
            best_f1 = results['best_f1']
            best_model_name = name
            best_matrix = results['confusion_matrix']
            best_metrics = results['best_metrics']
            best_errors = results['errors']
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
        dataset.label_vocab
    )

