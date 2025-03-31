import torch
import yaml
import os
from models import BiLSTM, BiLSTM_CRF, BiLSTM_Attention
from preprocessing.dataset import NERDataset
from experiments.train import NERTrainer
from configs.configs import Config

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def run_experiment(params, experiment_number):

    
    # Load the YAML config
    config_path = os.path.join(os.path.dirname(__file__), "configs", "base.yaml")
    config_dict = load_config(config_path)

    config = Config(config_dict)  # Convert dictionary to object
    config_dict.update(params)

    # Load the dataset
    dataset = NERDataset(config.data_path, config)
    vocab_size = len(dataset.word_vocab)
    num_classes = len(dataset.label_vocab)

    # Set required parameters
    config.vocab_size = vocab_size
    config.pad_idx = dataset.word_vocab["<PAD>"]  # FIXED pad_idx issue

    # Split dataset
    train_data, test_data = dataset.train_test_split()

    # Define models
    models = {
        "BiLSTM": BiLSTM(config, num_classes, dataset.word_vocab, config.pad_idx),
        "BiLSTM_CRF": BiLSTM_CRF(config, num_classes, dataset.word_vocab, config.pad_idx),
        "BiLSTM_Attention": BiLSTM_Attention(config, num_classes, dataset.word_vocab, config.pad_idx),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

