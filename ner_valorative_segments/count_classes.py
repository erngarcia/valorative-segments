from collections import Counter
from preprocessing.dataset import NERDataset
from experiments.configs.configs import Config
import yaml
import os

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# Load config and dataset
config_path = "experiments/configs/base.yaml"
config_dict = load_config(config_path)
config = Config(config_dict)

dataset = NERDataset(config.data_path, config)

# Count labels
label_counts = Counter()
for _, label_seq in dataset.data:
    label_counts.update(label_seq)

# Map indices back to labels
idx_to_label = {idx: label for label, idx in dataset.label_vocab.items()}
pretty_counts = {idx_to_label[k]: v for k, v in label_counts.items()}

# Sort and print
for label, count in sorted(pretty_counts.items(), key=lambda x: x[0]):
    print(f"{label}: {count}")