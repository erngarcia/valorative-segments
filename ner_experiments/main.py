import yaml
from experiments.run_experiment import run_experiment
import itertools

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def main():
    search_space = {
    "lr": [1e-3, 1e-4],
    "batch_size": [16, 32],
    "hidden_dim": [128, 192],
    "dropout": [0.3],
}

    # Create all combinations
    keys = list(search_space.keys())
    combinations = list(itertools.product(*search_space.values()))

    for i, combo in enumerate(combinations):
        hparams = dict(zip(keys, combo))
        run_experiment(hparams, experiment_number=i + 1)  
        print("=" * 80)  

if __name__ == "__main__":
    main()

