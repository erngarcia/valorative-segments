import itertools
from typing import Dict, Any, List
from experiments.run_experiment import run_experiment


def run_grid_search(search_space: Dict[str, List[Any]], device):
    """
    Perform a grid search and return the best-performing experiment.

    Args:
        search_space: Dict of hyperparameter name -> list of values
        device: torch.device (CPU or GPU)

    Returns:
        Dict containing details of the best run (model, metrics, vocab, etc.)
    """
    keys = list(search_space.keys())
    combinations = list(itertools.product(*search_space.values()))

    best_f1 = 0.0
    best_run = {}

    for i, combo in enumerate(combinations, start=1):
        hparams = dict(zip(keys, combo))
        print(f"\n{'='*30} Experiment {i}/{len(combinations)} {'='*30}")
        print(f"Hyperparameters: {hparams}")

        (
            f1,
            metrics,
            confusion_matrix,
            model_name,
            errors,
            model,
            dataset,
            word_vocab,
            label_vocab,
        ) = run_experiment(hparams, experiment_number=i)

        if f1 > best_f1:
            best_f1 = f1
            best_run = {
                "f1": f1,
                "metrics": metrics,
                "matrix": confusion_matrix,
                "model_name": model_name,
                "errors": errors,
                "model": model,
                "dataset": dataset,
                "word_vocab": word_vocab,
                "label_vocab": label_vocab,
                "hparams": hparams,
            }

    return best_run
