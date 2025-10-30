from experiments.grid_search import run_grid_search
from experiments.artifacts import save_artifacts
import torch


def main():

    """
    Main entry point for running valorative segment tagging experiments.

    This script orchestrates the grid search across hyperparameter configurations,
    identifies the best-performing model, and saves the associated artifacts
    (metrics, confusion matrix, sample predictions, etc.).

    Usage:
        python -m ner_valorative_segments.main
    """

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the hyperparameter search space
    search_space = {
        "lr": [1e-3, 1e-4],
        "batch_size": [16, 32],
        "hidden_dim": [128, 192],
        "dropout": [0.3],
    }

    # Run all experiments and retrieve the best result
    best_run = run_grid_search(search_space, device)

    # Save artifacts (metrics, errors, predictions)
    save_artifacts(best_run, output_dir="artifacts")

    print("Grid search complete.")
    print(f"Best model: {best_run['model_name']}")
    print(f"Best F1 Score: {best_run['f1']:.4f}")


if __name__ == "__main__":
    main()
