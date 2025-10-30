"""
Main entry point for valorative segment tagging experiments.

This script supports both direct execution and CLI-based configuration
via command-line arguments. It runs a grid search over hyperparameters,
selects the best-performing model, and saves results to `artifacts/`.
"""

import argparse
import torch
from experiments.grid_search import run_grid_search
from experiments.artifacts import save_artifacts


def parse_args():
    """Parse command-line arguments for experiment configuration."""
    parser = argparse.ArgumentParser(
        description="Run grid search experiments for valorative segment tagging."
    )

    parser.add_argument(
        "--lr", nargs="+", type=float, default=[1e-3, 1e-4],
        help="Learning rates to explore (default: [1e-3, 1e-4])"
    )
    parser.add_argument(
        "--batch_size", nargs="+", type=int, default=[16, 32],
        help="Batch sizes to explore (default: [16, 32])"
    )
    parser.add_argument(
        "--hidden_dim", nargs="+", type=int, default=[128, 192],
        help="Hidden dimensions to explore (default: [128, 192])"
    )
    parser.add_argument(
        "--dropout", nargs="+", type=float, default=[0.3],
        help="Dropout rates to explore (default: [0.3])"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default=None,
        help="Force device (cpu/cuda). Defaults to auto-detect."
    )

    return parser.parse_args()


def main():
    """Execute grid search experiments with optional CLI arguments."""
    args = parse_args()

    # Set reproducibility
    torch.manual_seed(args.seed)
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Build search space dynamically from CLI args
    search_space = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
    }

    print(f"\nStarting grid search on device: {device}")
    print(f"Search space: {search_space}")

    # Run experiments
    best_run = run_grid_search(search_space, device)

    # Save artifacts
    save_artifacts(best_run, output_dir="artifacts")

    # Report summary
    print("\nGrid search complete.")
    print(f"Best model: {best_run['model_name']}")
    print(f"Best F1 Score: {best_run['f1']:.4f}")


if __name__ == "__main__":
    main()
