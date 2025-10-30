import os
import json
from datetime import datetime
from typing import Dict, Any
import torch
from experiments.inference import annotate_examples


def save_artifacts(best_run: Dict[str, Any], output_dir: str = "artifacts") -> None:
    """Save error logs, predictions, and summaries."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = best_run["model_name"]

    errors_path = os.path.join(output_dir, f"errors_{model_name}_{timestamp}.json")
    with open(errors_path, "w", encoding="utf-8") as f:
        json.dump(best_run["errors"], f, indent=4, ensure_ascii=False)
    print(f"[+] Error report saved to {errors_path}")

    word_vocab, label_vocab = best_run["word_vocab"], best_run["label_vocab"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample_sentences = [
        "cuándo han protegido a un hombre cuando una mujer lo golpea sabes lo que la gente dice es maricón por no defenderse...",
        "literalmente estás diciendo que las matan por ser mujeres te das cuenta de lo estúpido que suena eso",
        "un violador es una persona enferma y no le importará si su víctima es mujer o niño",
    ]

    preds_path = os.path.join(output_dir, f"sample_predictions_{model_name}_{timestamp}.txt")
    with open(preds_path, "w", encoding="utf-8") as f:
        for i, text in enumerate(sample_sentences, start=1):
            preds = annotate_examples(best_run["model"], text, word_vocab, label_vocab, device)
            f.write(f"Example {i}\nText: {text}\nPredictions: {preds}\n{'-'*80}\n")

    print(f"[+] Sample predictions saved to {preds_path}")


    summary_path = os.path.join(output_dir, f"summary_{model_name}_{timestamp}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Best Model Summary\n" + "=" * 80 + "\n")
        f.write(f"Model: {model_name}\n\nBest Hyperparameters:\n")
        for k, v in best_run["hparams"].items():
            f.write(f"  - {k}: {v}\n")

        f.write("\n📊 Metrics:\n")
        for k, v in best_run["metrics"].items():
            if k != "confusion_matrix":
                f.write(f"  - {k}: {v:.4f}\n")

        f.write("\nConfusion Matrix:\n")
        for row in best_run["matrix"]:
            f.write("  " + "\t".join(map(str, row)) + "\n")

    print(f"[+] Model summary saved to {summary_path}")
