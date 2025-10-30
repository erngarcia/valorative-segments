# Valorative Segment Tagging with Neural Sequence Models

This repository supports the experiments described in our 2025 paper on automatic appraisal detection in YouTube comments using BiLSTM-based models. We focus on the attitude domain of **Appraisal Theory** (Martin & White, 2005), with a special emphasis on the **affect**, **judgement**, and **appreciation** subcategories.

---

## Task Overview
The task is formulated as a **sequence labeling problem** where each token in a YouTube comment is tagged as part of a valorative segment (B-, I-) or background (O). Labels correspond to appraisal types: `B-judgement`, `I-affect`, etc.

We evaluate several neural architectures:
- **BiLSTM**
- **BiLSTM + CRF**
- **BiLSTM + Attention**

---

## Repo Structure
```
ner_valorative_segments/
├── experiments/
│   ├── configs/              # YAML experiment configurations
│   ├── run_experiment.py     # Single experiment logic
│   ├── grid_search.py        # Hyperparameter sweeps
│   ├── artifacts.py          # Saving metrics, predictions, and summaries
│   └── train.py              # Training and evaluation loop
├── models/                   # BiLSTM, CRF, and Attention architectures
├── preprocessing/            # Dataset and vocabulary utilities
├── scripts/                  # Auxiliary tools (e.g., label distribution analysis)
├── data/                     # Sample corpus for smoke tests
├── artifacts/                # Created automatically during experiments
└── main.py                   # CLI entry point

```

## Running a Single Experiment

### 1. Command-Line Execution

The main script supports command-line arguments via argparse:

```
python -m ner_valorative_segments.main \
    --lr 1e-3 1e-4 \
    --batch_size 16 32 \
    --hidden_dim 128 192 \
    --dropout 0.3 \
    --device cuda
```

All arguments are optional; the defaults perform a small grid search.

### 2. Programmatic Execution

```
from ner_valorative_segments.experiments.run_experiment import run_experiment

hparams = {
    "lr": 1e-4,
    "batch_size": 32,
    "hidden_dim": 192,
    "dropout": 0.3
}
run_experiment(hparams, experiment_number=1)
```

## Metrics Logged
After each validation step:
- Precision
- Recall
- F1 Score
- Accuracy

Results and artifacts are stored automatically under:
```
artifacts/
├── summary_<timestamp>.txt
├── errors_<timestamp>.json
└── sample_predictions_<timestamp>.txt
```

## Paper Information
This codebase supports the experiments for:

**Title:** An LSTM Approach to Appraisal Classification in YouTube Comments in Spanish
**Authors:** Luis Ernesto García Estrada, Adrián Vergara Heidke, Valentina Tretti Beckles
**Conference:** TBD

---

## Citation
TBD

---

## License
MIT License
