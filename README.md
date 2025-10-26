# Valorative Segment Tagging with Neural Sequence Models

This repository supports the experiments described in our SEPLN 2024 paper on automatic appraisal detection in YouTube comments using BiLSTM-based models. We focus on the attitude domain of **Appraisal Theory** (Martin & White, 2005), with a special emphasis on the **affect**, **judgement**, and **appreciation** subcategories.

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
ner_experiments/
├── configs/            # YAML hyperparameter configurations
├── models/             # BiLSTM, CRF, Attention variants
├── preprocessing/      # Tokenization, vocabulary, dataset loaders
├── experiments/        # Training, validation, logging
├── scripts/            # Sweeps, visualization, utilities
├── results/            # Output metrics and checkpoints
├── main.py             # Entry point for training a model
└── sweep_experiments.py # Grid search over hyperparameters
```

---

## Running a Single Experiment

```bash
python main.py
```

To run a programmatic experiment (for sweep or script usage):
```python
from main import run_experiment
hparams = {
    "lr": 1e-4,
    "batch_size": 32,
    "hidden_dim": 192,
    "dropout": 0.3,
    "model_type": "BiLSTM"
}
run_experiment(hparams, experiment_number=1)
```

---

## Running Multiple Experiments

```bash
python sweep_experiments.py
```

You can customize the hyperparameter search space in `sweep_experiments.py`.

---

## Metrics Logged
After each validation step:
- Precision
- Recall
- F1 Score
- Accuracy

Metrics are saved to `results/experiments_log.csv` along with:
- Timestamp
- Model type
- Hyperparameters

---

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
