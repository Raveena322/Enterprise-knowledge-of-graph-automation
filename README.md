# Enterprise Knowledge Graph Automation

A self-contained, synthetic Phase-I experiment comparing three methods for validating candidate knowledge-graph triples against enterprise-style sentences:

- Rule-based trigger matching
- A lightweight Open-IE-style baseline
- A trained word/character TF-IDF model with engineered features

> **Scope:** The dataset is generated from templates in `dataset.py`; it is not a real enterprise corpus. Results measure behavior on this synthetic benchmark and should not be interpreted as evidence of production performance or generalization to real documents.

## Setup

Use Python 3.10 or newer, then install the dependencies:

```text
pip install -r requirements.txt
```

## Run

From the project root:

```text
python run_experiment.py
python make_figures.py
```

The experiment writes `results/results.json`. Figure generation reads that file and writes five PNGs under `figures/`. Generated output folders and Python cache files are intentionally excluded from version control.

To generate and summarize a standalone synthetic dataset:

```text
python dataset.py
```

## Reproducibility

The experiment seeds its Python random-number generator and uses fixed random states for dataset splitting, feature-model training, and epoch shuffling. Importing `dataset.py` does not change the caller's random state.
