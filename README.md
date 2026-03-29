# Fake News Detection Using LLM-Enhanced Hybrid CNN-BiGRU with Sequential Attention

University of Windsor — Intro to AI

## Team

| Name | Role |
|---|---|
| Revanth Katari | Model architecture & LLM embeddings |
| Kruthika Shantha Murthy | Data preprocessing & EDA |
| Naga Sai Bharath Potla | CNN + BiGRU implementation & hyperparameter tuning |
| Kavya Pagaria | Sequential attention & ablation / interpretability |
| Sai Srinivas Uppara | Experiments, evaluation & result documentation |

## Architecture

```
News Article
    │
    ▼
Sentence Splitting (up to 16 sentences)
    │
    ▼
all-MiniLM-L6-v2  →  384-dim embedding per sentence
    │
    ▼
CNN (multi-kernel: 3,5,7)  →  local feature extraction
    │
    ▼
BiGRU (2-layer, bidirectional)  →  sequential modeling
    │
    ▼
Sequential Attention  →  interpretable weighting
    │
    ▼
Fully Connected  →  Real / Fake
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run notebooks in order

Open each notebook in Jupyter and run all cells:

| # | Notebook | What it does | Time estimate |
|---|---|---|---|
| 1 | `notebooks/1_data_preparation.ipynb` | EDA + compute & cache LLM embeddings | ~15 min (GPU) / ~2 hr (CPU) |
| 2 | `notebooks/2_baseline_models.ipynb` | Train LogReg, BiLSTM, BiGRU baselines | ~10 min (GPU) |
| 3 | `notebooks/3_hybrid_model.ipynb` | Train proposed CNN-BiGRU-Attention | ~10 min (GPU) |
| 4 | `notebooks/4_attention_ablation.ipynb` | Ablation study + attention visualization | ~30 min (GPU) |
| 5 | `notebooks/5_experiment_results.ipynb` | Consolidated comparison & charts | Instant |

> **Tip:** Notebook 1 caches embeddings to `cache/`. After the first run, all subsequent notebooks load instantly.

### 3. Launch the web app

```bash
cd app
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

## Folder Structure

```
FINAL/
├── data/
│   └── WELFake_Dataset.csv          # 72,134 labeled articles
├── cache/                            # auto-created by notebook 1
│   ├── sentence_embeddings.npy       # (N, 16, 384) cached embeddings
│   └── labels.npy
├── src/
│   ├── config.py                     # all paths, hyperparameters
│   ├── models.py                     # BiLSTM, BiGRU, Hybrid CNN-BiGRU-Attn
│   ├── data_utils.py                 # data loading, embedding, splits
│   └── train_utils.py                # train/evaluate/attention utilities
├── notebooks/
│   ├── 1_data_preparation.ipynb      # Kruthika
│   ├── 2_baseline_models.ipynb       # Sai Srinivas
│   ├── 3_hybrid_model.ipynb          # Revanth
│   ├── 4_attention_ablation.ipynb    # Kavya
│   └── 5_experiment_results.ipynb    # Naga Sai Bharath
├── app/
│   ├── app.py                        # Flask server
│   └── templates/index.html          # Web UI
├── results/                          # auto-created (JSON metrics)
├── saved_models/                     # auto-created (.pt model weights)
├── requirements.txt
└── README.md
```

## Dataset

**WELFake** — 72,134 news articles labeled as Real (0) or Fake (1), collected from multiple online sources.

> Verma et al., "WELFake: A Large-Scale Dataset for Fake News Detection," IEEE TCSS, 2023.

## Models

| Model | Type | Input |
|---|---|---|
| Logistic Regression | Baseline (ML) | Mean-pooled 384-dim LLM embeddings |
| BiLSTM + Attention | Baseline (DL) | Sentence-level 384-dim embeddings |
| BiGRU + Attention | Baseline (DL) | Sentence-level 384-dim embeddings |
| **CNN-BiGRU-Attention** | **Proposed** | Sentence-level 384-dim embeddings |

## Portability

This project is fully self-contained. To run on another machine:

1. Copy the entire `FINAL/` folder
2. `pip install -r requirements.txt`
3. Run notebooks 1 through 5
4. Launch `app/app.py`

All paths are relative. No hardcoded directories. No pre-computed files required.
