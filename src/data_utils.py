"""
Data loading, sentence-level embedding computation, and split utilities.


Workflow:
    1.  load_dataset()       -> texts, labels
    2.  compute_embeddings() -> (N, MAX_SENTENCES, 384)  — cached to .npy
    3.  create_splits()      -> train / val / test numpy arrays
    4.  make_dataloader()    -> PyTorch DataLoader
"""

import os
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


# ── helpers ────────────────────────────────────────────────────
def _split_sentences(text, max_sentences=16):
    """Split text into sentences, truncate to max_sentences."""
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    sents = [s.strip() for s in sents if len(s.strip()) > 10]
    if not sents:
        sents = [text[:512]]
    return sents[:max_sentences]


# ── public API ─────────────────────────────────────────────────
def load_dataset(dataset_path, text_col="text", title_col="title",
                 label_col="label"):
    """Return (list[str], np.ndarray) of full-texts and integer labels."""
    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=[text_col]).reset_index(drop=True)

    title = df[title_col].fillna("").astype(str) if title_col in df.columns else ""
    text  = df[text_col].fillna("").astype(str)
    full_text = (title + ". " + text).str.strip() if title_col in df.columns else text

    labels = df[label_col].values.astype(int)
    return full_text.tolist(), labels


def compute_embeddings(
    texts,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    max_sentences=16,
    embedding_dim=384,
    cache_path=None,
    batch_size=256,
):
    """
    Encode each article as a matrix of sentence embeddings.
    Returns shape (N, max_sentences, embedding_dim).
    Caches to *cache_path* on first run.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"[data] Loading cached embeddings from {cache_path}")
        data = np.load(cache_path)
        if data.dtype == np.float16:
            data = data.astype(np.float32)
        return data

    from sentence_transformers import SentenceTransformer
    print(f"[data] Loading embedding model: {model_name}")
  
    model = SentenceTransformer(model_name, device="cuda")

    # flatten all sentences with a map back to their article
    all_sents = []
    article_map = []          # (start_idx, count)
    for i, text in enumerate(texts):
        sents = _split_sentences(text, max_sentences)
        article_map.append((len(all_sents), len(sents)))
        all_sents.extend(sents)
        if (i + 1) % 10_000 == 0:
            print(f"  Segmented {i + 1}/{len(texts)} articles")

    print(f"[data] {len(all_sents):,} sentences from {len(texts):,} articles")
    print("[data] Encoding (may take a few minutes on GPU, longer on CPU) ...")

    sent_vecs = model.encode(all_sents, show_progress_bar=True,
                             batch_size=batch_size)

    embeddings = np.zeros((len(texts), max_sentences, embedding_dim),
                          dtype=np.float32)
    for i, (start, cnt) in enumerate(article_map):
        n = min(cnt, max_sentences)
        embeddings[i, :n] = sent_vecs[start:start + n]

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, embeddings.astype(np.float16))
        mb = os.path.getsize(cache_path) / 1024 ** 2
        print(f"[data] Saved cache ({mb:.0f} MB) -> {cache_path}")

    return embeddings


def create_splits(embeddings, labels, test_size=0.15, val_size=0.15, seed=42):
    """Stratified train / val / test split."""
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        embeddings, labels, test_size=test_size,
        random_state=seed, stratify=labels,
    )
    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_frac,
        random_state=seed, stratify=y_tmp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_dataloader(X, y, batch_size=64, shuffle=True):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
