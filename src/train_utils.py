"""
Training and evaluation utilities for PyTorch models.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(X)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    return (running_loss / len(loader.dataset),
            accuracy_score(all_labels, all_preds))


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits, _ = model(X)
            loss = criterion(logits, y)

            running_loss += loss.item() * X.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    metrics = {
        "accuracy":  round(accuracy_score(all_labels, all_preds), 4),
        "precision": round(precision_score(all_labels, all_preds, average="weighted", zero_division=0), 4),
        "recall":    round(recall_score(all_labels, all_preds, average="weighted", zero_division=0), 4),
        "f1":        round(f1_score(all_labels, all_preds, average="weighted", zero_division=0), 4),
        "auc":       round(roc_auc_score(all_labels, all_probs), 4),
    }
    return avg_loss, metrics, np.array(all_preds), np.array(all_probs), np.array(all_labels)


def get_attention_weights(model, loader, device, max_samples=200):
    """Return attention weights and labels for visualization."""
    model.eval()
    weights_list, labels_list, preds_list = [], [], []

    with torch.no_grad():
        collected = 0
        for X, y in loader:
            X = X.to(device)
            _, attn_w = model(X)
            weights_list.append(attn_w.cpu().numpy())
            labels_list.extend(y.numpy())
            preds_list.extend(_.argmax(1).cpu().numpy())
            collected += X.size(0)
            if collected >= max_samples:
                break

    return (np.concatenate(weights_list)[:max_samples],
            np.array(labels_list)[:max_samples],
            np.array(preds_list)[:max_samples])
