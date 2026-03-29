"""
Flask app for Fake News Detection inference and model dashboard.

Loads the trained Hybrid CNN-BiGRU-Attention model and uses
all-MiniLM-L6-v2 for real-time sentence embedding.

Run:  python app.py
"""

import sys, os, json, re

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import torch
from flask import Flask, render_template, request, jsonify

from config import (
    LLM_MODEL_NAME, EMBEDDING_DIM, MAX_SENTENCES,
    HIDDEN_DIM, NUM_CLASSES, NUM_LAYERS, DROPOUT,
    NUM_FILTERS, KERNEL_SIZES, MODELS_DIR, RESULTS_DIR,
)
from models import HybridCNNBiGRUAttention

app = Flask(__name__)

_st_model = None
_model = None
_device = None
_all_metrics = {}


def _split_sentences(text: str, max_s: int = MAX_SENTENCES) -> list[str]:
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    sents = [s.strip() for s in sents if len(s.strip()) > 10]
    return sents[:max_s] if sents else [text[:512]]


def _load():
    global _st_model, _model, _device, _all_metrics

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # sentence transformer
    from sentence_transformers import SentenceTransformer
    _st_model = SentenceTransformer(LLM_MODEL_NAME)

    # hybrid model
    weight_path = os.path.join(MODELS_DIR, "hybrid_cnn_bigru_attn.pt")
    if os.path.exists(weight_path):
        _model = HybridCNNBiGRUAttention(
            embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
            num_filters=NUM_FILTERS, kernel_sizes=KERNEL_SIZES,
            num_layers=NUM_LAYERS, num_classes=NUM_CLASSES,
            dropout=DROPOUT, use_cnn=True, use_attention=True,
        ).to(_device)
        _model.load_state_dict(
            torch.load(weight_path, map_location=_device, weights_only=True)
        )
        _model.eval()
        print(f"[app] Loaded hybrid model from {weight_path}")
    else:
        print(f"[app] WARNING: No model at {weight_path}. Run notebook 3 first.")

    # dashboard metrics
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for fp in sorted(os.listdir(RESULTS_DIR)):
        if fp.endswith(".json"):
            with open(os.path.join(RESULTS_DIR, fp)) as f:
                data = json.load(f)
            name = fp.replace("_metrics.json", "").replace("_", " ").title()
            _all_metrics[name] = data.get("test", data)


def _predict(text: str) -> dict:
    if _model is None or _st_model is None:
        return {"error": "Model not loaded. Train it first (notebook 3)."}

    sents = _split_sentences(text)
    embs = _st_model.encode(sents)

    x = np.zeros((1, MAX_SENTENCES, EMBEDDING_DIM), dtype=np.float32)
    x[0, :len(sents)] = embs[:MAX_SENTENCES]

    with torch.no_grad():
        logits, attn_w = _model(torch.tensor(x).to(_device))
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred = int(logits.argmax(1).item())
        weights = attn_w[0].cpu().numpy()[:len(sents)]

    sent_attention = [
        {"sentence": s, "weight": round(float(w), 4)}
        for s, w in zip(sents, weights)
    ]

    return {
        "label": "FAKE" if pred == 1 else "REAL",
        "confidence": round(float(probs.max()), 4),
        "prob_real": round(float(probs[0]), 4),
        "prob_fake": round(float(probs[1]), 4),
        "sentences": sent_attention,
    }


@app.route("/")
def index():
    return render_template("index.html", metrics=_all_metrics)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    return jsonify(_predict(text))


if __name__ == "__main__":
    _load()
    print("[app] Starting server at http://localhost:5000")
    app.run(debug=False, port=5000)
