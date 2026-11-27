#!/usr/bin/env python3
"""Train a TF-IDF + feed-forward neural network sentiment classifier on IMDb reviews.

Pipeline
--------
1. Clean raw review text (strip HTML tags, normalise whitespace, lowercase).
2. Fit a TF-IDF vectorizer on the training split only (no leakage).
3. Train a small PyTorch MLP on the sparse TF-IDF features.
4. Evaluate on the held-out test split and persist every artifact needed for
   inference / Hugging Face deployment:

   model/model.pt          TensorFlow-independent PyTorch state dict + metadata
   model/vectorizer.pkl    Fitted sklearn TfidfVectorizer
   model/config.json       Architecture + vectorizer + training hyperparameters
   model/metrics.json      Test-split metrics and per-class report

Usage:
    python train.py                        # default: data/imdb_top_500.csv
    python train.py --epochs 60 --seed 7
"""

from __future__ import annotations

import argparse
import html
import json
import os
import pickle
import random
import re
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

DEFAULT_DATA = os.path.join("data", "imdb_top_500.csv")
DEFAULT_OUT = "model"

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# Text preprocessing
# --------------------------------------------------------------------------- #
def clean_text(raw: str) -> str:
    """Normalise a raw IMDb review.

    The dataset ships with literal ``<br />`` markers left over from the
    original HTML pages; unescape entities, drop tags and collapse whitespace.
    """
    text = html.unescape(str(raw))
    text = _TAG_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class MLPClassifier(nn.Module):
    """Feed-forward classifier over a (sparse) TF-IDF feature vector."""

    def __init__(self, input_dim: int, hidden_dims=(256, 64), dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for hidden in hidden_dims:
            layers += [
                nn.Linear(prev, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = hidden
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_dense_tensor(sparse_matrix) -> torch.Tensor:
    return torch.from_numpy(sparse_matrix.toarray().astype(np.float32))


@torch.no_grad()
def predict_proba(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    model.eval()
    logits = model(x)
    return torch.softmax(logits, dim=1).numpy()


def evaluate(y_true: np.ndarray, proba: np.ndarray) -> dict:
    pred = proba.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision_macro": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba[:, 1])) if len(set(y_true)) > 1 else None,
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
        "classification_report": classification_report(
            y_true, pred, target_names=["negative", "positive"], zero_division=0, output_dict=True
        ),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Train TF-IDF + MLP sentiment classifier")
    parser.add_argument("--data", default=DEFAULT_DATA, help="path to the labelled CSV")
    parser.add_argument("--out-dir", default=DEFAULT_OUT, help="artifact output directory")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=15, help="early-stopping patience on val F1")
    args = parser.parse_args()

    set_seed(args.seed)
    started = time.time()

    # ---- data -------------------------------------------------------------
    df = pd.read_csv(args.data)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"expected 'text' and 'label' columns, got {list(df.columns)}")
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].map(clean_text)
    df["label"] = df["label"].astype(int)
    df = df[df["text"].str.len() > 0]
    print(f"[data] {len(df)} labelled reviews | label balance {df['label'].value_counts().to_dict()}")

    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        df["text"].values,
        df["label"].values,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["label"].values,
    )

    # ---- vectorise (fit on train split only) ------------------------------
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=args.max_features,
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )
    x_train_np = vectorizer.fit_transform(x_train_raw)
    x_test_np = vectorizer.transform(x_test_raw)
    print(f"[tfidf] vocabulary size {len(vectorizer.vocabulary_)} -> {x_train_np.shape}")

    x_train = to_dense_tensor(x_train_np)
    x_test = to_dense_tensor(x_test_np)
    y_train_t = torch.from_numpy(y_train.astype(np.int64))
    y_test_t = torch.from_numpy(y_test.astype(np.int64))

    # ---- model ------------------------------------------------------------
    model = MLPClassifier(input_dim=x_train.shape[1], hidden_dims=(256, 64), dropout=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(x_train, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    best_state, best_f1, best_epoch, bad_epochs = None, -1.0, 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(dataset)

        train_metrics = evaluate(y_train, predict_proba(model, x_train))
        val_metrics = evaluate(y_test, predict_proba(model, x_test))
        scheduler.step(val_metrics["f1_macro"])
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "train_f1": train_metrics["f1_macro"],
             "test_f1": val_metrics["f1_macro"], "test_acc": val_metrics["accuracy"]}
        )
        print(
            f"[epoch {epoch:03d}] loss {train_loss:.4f} | train F1 {train_metrics['f1_macro']:.4f} "
            f"| test F1 {val_metrics['f1_macro']:.4f} acc {val_metrics['accuracy']:.4f}"
        )

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[early-stop] no improvement for {args.patience} epochs")
                break

    # ---- final evaluation with the best checkpoint -------------------------
    model.load_state_dict(best_state)
    final_metrics = evaluate(y_test, predict_proba(model, x_test))

    # ---- persist artifacts -------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "model.pt")
    vec_path = os.path.join(args.out_dir, "vectorizer.pkl")
    cfg_path = os.path.join(args.out_dir, "config.json")
    metrics_path = os.path.join(args.out_dir, "metrics.json")

    torch.save(
        {
            "state_dict": best_state,
            "input_dim": int(x_train.shape[1]),
            "hidden_dims": [256, 64],
            "dropout": 0.3,
            "num_classes": 2,
            "class_names": ["negative", "positive"],
        },
        ckpt_path,
    )
    with open(vec_path, "wb") as fh:
        pickle.dump(vectorizer, fh)

    config = {
        "task": "binary_sentiment_analysis",
        "dataset": os.path.basename(args.data),
        "n_samples": int(len(df)),
        "n_train": int(len(x_train_raw)),
        "n_test": int(len(x_test_raw)),
        "model_type": "MLP over TF-IDF features",
        "arch": {"input_dim": int(x_train.shape[1]), "hidden_dims": [256, 64], "dropout": 0.3},
        "vectorizer": {
            "type": "TfidfVectorizer",
            "ngram_range": [1, 2],
            "max_features": args.max_features,
            "min_df": 2,
            "sublinear_tf": True,
        },
        "training": {
            "epochs_run": len(history),
            "best_epoch": best_epoch,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "optimizer": "AdamW",
            "early_stopping_patience": args.patience,
            "seed": args.seed,
        },
        "labels": {"0": "negative", "1": "positive"},
    }
    with open(cfg_path, "w") as fh:
        json.dump(config, fh, indent=2)

    metrics = {
        **final_metrics,
        "best_test_f1_macro": best_f1,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "history": history,
        "train_seconds": round(time.time() - started, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)

    print("\n[result] best epoch", best_epoch)
    print(f"[result] test accuracy {final_metrics['accuracy']:.4f} | "
          f"macro F1 {final_metrics['f1_macro']:.4f} | ROC-AUC {final_metrics['roc_auc']}")
    print("[artifacts]", ckpt_path, vec_path, cfg_path, metrics_path, sep="\n  ")


if __name__ == "__main__":
    main()
