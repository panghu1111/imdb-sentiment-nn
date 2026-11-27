#!/usr/bin/env python3
"""Run sentiment inference with the trained TF-IDF + MLP pipeline.

The same script works against a local artifact directory (produced by
``train.py``) or directly against the Hugging Face Hub repository that CI/CD
publishes, so users never need to clone or retrain anything.

Examples
--------
    # local artifacts in ./model
    python predict.py --text "A quiet, beautifully acted film. I loved every minute."

    # interactively type reviews until Ctrl-D
    python predict.py --interactive

    # pull artifacts straight from the Hub
    python predict.py --hf-repo <hf-username>/imdb-sentiment-nn \
        --text "Two hours I will never get back."
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import torch

from train import MLPClassifier, clean_text, to_dense_tensor

ARTIFACTS = ("model.pt", "vectorizer.pkl", "config.json", "metrics.json")


def download_artifacts(repo_id: str, local_dir: str) -> str:
    """Fetch all deployment artifacts from the Hub into ``local_dir``."""
    from huggingface_hub import hf_hub_download

    os.makedirs(local_dir, exist_ok=True)
    for name in ARTIFACTS:
        try:
            hf_hub_download(repo_id=repo_id, filename=name, local_dir=local_dir)
        except Exception as exc:  # metrics.json is optional for inference
            print(f"[warn] could not download {name}: {exc}", file=sys.stderr)
    return local_dir


class SentimentPredictor:
    """Thin wrapper around the persisted model / vectorizer / config."""

    def __init__(self, artifact_dir: str = "model"):
        ckpt = torch.load(os.path.join(artifact_dir, "model.pt"), map_location="cpu")
        self.class_names = ckpt.get("class_names", ["negative", "positive"])
        self.model = MLPClassifier(
            input_dim=ckpt["input_dim"],
            hidden_dims=tuple(ckpt.get("hidden_dims", (256, 64))),
            dropout=ckpt.get("dropout", 0.3),
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        with open(os.path.join(artifact_dir, "vectorizer.pkl"), "rb") as fh:
            self.vectorizer = pickle.load(fh)

        cfg_path = os.path.join(artifact_dir, "config.json")
        self.config = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        x = to_dense_tensor(self.vectorizer.transform([clean_text(text)]))
        proba = torch.softmax(self.model(x), dim=1)[0].tolist()
        idx = int(max(range(len(proba)), key=proba.__getitem__))
        return {
            "label": self.class_names[idx],
            "label_id": idx,
            "confidence": round(proba[idx], 4),
            "probabilities": {name: round(p, 4) for name, p in zip(self.class_names, proba)},
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict sentiment for IMDb-style reviews")
    parser.add_argument("--text", help="review text to classify")
    parser.add_argument("--artifact-dir", default="model", help="local artifact directory")
    parser.add_argument("--hf-repo", help="Hugging Face repo id, e.g. user/imdb-sentiment-nn")
    parser.add_argument("--interactive", action="store_true", help="read reviews from stdin")
    args = parser.parse_args()

    artifact_dir = args.artifact_dir
    if args.hf_repo:
        artifact_dir = download_artifacts(args.hf_repo, args.artifact_dir)

    if not os.path.exists(os.path.join(artifact_dir, "model.pt")):
        raise SystemExit(
            f"no model.pt in '{artifact_dir}'. Run `python train.py` first, "
            "or pass --hf-repo <user>/<repo>."
        )

    predictor = SentimentPredictor(artifact_dir)

    if args.interactive:
        print("Type a review, press Enter. Empty line quits.")
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                break
            print(json.dumps(predictor.predict(line), ensure_ascii=False))
        return

    text = args.text or sys.stdin.read()
    if not text.strip():
        parser.error("provide --text, --interactive, or pipe text on stdin")
    print(json.dumps(predictor.predict(text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
