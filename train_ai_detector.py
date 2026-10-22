#!/usr/bin/env python3
"""Train DocVerify's local English AI-text classifier from a public dataset.

Default dataset: artem9k/ai-text-detection-pile (MIT license on the dataset card).
This script samples a balanced subset from the Hugging Face dataset in streaming
mode so it does not need to download the full ~2 GB dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import joblib
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline

DATASET_ID = "artem9k/ai-text-detection-pile"


def get_first(row, names):
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def label_to_int(raw):
    """Map dataset source labels to 0=human and 1=AI; reject unknown labels."""
    value = str(raw).strip().lower()
    if value in {"human", "human-written", "human written", "0", "false"}:
        return 0
    if value in {"ai", "ai-generated", "ai generated", "machine", "machine-generated", "1", "true"}:
        return 1
    if any(token in value for token in ("chatgpt", "gpt-2", "gpt2", "gpt-3", "gpt3", "gpt-j", "gptj", "ai-generated", "machine-generated")):
        return 1
    if "human" in value:
        return 0
    return None


def clean_text(value):
    if not isinstance(value, str):
        return ""
    value = re.sub(r"\s+", " ", value).strip()
    # Avoid enormous examples dominating the streaming run and keep inference bounded.
    return value[:6000]


def collect_balanced(max_per_class: int, min_chars: int, seed: int):
    print(f"Loading {DATASET_ID} in streaming mode…")
    stream = load_dataset(DATASET_ID, split="train", streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=20_000)
    texts = {0: [], 1: []}
    hashes = set()
    seen_rows = 0
    skipped_unknown = 0

    for row in stream:
        seen_rows += 1
        raw_text = get_first(row, ("text", "TEXT", "content", "document"))
        raw_label = get_first(row, ("source", "SOURCE", "label", "class", "target"))
        label = label_to_int(raw_label)
        text = clean_text(raw_text)
        if label is None:
            skipped_unknown += 1
            continue
        if len(text) < min_chars or len(text.split()) < 30:
            continue
        digest = hashlib.sha256(text.lower().encode("utf-8", errors="ignore")).hexdigest()
        if digest in hashes:
            continue
        hashes.add(digest)
        if len(texts[label]) < max_per_class:
            texts[label].append(text)
        if len(texts[0]) >= max_per_class and len(texts[1]) >= max_per_class:
            break
        if seen_rows % 50_000 == 0:
            print(f"Scanned {seen_rows:,} rows; human={len(texts[0]):,}, AI={len(texts[1]):,}")

    print(f"Finished scanning {seen_rows:,} rows (unknown labels skipped: {skipped_unknown:,}).")
    print(f"Collected human={len(texts[0]):,}, AI={len(texts[1]):,} unique examples.")
    if min(len(texts[0]), len(texts[1])) < 500:
        raise RuntimeError(
            "Not enough examples from both classes. Check dataset column names/labels and internet access."
        )
    # Enforce exact balance even if the stream ended early.
    n = min(len(texts[0]), len(texts[1]))
    combined_texts = texts[0][:n] + texts[1][:n]
    combined_labels = [0] * n + [1] * n
    return combined_texts, np.asarray(combined_labels, dtype=np.int8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-per-class", type=int, default=12000,
                        help="Maximum human examples and AI examples (default: 12000 each).")
    parser.add_argument("--min-chars", type=int, default=250,
                        help="Discard shorter training examples (default: 250 characters).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="models/ai_detector.joblib")
    parser.add_argument("--metrics", default="models/evaluation.json")
    args = parser.parse_args()
    if args.max_per_class < 500:
        parser.error("--max-per-class must be at least 500 for a minimally useful run.")

    texts, labels = collect_balanced(args.max_per_class, args.min_chars, args.seed)
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=args.seed, stratify=labels
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.98,
            max_features=120_000,
            sublinear_tf=True,
            dtype=np.float32,
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=args.seed,
            solver="liblinear",
        )),
    ])

    print(f"Training on {len(x_train):,} examples; testing on {len(x_test):,} held-out examples…")
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, list(pipeline.classes_).index(1)]
    metrics = {
        "dataset": DATASET_ID,
        "seed": args.seed,
        "total_samples": len(texts),
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "human_samples": int((labels == 0).sum()),
        "ai_samples": int((labels == 1).sum()),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_ai": float(precision_score(y_test, predictions, pos_label=1, zero_division=0)),
        "recall_ai": float(recall_score(y_test, predictions, pos_label=1, zero_division=0)),
        "f1_ai": float(f1_score(y_test, predictions, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix_labels_0_human_1_ai": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test, predictions, labels=[0, 1], target_names=["human", "AI"],
            output_dict=True, zero_division=0
        ),
        "warning": "Random held-out evaluation can overestimate real-world generalization if sources/styles overlap. Test on a separate dataset and modern AI writing before relying on results.",
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": pipeline,
        "metadata": {
            "model_name": "DocVerify TF-IDF + Logistic Regression",
            "dataset": DATASET_ID,
            "training_samples": len(texts),
            "classes": {"0": "human-written", "1": "AI-generated"},
            "metrics": {key: metrics[key] for key in ("accuracy", "precision_ai", "recall_ai", "f1_ai", "roc_auc")},
            "model_format_version": 1,
        },
    }
    joblib.dump(bundle, output_path, compress=3)
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nHELD-OUT EVALUATION (not a guarantee of real-world accuracy)")
    print(classification_report(y_test, predictions, labels=[0, 1], target_names=["human", "AI"], zero_division=0))
    print("Confusion matrix [human, AI]:")
    print(np.asarray(metrics["confusion_matrix_labels_0_human_1_ai"]))
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Saved model: {output_path.resolve()} ({output_path.stat().st_size / (1024 * 1024):.1f} MB)")
    print(f"Saved metrics: {metrics_path.resolve()}")
    print("\nDo not describe these numbers as proof that the detector identifies authorship reliably.")


if __name__ == "__main__":
    main()
