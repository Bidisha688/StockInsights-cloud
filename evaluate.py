# evaluate.py
"""
Run offline evaluation for your project.

Usage examples:
  Summarization (ROUGE + BERTScore):
    python evaluate.py --task summarization --split "test[:50]"
  Sentiment (Accuracy + F1):
    python evaluate.py --task sentiment --split "test[:200]"
"""

import argparse
import json
import sys
import time
import random
from typing import List, Tuple

# --- Reproducibility (best-effort on CPU) ---
random.seed(42)

# --- Project imports (ensure these files exist in your repo) ---
try:
    from summarizer_local import summarize_chunks_local
except Exception as e:
    print("ERROR: Could not import summarize_chunks_local from summarizer_local.py")
    raise

try:
    from sentiment import finbert_sentiment_batch  # must return labels: 'negative'|'neutral'|'positive'
except Exception as e:
    print("ERROR: Could not import finbert_sentiment_batch from sentiment.py")
    raise

# --- External deps needed: datasets, rouge-score, bert-score, scikit-learn ---
from datasets import load_dataset
from rouge_score import rouge_scorer
from bert_score import score as bertscore
from sklearn.metrics import accuracy_score, f1_score, classification_report


# -------------------------
# Summarization Evaluation
# -------------------------
def eval_summarization(split: str = "test[:50]") -> dict:
    """
    Evaluate summarization with ROUGE-1 / ROUGE-L and BERTScore on CNN/DailyMail.
    Uses your summarize_chunks_local() as the system summarizer.
    """
    print(f"[Summarization] Loading dataset cnn_dailymail:{split} ...")
    ds = load_dataset("cnn_dailymail", "3.0.0", split=split)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    rouge1_f, rougeL_f, bert_f1 = [], [], []

    print(f"[Summarization] Evaluating {len(ds)} samples ...")
    for i, ex in enumerate(ds):
        article = ex["article"] or ""
        gold = (ex.get("highlights") or "").strip()
        if not gold:
            # skip empty gold
            continue

        # Use your project summarizer (we pass the whole article as one chunk for simplicity)
        sys_sum = summarize_chunks_local([article], bullets=8).strip()

        # ROUGE (F-measure)
        r = scorer.score(gold, sys_sum)
        rouge1_f.append(r["rouge1"].fmeasure)
        rougeL_f.append(r["rougeL"].fmeasure)

        # BERTScore (F1)
        P, R, F1 = bertscore([sys_sum], [gold], lang="en")
        bert_f1.append(float(F1.mean().item()))

        if (i + 1) % 10 == 0:
            print(f"  processed: {i+1}/{len(ds)}")

    # Aggregate
    metrics = {
        "rouge1_f": sum(rouge1_f) / max(len(rouge1_f), 1),
        "rougeL_f": sum(rougeL_f) / max(len(rougeL_f), 1),
        "bertscore_f1": sum(bert_f1) / max(len(bert_f1), 1),
        "num_samples": len(rouge1_f),
        "dataset": "cnn_dailymail:3.0.0",
        "split": split,
    }

    print("\n[Summarization] Results")
    print(f"  ROUGE-1 (F): {metrics['rouge1_f']:.4f}")
    print(f"  ROUGE-L (F): {metrics['rougeL_f']:.4f}")
    print(f"  BERTScore F1: {metrics['bertscore_f1']:.4f}")
    return metrics


# ---------------------
# Sentiment Evaluation
# ---------------------
# Label mapping expected from your sentiment.py
LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}

def eval_sentiment(split: str = "test[:200]") -> dict:
    """
    Evaluate sentiment with Accuracy and weighted F1 on Financial PhraseBank.
    Uses your finbert_sentiment_batch() from sentiment.py.
    """
    print(f"[Sentiment] Loading dataset financial_phrasebank:{split} ...")
    # 'sentences_50agree' is a common config for higher label agreement
    ds = load_dataset("financial_phrasebank", "sentences_50agree", split=split)

    texts: List[str] = ds["sentence"]
    gold_ids: List[int] = ds["label"]  # 0=negative, 1=neutral, 2=positive

    print(f"[Sentiment] Running model on {len(texts)} samples ...")
    preds: List[str] = finbert_sentiment_batch(texts, batch_size=16)

    try:
        pred_ids: List[int] = [LABEL_TO_ID[p] for p in preds]
    except KeyError as e:
        raise RuntimeError(
            f"Unexpected label from finbert_sentiment_batch: {e}. "
            f"Expected one of {list(LABEL_TO_ID.keys())}."
        )

    acc = accuracy_score(gold_ids, pred_ids)
    f1 = f1_score(gold_ids, pred_ids, average="weighted")

    print("\n[Sentiment] Results")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 (weighted): {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(gold_ids, pred_ids, target_names=["negative", "neutral", "positive"]))

    metrics = {
        "accuracy": float(acc),
        "f1_weighted": float(f1),
        "num_samples": len(texts),
        "dataset": "financial_phrasebank:sentences_50agree",
        "split": split,
    }
    return metrics


# ----------
# Utilities
# ----------
def save_results(task: str, split: str, metrics: dict) -> str:
    ts = int(time.time())
    fname = f"eval_{task}_{ts}.json"
    payload = {
        "task": task,
        "split": split,
        "metrics": metrics,
        "timestamp": ts,
    }
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved results to: {fname}")
    return fname


def main():
    parser = argparse.ArgumentParser(description="Evaluate summarization or sentiment for this project.")
    parser.add_argument("--task", choices=["summarization", "sentiment"], required=True,
                        help="Which evaluation to run.")
    parser.add_argument("--split", type=str,
                        default="test[:50]",
                        help="HF split slice (e.g., 'test[:50]' or 'train[:100]')")

    args = parser.parse_args()

    if args.task == "summarization":
        metrics = eval_summarization(split=args.split)
    else:
        # Provide a friendlier default for sentiment if the user leaves the default summarization split
        if args.split == "test[:50]":
            args.split = "test[:200]"
        metrics = eval_sentiment(split=args.split)

    save_results(args.task, args.split, metrics)


if __name__ == "__main__":
    # Make sure stdout flushes promptly when run from Streamlit subprocess
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
