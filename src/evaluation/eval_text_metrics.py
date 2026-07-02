"""
Computes BLEU, ROUGE, and BERTScore between model predictions and reference (real) replies.

Input is the predictions.jsonl produced by src/inference/infer.py --batch-in/--batch-out, which
has one {"prompt_messages", "reference", "prediction", ...} row per held-out example.

Usage:
    python eval_text_metrics.py --predictions eval_outputs/predictions.jsonl \
        --output eval_outputs/text_metrics.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load_predictions(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_bleu(predictions: list[str], references: list[str]) -> dict:
    import sacrebleu

    # sacrebleu expects references as list-of-lists (multiple refs per hypothesis supported;
    # we have exactly one reference per prediction here).
    bleu = sacrebleu.corpus_bleu(predictions, [references], tokenize="intl")
    per_example = [
        sacrebleu.sentence_bleu(p, [r], tokenize="intl").score for p, r in zip(predictions, references)
    ]
    return {
        "corpus_bleu": bleu.score,
        "per_example_mean": statistics.mean(per_example),
        "per_example_median": statistics.median(per_example),
    }


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for p, r in zip(predictions, references):
        result = scorer.score(r, p)  # (target, prediction)
        for key in scores:
            scores[key].append(result[key].fmeasure)
    return {k: {"mean_f1": statistics.mean(v), "median_f1": statistics.median(v)} for k, v in scores.items()}


def compute_bertscore(predictions: list[str], references: list[str]) -> dict:
    from bert_score import score as bertscore_score

    # Multilingual model (not the English-only default) — required given the dataset's
    # Hindi/Hinglish/English code-mixing (see docs/01_dataset_analysis.md section 3).
    P, R, F1 = bertscore_score(
        predictions, references, model_type="bert-base-multilingual-cased", lang="hi", verbose=False
    )
    return {
        "precision_mean": P.mean().item(),
        "recall_mean": R.mean().item(),
        "f1_mean": F1.mean().item(),
        "f1_median": F1.median().item(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = load_predictions(args.predictions)
    predictions = [r["prediction"] for r in rows]
    references = [r["reference"] for r in rows]

    print(f"Scoring {len(rows)} prediction/reference pairs...")
    results = {
        "num_examples": len(rows),
        "bleu": compute_bleu(predictions, references),
        "rouge": compute_rouge(predictions, references),
        "bertscore": compute_bertscore(predictions, references),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nSaved to {args.output}")
    print(
        "\nReminder: treat BLEU/ROUGE as coarse regression signals only, not quality scores — "
        "see docs/04_evaluation_guide.md sections 1-3 for why paraphrase-heavy, open-ended "
        "responses like these make lexical-overlap metrics an unreliable sole measure of quality."
    )


if __name__ == "__main__":
    main()
