"""
Merges every training-data source (real conversations cleaned by
analyze_and_clean.py, the 100 synthetic conversations, and the 5 manually
written long-form consultations) into one corpus, deduplicates across
sources, and writes a stratified 90/10 train/val split.

Stratification is done PER SOURCE FILE (not per topic) because the real
dataset has only 50 examples total -- a naive global shuffle-split could by
chance put 0 real examples in validation, which would make the eval loss
meaningless for the thing we actually care about (does the model still sound
like the real data). Splitting each source 90/10 independently and then
concatenating guarantees every source is represented in both splits.

Usage:
    python split_dataset.py \
        --inputs ../../data/processed/clean_dataset.jsonl \
                 ../../data/synthetic/synthetic_astrology_100.jsonl \
                 ../../data/manual_conversations/manual_conversations.jsonl \
        --out-dir ../../data/processed \
        --val-ratio 0.1 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON — {e}") from e
    return rows


def conv_hash(messages: list[dict]) -> str:
    return hashlib.sha256(
        json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seen_hashes: set[str] = set()
    train_rows, val_rows = [], []
    per_source_counts = {}

    for input_path in args.inputs:
        path = Path(input_path)
        rows = load_jsonl(path)
        deduped = []
        for r in rows:
            msgs = r.get("messages")
            if not msgs:
                continue
            h = conv_hash(msgs)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            deduped.append({"messages": msgs})

        rng.shuffle(deduped)
        n_val = max(1, round(len(deduped) * args.val_ratio)) if len(deduped) >= 5 else 0
        val_rows.extend(deduped[:n_val])
        train_rows.extend(deduped[n_val:])
        per_source_counts[path.name] = {"total": len(deduped), "val": n_val, "train": len(deduped) - n_val}

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / "val.jsonl", "w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "per_source": per_source_counts,
        "total_unique_conversations": len(train_rows) + len(val_rows),
        "train_count": len(train_rows),
        "val_count": len(val_rows),
    }
    (out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
