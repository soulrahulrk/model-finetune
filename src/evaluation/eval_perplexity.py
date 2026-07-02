"""
Computes teacher-forced perplexity of the model on the held-out validation set, masking loss to
assistant-only tokens (consistent with how the model was trained — see
docs/02_finetuning_strategy.md section 5, "Loss masking"). This is the same quantity
`eval_strategy="epoch"` tracks during training; this script re-derives it standalone against a
specific saved checkpoint for reporting/comparison purposes.

Usage:
    python eval_perplexity.py --model ../training/outputs/qwen3-8b-vedaz-merged-fp16 \
        --val-file ../../data/processed/val.jsonl --output eval_outputs/perplexity.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--val-file", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-seq-length", type=int, default=4096)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    rows = []
    with open(args.val_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    per_example_ppl = []
    total_loss_sum = 0.0
    total_token_count = 0

    with torch.no_grad():
        for row in rows:
            messages = row["messages"]
            input_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False, enable_thinking=False, return_tensors="pt"
            ).to(model.device)
            if input_ids.shape[-1] > args.max_seq_length:
                input_ids = input_ids[:, : args.max_seq_length]

            # Mask everything up to and including the last "<|im_start|>assistant\n" boundary so
            # perplexity is measured only on assistant tokens, matching the training loss mask.
            labels = input_ids.clone()
            assistant_marker_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
            marker_len = len(assistant_marker_ids)
            seq = input_ids[0].tolist()
            last_start = None
            for i in range(len(seq) - marker_len, -1, -1):
                if seq[i : i + marker_len] == assistant_marker_ids:
                    last_start = i + marker_len
                    break
            if last_start is None:
                continue  # skip malformed example rather than scoring the whole sequence
            labels[0, :last_start] = -100

            outputs = model(input_ids=input_ids, labels=labels)
            n_scored_tokens = (labels != -100).sum().item()
            if n_scored_tokens == 0:
                continue

            loss = outputs.loss.item()
            per_example_ppl.append(math.exp(loss))
            total_loss_sum += loss * n_scored_tokens
            total_token_count += n_scored_tokens

    corpus_ppl = math.exp(total_loss_sum / total_token_count) if total_token_count else float("nan")

    result = {
        "model": args.model,
        "val_file": args.val_file,
        "num_examples_scored": len(per_example_ppl),
        "corpus_perplexity": corpus_ppl,
        "per_example_perplexity_mean": sum(per_example_ppl) / len(per_example_ppl) if per_example_ppl else None,
        "per_example_perplexity_min": min(per_example_ppl) if per_example_ppl else None,
        "per_example_perplexity_max": max(per_example_ppl) if per_example_ppl else None,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
