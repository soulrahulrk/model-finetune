"""
Benchmarks the fine-tuned model's inference performance: latency (time-to-first-token proxy via
prompt processing + total generation time), throughput (tokens/sec), and VRAM footprint. Run
this against both the base model and the fine-tuned merged model to quantify the cost of the
fine-tune (should be ~0 — LoRA/QLoRA does not change inference-time architecture once merged)
and against the vLLM-served endpoint (Task 2) to quantify the serving-stack speedup.

Two modes:
  --backend transformers   Direct HF generate() loop (what infer.py uses) — measures raw model
                            performance, useful for comparing base vs fine-tuned in isolation.
  --backend vllm           Hits a running OpenAI-compatible vLLM server (see Task 2 guide) —
                            measures what production users actually experience, including
                            continuous batching effects.

Usage:
    python benchmark.py --backend transformers --model ../training/outputs/qwen3-8b-vedaz-merged-fp16 \
        --prompts-file bench_prompts.jsonl --output bench_results_transformers.json

    python benchmark.py --backend vllm --api-base http://localhost:8000/v1 --model qwen3-8b-vedaz \
        --prompts-file bench_prompts.jsonl --output bench_results_vllm.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

DEFAULT_PROMPTS = [
    "Mera career kab tak stabilize hoga? Main bahut confused hoon.",
    "What does my Saturn transit mean for my marriage prospects in the next year?",
    "Mujhe business mein bahut loss ho gaya hai, kya karu?",
    "Is this a good time to switch jobs? My Dasha seems to be changing soon.",
    "मेरी बेटी की शादी में देरी क्यों हो रही है?",
]


def load_prompts(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_PROMPTS
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompts.append(row["prompt"] if "prompt" in row else row["messages"][-1]["content"])
    return prompts


def bench_transformers(model_path: str, prompts: list[str], max_new_tokens: int) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    system_prompt = "You are Vedaz's AI Vedic astrologer. Give balanced, compassionate, non-fatalistic guidance."

    latencies, tok_per_sec, output_lens = [], [], []
    for prompt in prompts:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors="pt"
        ).to(model.device)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start

        n_new = out.shape[-1] - input_ids.shape[-1]
        latencies.append(elapsed)
        output_lens.append(n_new)
        tok_per_sec.append(n_new / elapsed if elapsed > 0 else 0)

    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None

    return {
        "backend": "transformers",
        "model": model_path,
        "num_prompts": len(prompts),
        "latency_seconds": {"mean": statistics.mean(latencies), "p50": statistics.median(latencies), "max": max(latencies)},
        "tokens_per_second": {"mean": statistics.mean(tok_per_sec), "min": min(tok_per_sec), "max": max(tok_per_sec)},
        "output_tokens": {"mean": statistics.mean(output_lens), "max": max(output_lens)},
        "peak_vram_gb": peak_vram_gb,
    }


def bench_vllm(api_base: str, model_name: str, prompts: list[str], max_new_tokens: int, api_key: str) -> dict:
    from openai import OpenAI

    client = OpenAI(base_url=api_base, api_key=api_key)
    system_prompt = "You are Vedaz's AI Vedic astrologer. Give balanced, compassionate, non-fatalistic guidance."

    latencies, tok_per_sec, output_lens = [], [], []
    for prompt in prompts:
        start = time.time()
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=0,
        )
        elapsed = time.time() - start
        n_new = resp.usage.completion_tokens
        latencies.append(elapsed)
        output_lens.append(n_new)
        tok_per_sec.append(n_new / elapsed if elapsed > 0 else 0)

    return {
        "backend": "vllm",
        "model": model_name,
        "api_base": api_base,
        "num_prompts": len(prompts),
        "latency_seconds": {"mean": statistics.mean(latencies), "p50": statistics.median(latencies), "max": max(latencies)},
        "tokens_per_second": {"mean": statistics.mean(tok_per_sec), "min": min(tok_per_sec), "max": max(tok_per_sec)},
        "output_tokens": {"mean": statistics.mean(output_lens), "max": max(output_lens)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["transformers", "vllm"], required=True)
    ap.add_argument("--model", required=True, help="Local model path (transformers) or served model name (vllm)")
    ap.add_argument("--api-base", default="http://localhost:8000/v1", help="vLLM OpenAI-compatible base URL")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--prompts-file", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--output", default="bench_results.json")
    args = ap.parse_args()

    prompts = load_prompts(args.prompts_file)
    print(f"Benchmarking {len(prompts)} prompts on backend={args.backend}, model={args.model}")

    if args.backend == "transformers":
        result = bench_transformers(args.model, prompts, args.max_new_tokens)
    else:
        result = bench_vllm(args.api_base, args.model, prompts, args.max_new_tokens, args.api_key)

    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
