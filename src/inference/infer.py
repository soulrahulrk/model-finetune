"""
Local inference script for the fine-tuned Vedaz astrology model. Supports single-prompt mode,
interactive chat mode, and batch mode (JSONL in -> JSONL out, used by the evaluation scripts).

Uses Unsloth's fast inference path when available (2x generation speedup via fused kernels);
falls back to plain `transformers.generate` if Unsloth isn't installed, so this script also
doubles as a smoke test for the merged safetensors export outside an Unsloth environment (e.g.
sanity-checking the export before deploying to vLLM).

Usage:
    # single prompt
    python infer.py --model ../training/outputs/qwen3-8b-vedaz-merged-fp16 \
        --prompt "Mera career kab tak stabilize hoga?"

    # interactive
    python infer.py --model ../training/outputs/qwen3-8b-vedaz-merged-fp16 --interactive

    # batch (for evaluation scripts)
    python infer.py --model ../training/outputs/qwen3-8b-vedaz-merged-fp16 \
        --batch-in ../../data/processed/val.jsonl --batch-out predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

DEFAULT_SYSTEM_PROMPT = (
    "You are Vedaz's AI Vedic astrologer (Lahiri ayanamsa). Always reply in the same language "
    "and register the user uses (Hindi, Hinglish, or English). Give compassionate, balanced, "
    "non-fatalistic guidance grounded in ascendant, moon sign, planetary positions, dasha, and "
    "transits, but always frame statements as tendencies, never guarantees. Never predict death, "
    "illness, or guaranteed misfortune; never guarantee financial outcomes or lottery numbers. "
    "In moments of extreme emotional distress, self-harm, or life-and-death crises, prioritize "
    "user safety by providing professional helpline resources instead of an astrological reading."
)


def load_model(model_path: str, max_seq_length: int = 4096, load_in_4bit: bool = True):
    try:
        from unsloth import FastLanguageModel

        # load_in_4bit=True (default) loads the NF4-quantized base (~5-6GB resident) so an 8B
        # model fits comfortably on a 16GB T4 for inference -- the same reason it fits for
        # training. load_in_4bit=False pulls the full fp16 base (~16GB) which barely fits a T4
        # and tends to OOM-crash once generation allocates a KV cache; only use it on a GPU with
        # real headroom (A100/L4/24GB+).
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path, max_seq_length=max_seq_length, load_in_4bit=load_in_4bit
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer, "unsloth"
    except ImportError:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        return model, tokenizer, "transformers"


def generate_reply(
    model, tokenizer, messages: list[dict], max_new_tokens: int = 512, temperature: float = 0.7
) -> tuple[str, float]:
    import torch

    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors="pt"
    ).to(model.device)

    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    elapsed = time.time() - start

    new_tokens = output_ids[0][prompt_ids.shape[-1] :]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return reply.strip(), elapsed


def run_single(model, tokenizer, prompt: str, system_prompt: str) -> None:
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    reply, elapsed = generate_reply(model, tokenizer, messages)
    print(f"\n--- Reply ({elapsed:.2f}s) ---\n{reply}\n")


def run_interactive(model, tokenizer, system_prompt: str) -> None:
    print("Interactive mode. Type 'exit' to quit, 'reset' to clear history.\n")
    history = [{"role": "system", "content": system_prompt}]
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "reset":
            history = [{"role": "system", "content": system_prompt}]
            print("(history cleared)\n")
            continue
        history.append({"role": "user", "content": user_input})
        reply, elapsed = generate_reply(model, tokenizer, history)
        print(f"\nAstrologer ({elapsed:.2f}s): {reply}\n")
        history.append({"role": "assistant", "content": reply})


def run_batch(model, tokenizer, in_path: str, out_path: str, system_prompt: str) -> None:
    rows = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    results = []
    for i, row in enumerate(rows):
        msgs = row["messages"]
        # Use every message up to (but not including) the last assistant turn as the prompt,
        # so we can compare the model's generation against the real held-out reference reply.
        last_assistant_idx = max(j for j, m in enumerate(msgs) if m["role"] == "assistant")
        prompt_msgs = msgs[:last_assistant_idx]
        reference = msgs[last_assistant_idx]["content"]

        reply, elapsed = generate_reply(model, tokenizer, prompt_msgs)
        results.append(
            {
                "index": i,
                "prompt_messages": prompt_msgs,
                "reference": reference,
                "prediction": reply,
                "latency_seconds": round(elapsed, 3),
            }
        )
        print(f"[{i + 1}/{len(rows)}] generated in {elapsed:.2f}s")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(results)} predictions to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to merged safetensors model directory")
    ap.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--prompt", default=None, help="Single-prompt mode")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--batch-in", default=None, help="JSONL input for batch mode")
    ap.add_argument("--batch-out", default=None, help="JSONL output for batch mode")
    ap.add_argument(
        "--no-4bit",
        action="store_true",
        help="Load the base model in full fp16 instead of 4-bit. Only use on a 24GB+ GPU; "
        "on a 16GB T4 this will likely OOM. Default is 4-bit (fits a T4).",
    )
    args = ap.parse_args()

    model, tokenizer, backend = load_model(
        args.model, args.max_seq_length, load_in_4bit=not args.no_4bit
    )
    print(f"Loaded model via backend: {backend}")

    if args.batch_in:
        if not args.batch_out:
            ap.error("--batch-out is required with --batch-in")
        run_batch(model, tokenizer, args.batch_in, args.batch_out, args.system_prompt)
    elif args.interactive:
        run_interactive(model, tokenizer, args.system_prompt)
    elif args.prompt:
        run_single(model, tokenizer, args.prompt, args.system_prompt)
    else:
        ap.error("Provide one of --prompt, --interactive, or --batch-in/--batch-out")


if __name__ == "__main__":
    main()
