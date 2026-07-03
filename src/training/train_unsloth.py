"""
Production QLoRA fine-tuning script for Qwen3-8B on the Vedaz astrology chat dataset, using
Unsloth for 4-bit quantized loading, fused LoRA kernels, and Unsloth-optimized gradient
checkpointing.

Design decisions are documented in ../../docs/02_finetuning_strategy.md — this script is a
direct implementation of that document's section 5 (training configuration) and should not
diverge from it without updating the doc.

Requires a CUDA GPU (see docs/02_finetuning_strategy.md section 2 for VRAM sizing). Will not
run meaningfully on CPU.

Usage:
    python train_unsloth.py --config training_config.yaml
    python train_unsloth.py --config training_config.yaml --num_train_epochs 4 --learning_rate 1e-4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def load_config(path: str, overrides: dict) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # shallow CLI overrides land in the `training` block, the only block users typically
    # want to tweak per-run without editing the YAML.
    for k, v in overrides.items():
        if v is None:
            continue
        cfg["training"][k] = v
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="training_config.yaml")
    p.add_argument("--num_train_epochs", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--per_device_train_batch_size", type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--report_to", type=str, default=None, help='"none" or "wandb"')
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    overrides = {
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "output_dir": args.output_dir,
        "report_to": args.report_to,
    }
    cfg = load_config(args.config, overrides)

    # Imported lazily so `--help` and config-parsing work without a GPU/Unsloth installed,
    # which is convenient for CI-linting this script and for reading --help on a laptop.
    #
    # IMPORT ORDER MATTERS: unsloth must be imported before trl/transformers/peft. Unsloth's
    # patching (which fixes up chat-template EOS/PAD token wiring) only applies correctly if it
    # runs before trl's classes are touched -- importing trl first is a documented Unsloth bug
    # (https://github.com/unslothai/unsloth/issues/2797) that causes SFTConfig's eos_token to be
    # silently overwritten with the literal unresolved placeholder "<EOS_TOKEN>", which then
    # fails trl's vocabulary check. This exact ordering is also what Unsloth's own startup
    # warning has been telling us on every run ("Unsloth should be imported before [trl,
    # transformers, peft]") -- it's not just a performance tip, it's required for correctness.
    import re
    import torch
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from datasets import load_dataset
    from transformers import EarlyStoppingCallback
    from trl import SFTConfig, SFTTrainer

    m_cfg = cfg["model"]
    l_cfg = cfg["lora"]
    d_cfg = cfg["data"]
    t_cfg = cfg["training"]

    print(f"[1/7] Loading base model in 4-bit: {m_cfg['base_model']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=m_cfg["base_model"],
        max_seq_length=m_cfg["max_seq_length"],
        dtype=m_cfg["dtype"],
        load_in_4bit=m_cfg["load_in_4bit"],
    )

    print("[2/7] Attaching Qwen3 chat template")
    tokenizer = get_chat_template(tokenizer, chat_template="qwen3")

    # Some Unsloth/trl/transformers version combinations leave tokenizer.eos_token as an
    # unsubstituted chat-template placeholder string (e.g. literal "<EOS_TOKEN>") instead of a
    # real token, which trl's SFTTrainer then rejects with "not found in the vocabulary". Detect
    # and repair this by falling back to Qwen's real ChatML end-of-turn token before it's used.
    vocab = tokenizer.get_vocab()
    print(f"    Diagnostic: tokenizer.eos_token={tokenizer.eos_token!r}, "
          f"in_vocab={tokenizer.eos_token in vocab}, eos_token_id={tokenizer.eos_token_id!r}")
    if tokenizer.eos_token not in vocab:
        for candidate in ("<|im_end|>", "<|endoftext|>"):
            if candidate in vocab:
                print(f"    Note: tokenizer.eos_token was '{tokenizer.eos_token}' (not in vocab); "
                      f"resetting to '{candidate}'.")
                tokenizer.eos_token = candidate
                break
        else:
            raise RuntimeError(
                f"tokenizer.eos_token '{tokenizer.eos_token}' is not in the vocabulary and no "
                f"known fallback token (<|im_end|>, <|endoftext|>) was found either."
            )
    print(f"    Diagnostic: tokenizer.eos_token AFTER repair check={tokenizer.eos_token!r}")

    print(f"[3/7] Wrapping model with LoRA adapters (r={l_cfg['r']}, alpha={l_cfg['lora_alpha']})")
    model = FastLanguageModel.get_peft_model(
        model,
        r=l_cfg["r"],
        target_modules=l_cfg["target_modules"],
        lora_alpha=l_cfg["lora_alpha"],
        lora_dropout=l_cfg["lora_dropout"],
        bias=l_cfg["bias"],
        use_gradient_checkpointing=l_cfg["use_gradient_checkpointing"],
        random_state=l_cfg["random_state"],
    )

    print("[4/7] Loading and formatting datasets")

    def formatting_func(examples):
        convos = examples["messages"]
        texts = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=m_cfg["enable_thinking"],
            )
            for convo in convos
        ]
        return {"text": texts}

    data_files = {"train": d_cfg["train_file"], "validation": d_cfg["val_file"]}
    dataset = load_dataset("json", data_files=data_files)
    dataset = dataset.map(formatting_func, batched=True)
    print(f"    train examples: {len(dataset['train'])} | val examples: {len(dataset['validation'])}")

    print("[5/7] Configuring SFTTrainer")
    Path(t_cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    # trl has repeatedly renamed/relocated the SFT-specific fields (max_seq_length -> max_length,
    # dataset_text_field / packing moving in and out of SFTConfig vs. SFTTrainer kwargs) across
    # versions. Rather than hard-code one version's field names, build the full desired kwarg set,
    # try constructing SFTConfig, and on a TypeError naming an unexpected keyword, drop/rename that
    # one field and retry -- until construction succeeds or we run out of optional fields to drop.
    sft_config_kwargs = {
        "output_dir": t_cfg["output_dir"],
        "num_train_epochs": t_cfg["num_train_epochs"],
        "per_device_train_batch_size": t_cfg["per_device_train_batch_size"],
        "per_device_eval_batch_size": t_cfg["per_device_eval_batch_size"],
        "gradient_accumulation_steps": t_cfg["gradient_accumulation_steps"],
        "learning_rate": t_cfg["learning_rate"],
        "lr_scheduler_type": t_cfg["lr_scheduler_type"],
        "warmup_ratio": t_cfg["warmup_ratio"],
        "optim": t_cfg["optim"],
        "weight_decay": t_cfg["weight_decay"],
        "max_grad_norm": t_cfg["max_grad_norm"],
        "bf16": t_cfg["bf16"] and is_bfloat16_supported(),
        "fp16": not is_bfloat16_supported(),
        "logging_steps": t_cfg["logging_steps"],
        "eval_strategy": t_cfg["eval_strategy"],
        "save_strategy": t_cfg["save_strategy"],
        "save_total_limit": t_cfg["save_total_limit"],
        "load_best_model_at_end": t_cfg["load_best_model_at_end"],
        "metric_for_best_model": t_cfg["metric_for_best_model"],
        "greater_is_better": t_cfg["greater_is_better"],
        "seed": t_cfg["seed"],
        "report_to": t_cfg["report_to"],
        "max_seq_length": m_cfg["max_seq_length"],
        "dataset_text_field": "text",
        "packing": d_cfg["packing"],
        # Some Unsloth-compiled SFTConfig wrappers add their own `eos_token` field with a
        # placeholder default ("<EOS_TOKEN>") that's only overridden if explicitly passed in --
        # repairing tokenizer.eos_token alone (above) does not reach this separate field, so we
        # pass the real value through explicitly. Harmless no-op if this trl version doesn't
        # recognize the field (the retry loop below just drops it).
        "eos_token": tokenizer.eos_token,
    }

    def _build_sft_config(kwargs):
        kwargs = dict(kwargs)
        dropped = []
        while True:
            try:
                return SFTConfig(**kwargs), dropped
            except TypeError as e:
                m = re.search(r"unexpected keyword argument '(\w+)'", str(e))
                if not m or m.group(1) not in kwargs:
                    raise
                bad_key = m.group(1)
                del kwargs[bad_key]
                dropped.append(bad_key)
                if bad_key == "max_seq_length" and "max_length" not in kwargs:
                    kwargs["max_length"] = m_cfg["max_seq_length"]

    sft_config, dropped_fields = _build_sft_config(sft_config_kwargs)
    if dropped_fields:
        print(f"    Note: this trl version's SFTConfig doesn't accept {dropped_fields}; "
              f"proceeding without them.")
    print(f"    Diagnostic: sft_config.eos_token={getattr(sft_config, 'eos_token', '<attr not set>')!r}")

    # Unsloth caches its JIT-generated SFTTrainer/SFTConfig wrapper code on disk and reuses it
    # across runs to avoid recompiling -- if a previous run generated a broken wrapper (e.g. with
    # an unsubstituted eos_token placeholder baked in), it can keep being reused even after this
    # script changes, silently ignoring any fix made above. Force a fresh generation every run.
    import shutil as _shutil
    compiled_cache_dir = Path(__file__).parent / "unsloth_compiled_cache"
    if compiled_cache_dir.exists():
        print(f"    Clearing stale Unsloth compiled cache at {compiled_cache_dir}")
        _shutil.rmtree(compiled_cache_dir, ignore_errors=True)

    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["validation"],
        "args": sft_config,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=t_cfg["early_stopping_patience"])],
    }

    try:
        # Older trl / transformers versions use tokenizer=
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)
    except TypeError as e:
        if "tok" in str(e).lower() or "processing_class" in str(e).lower():
            # Newer trl / transformers versions renamed it to processing_class=
            trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
        else:
            raise e

    if d_cfg["train_on_responses_only"]:
        print("[6/7] Masking loss to assistant-only tokens")
        trainer = train_on_responses_only(
            trainer,
            instruction_part=d_cfg["instruction_part"],
            response_part=d_cfg["response_part"],
        )

    print("[7/7] Training")
    gpu_stats = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    if gpu_stats:
        print(f"    GPU: {gpu_stats.name} | Total VRAM: {gpu_stats.total_memory / 1e9:.1f} GB")

    trainer_stats = trainer.train()

    print("\nTraining complete. Metrics:")
    print(json.dumps(trainer_stats.metrics, indent=2))

    adapter_dir = cfg["export"]["adapter_only_dir"]
    print(f"\nSaving LoRA adapter + tokenizer to {adapter_dir}")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    metrics_path = Path(t_cfg["output_dir"]) / "final_train_metrics.json"
    metrics_path.write_text(json.dumps(trainer_stats.metrics, indent=2), encoding="utf-8")
    print(f"Saved training metrics to {metrics_path}")
    print("\nNext step: python merge_and_export.py --config training_config.yaml")


if __name__ == "__main__":
    main()
