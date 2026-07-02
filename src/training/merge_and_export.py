"""
Merges the trained LoRA adapter back into the base Qwen3-8B weights and exports in the three
formats a production deployment needs:

  1. Merged fp16/bf16 safetensors  -> used to serve with vLLM (Task 2)
  2. LoRA adapter only (safetensors) -> already saved by train_unsloth.py; re-saved here for
     completeness if this script is run standalone against an existing checkpoint dir
  3. GGUF (quantized)              -> for llama.cpp / Ollama / edge or CPU deployment

Usage:
    python merge_and_export.py --config training_config.yaml
    python merge_and_export.py --config training_config.yaml --skip-gguf   # faster iteration
"""

from __future__ import annotations

import argparse

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="training_config.yaml")
    ap.add_argument("--skip-gguf", action="store_true", help="Skip GGUF export (it is slow — llama.cpp conversion + quantization)")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    from unsloth import FastLanguageModel

    m_cfg = cfg["model"]
    e_cfg = cfg["export"]
    adapter_dir = e_cfg["adapter_only_dir"]

    print(f"Loading base model + adapter from {adapter_dir}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_dir,
        max_seq_length=m_cfg["max_seq_length"],
        dtype=m_cfg["dtype"],
        load_in_4bit=m_cfg["load_in_4bit"],
    )

    merged_dir = e_cfg["merged_fp16_dir"]
    print(f"Merging LoRA into base weights -> {merged_dir} (safetensors, fp16/bf16)")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    print("Merged safetensors export complete. This directory is what you point vLLM at.")

    if not args.skip_gguf:
        gguf_dir = e_cfg["gguf_dir"]
        for quant in e_cfg["gguf_quant_methods"]:
            print(f"Exporting GGUF ({quant}) -> {gguf_dir}")
            model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method=quant)
        print("GGUF export complete. Use these files with llama.cpp / Ollama / LM Studio.")
    else:
        print("Skipped GGUF export (--skip-gguf).")

    print("\nDone. Artifacts:")
    print(f"  Merged safetensors: {merged_dir}")
    print(f"  LoRA adapter only:  {adapter_dir}")
    if not args.skip_gguf:
        print(f"  GGUF quantized:     {e_cfg['gguf_dir']}")


if __name__ == "__main__":
    main()
