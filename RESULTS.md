# Training Results — Qwen3-8B Vedaz Astrology QLoRA Fine-Tune

This documents an actual, completed fine-tuning run of the pipeline in this repo, executed on a
free Google Colab **Tesla T4 (16 GB)** GPU via
[`upload_to_colab/Train_Qwen3_Vedaz_Astrology.ipynb`](upload_to_colab/Train_Qwen3_Vedaz_Astrology.ipynb).
Evidence artifacts are in [`evidence/`](evidence/).

## Final metrics (real run)

| Metric | Value |
|---|---|
| Base model | `Qwen/Qwen3-8B` (4-bit NF4, Unsloth) |
| Method | QLoRA, rank 16, α 16, dropout 0.05 |
| Trainable params | 43,646,976 of 8,234,382,336 (**0.53 %**) |
| Train examples / Val examples | 139 / 16 |
| Epochs / Total steps | 3 / 54 |
| Effective batch size | 8 (2 per-device × 4 grad-accum) |
| **Final train loss** | **1.4033** |
| **Final eval loss** | **1.502** |
| Train runtime | ~18.7 min (1122 s) |
| GPU | Tesla T4, 15.6 GB VRAM |
| Peak VRAM headroom | Comfortable — no OOM at seq-len 4096 |

Loss decreased smoothly across all 3 epochs (see the per-step log in
[`evidence/training_run_log.txt`](evidence/training_run_log.txt)). The run was **reproducible**:
two independent full runs produced train losses of 1.4033 and 1.4035 respectively.

## Artifacts produced

| Artifact | Path (on Colab) | Purpose |
|---|---|---|
| LoRA adapter | `outputs/qwen3-8b-vedaz-adapter/` (`adapter_model.safetensors`, 174 MB) | Small, portable; load on top of base Qwen3-8B |
| Merged fp16 model | `outputs/qwen3-8b-vedaz-merged-fp16/` | Full standalone model — what you point vLLM at (see `docs/03_vllm_vps_hosting_guide.md`) |
| Training metrics JSON | `outputs/qwen3-8b-vedaz-qlora/final_train_metrics.json` | Machine-readable metrics |

Both the adapter and the merged fp16 export completed successfully. (Model weight files
themselves are not committed to git — they are gitignored per standard practice; the
`train → merge_and_export` commands in the notebook regenerate them.)

## Evidence files

- [`evidence/training_screenshot.png`](evidence/training_screenshot.png) — screenshot of the
  training cell mid-run (loss table printing).
- [`evidence/training_run_log.txt`](evidence/training_run_log.txt) — complete captured stdout of
  the training cell, from model load through `Training complete. Metrics: {...}` and adapter save.
- The executed notebook itself retains all cell outputs (training loss table, eval loss, merge
  completion, adapter file listing).

## Notes on the debugging path (Colab environment drift)

Getting this running on Colab's rolling-latest package set required fixing several
version-compatibility issues between Unsloth / trl / transformers / bitsandbytes. Each fix is a
separate commit in the git history; the summary:

1. `%cd $VAR` doesn't expand shell env vars in IPython magics → combined `cd` + command into
   single `!bash` lines.
2. Colab's preinstalled `torchao` 0.10.0 broke `peft`'s LoRA dispatch → `pip uninstall -y torchao`
   (we don't use it; quantization is bitsandbytes NF4).
3. Manually `--no-deps`-pinned trl/peft/bitsandbytes resolved to mutually-incompatible "latest"
   versions → switched to `pip install unsloth` and let Unsloth resolve its own compatible set.
4. **Root cause of the persistent `eos_token ('<EOS_TOKEN>')` error:** calling Unsloth's
   `get_chat_template(tokenizer, "qwen3")` on Qwen3 — which already ships a valid native chat
   template — overwrote `eos_token` with an unresolved placeholder
   ([unslothai/unsloth#2797](https://github.com/unslothai/unsloth/issues/2797)). Fix: skip
   `get_chat_template()` and use the tokenizer's native template, matching Unsloth's own official
   reference examples.

For local validation of the pipeline without a full 8B run, `training_config.smoketest.yaml`
swaps in `Qwen/Qwen3-0.6B` at seq-len 512 so the whole SFTConfig/eos_token/trainer wiring can be
exercised in minutes on a small GPU.
