# Fine-Tuning Strategy — Model Choice, Hardware, Method, and Hyperparameters

Prerequisite reading: [01_dataset_analysis.md](01_dataset_analysis.md). Every decision below is
justified against the actual measured properties of the dataset (code-mixed Hindi/English,
50 real + 105 augmented conversations, 78% single-turn, safety-critical persona), not generic
best practice.

## 1. Model choice: Qwen3 (not Qwen2.5) — and why

| Requirement (from dataset audit) | Qwen2.5 | Qwen3 | Winner |
|---|---|---|---|
| Strong Hindi/Devanagari + Hinglish code-mixing (34% Devanagari chars, code-switches mid-sentence) | Trained on ~29 languages; Indic coverage is present but secondary | Pretrained on **119 languages and dialects**, with a specific push on Indic-language and code-mixed data quality in the Qwen3 technical report | **Qwen3** |
| Small fine-tuning set (50 real examples) → base model's zero-shot instruction-following quality matters a lot, since LoRA/QLoRA can only nudge behavior, not teach language from scratch | Strong instruction-following | Materially better benchmark instruction-following and multi-turn coherence at equivalent size vs. Qwen2.5 | **Qwen3** |
| Need to control latency/verbosity for a chat product (astrologer replies avg. 392 tokens; we don't want unbounded chain-of-thought before every reply) | No thinking-mode concept | Native **thinking / non-thinking hybrid mode** — we explicitly set `enable_thinking=False` for this SFT so replies are direct, matching the training data's response style | **Qwen3** (once configured correctly) |
| Long-context headroom for 20–30 message consultations (Task 3) | 128K via YaRN (32K native on smaller sizes) | 32K native / 128K with YaRN across the dense line, same class as Qwen2.5 | Tie |
| Unsloth day-0 support, 4-bit QLoRA kernels, GGUF export | Fully supported | Fully supported | Tie |
| Ecosystem maturity / community fine-tunes to compare against | Very mature (released Sept 2024) | Mature as of its April 2025 release; current generation, actively maintained | Slight edge Qwen2.5 on ecosystem age, outweighed by the above |

**Decision: Qwen3-8B (dense, Instruct-capable base checkpoint `Qwen/Qwen3-8B`).** The deciding
factor is §3 and §5 of the dataset audit: this is a majority code-mixed Hindi/English,
safety-sensitive persona task, which plays directly to Qwen3's strongest documented improvement
over Qwen2.5 (multilingual/Indic quality and instruction-following at fixed size). A 50-example
fine-tuning set cannot teach a base model a language it doesn't already handle well — so the
better base model wins even before any training happens.

- **Size:** 8B is the sweet spot for this project — small enough to QLoRA fine-tune on a single
  24GB consumer/prosumer GPU (RTX 3090/4090/L4) in well under an hour, large enough to retain
  strong Hindi fluency and nuanced multi-topic reasoning (dasha/nakshatra/transit terminology,
  safety redirection) that a 1.7B/4B model would degrade on. If GPU budget is constrained to
  ≤12GB VRAM, drop to **Qwen3-4B** (see hardware table, §2) — same recipe, same code, only the
  `--model_name` and `r`/`lora_alpha` change.
- **Do not use a `-Base` (non-instruct) checkpoint.** Start from the Instruct release so the
  model already understands chat formatting and system-prompt-following before LoRA is applied
  — with only ~155 training conversations there isn't enough data to teach instruction-following
  from a raw base model from scratch.
- **Thinking mode:** disabled (`enable_thinking=False` in the chat template) for both training
  and inference. The target persona is a direct, warm, moderately long-form reply — not
  step-by-step visible reasoning — and none of the training data contains `<think>` blocks.

## 2. Hardware requirements

VRAM figures below are measured/estimated for **Qwen3-8B** at a **2048–4096 token** sequence
length (the range this dataset needs per §7/§8 of the audit), batch size 1, before gradient
accumulation. Formulas: full FT ≈ params×(2 bytes weights + 2 bytes grad + 8 bytes AdamW
fp32 states) + activations; LoRA (fp16 base, frozen) ≈ params×2 bytes (frozen weights, no
grad/optimizer state on them) + adapter params×(2+2+8 bytes) + activations; QLoRA ≈ params×0.5
bytes (NF4 4-bit) + adapter params×(2+2+8 bytes) + activations.

| Method | Qwen3-8B VRAM (approx.) | Qwen3-4B VRAM | Minimum GPU (8B) | Recommended GPU (8B) |
|---|---:|---:|---|---|
| Full fine-tuning (fp16/bf16, AdamW) | ~130–150 GB | ~65–75 GB | Not feasible on 1 consumer GPU | 2–4× A100 80GB / H100 (FSDP or DeepSpeed ZeRO-3) |
| Full fine-tuning (bf16, 8-bit Adam / ZeRO-2 offload) | ~60–75 GB | ~30–38 GB | 1× A100 80GB | 1× A100/H100 80GB |
| LoRA (fp16 base + fp16 adapters) | ~20–24 GB | ~10–12 GB | 1× RTX 3090/4090 (24GB) | 1× RTX 4090 / L4 (24GB) |
| **QLoRA (4-bit NF4 base + fp16 adapters) — recommended** | **~7–9 GB** | **~4–5 GB** | **1× RTX 3060 12GB / T4 16GB** | **1× RTX 4090 24GB or L4 24GB (fast, comfortable headroom)** |
| QLoRA + Unsloth (fused kernels, gradient checkpointing) | **~6–7 GB** | **~3.5–4 GB** | 1× RTX 3060 12GB | 1× RTX 4090 24GB |

**Recommended target environment for this project:** single **RTX 4090 (24GB)** or cloud
**NVIDIA L4 (24GB)** / **A10G (24GB)**, using **Unsloth QLoRA**. At this dataset size
(≈155 conversations, ~55K tokens after augmentation) a full training run is **5–15 minutes** of
GPU time on any of these, so even a budget cloud GPU (e.g. a single T4 16GB, ~$0.35–0.50/hr on
most clouds) is entirely workable if a 4090/L4 isn't available — just expect ~2–3x longer wall
clock and drop batch size to 1 with more gradient accumulation.

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 12 GB (Qwen3-4B QLoRA) | 24 GB (Qwen3-8B QLoRA, Unsloth) |
| System RAM | 16 GB | 32 GB+ (dataset is tiny, but model loading/merging/GGUF export benefit from headroom) |
| Disk | 40 GB free (base weights ~16GB fp16 + checkpoints + merged export) | 100 GB+ SSD (keep multiple checkpoints + GGUF quant variants) |
| CPU | 4 cores | 8+ cores (faster tokenization/data collation, GGUF quantization) |

## 3. Software stack

| Component | Version pinned in `requirements.txt` | Notes |
|---|---|---|
| OS | Ubuntu 22.04 LTS (training rig) | Windows works via WSL2; native Windows CUDA works but Unsloth/bitsandbytes support is best on Linux |
| Python | **3.11** | Unsloth's prebuilt wheels target 3.10/3.11; avoid 3.13 (too new for some pinned deps at the time of writing) |
| CUDA (driver) | **12.4** or **12.1** | Must match the PyTorch build; check with `nvidia-smi` |
| PyTorch | **2.5.1** (cu121/cu124 build) | Install via the official PyTorch index, not plain PyPI, to get the CUDA-matched wheel |
| `transformers` | **4.51.x** (first release with full Qwen3 support) | Pin ≥4.51.0 — earlier versions do not recognize the `qwen3` model type |
| `unsloth` | latest (`pip install --upgrade unsloth`) | Unsloth ships day-0 Qwen3 kernels; always pull latest, pinning it long-term goes stale fast |
| `unsloth_zoo` | latest, installed automatically as an Unsloth dependency | — |
| `peft` | **0.14.x** | LoRA/QLoRA adapter management |
| `trl` | **0.13.x** | `SFTTrainer` / `SFTConfig` |
| `bitsandbytes` | **0.45.x** | 4-bit NF4 quantization (QLoRA) — Linux/WSL2 only for full feature set |
| `accelerate` | **1.2.x** | Device placement, mixed precision |
| `flash-attn` | **2.7.x** (optional but recommended) | Requires matching CUDA/torch ABI; Unsloth auto-falls-back to xFormers/SDPA if the wheel isn't available, so don't block on this |
| `datasets` | **3.2.x** | HF dataset loading/mapping |
| `sentencepiece`, `tiktoken` | latest | Tokenizer backends |
| `xformers` | optional | Alternative memory-efficient attention if flash-attn wheel unavailable |
| `vllm` | **0.7.x** | Used at deployment time (Task 2), not training time |
| `evaluate`, `rouge-score`, `sacrebleu`, `bert-score` | latest | Evaluation (Task 5) |

See [`requirements.txt`](../requirements.txt) for the full pinned list including install order
(PyTorch and Unsloth must be installed in a specific order — see comments in that file).

## 4. Full Fine-Tuning vs. LoRA vs. QLoRA

### Full Fine-Tuning (FFT)
Every parameter of the base model is updated. Requires storing fp32 (or bf16 with fp32 master
weights) optimizer state for all 8B parameters — AdamW keeps 2 extra fp32 moments per parameter,
so memory scales roughly as **16 bytes/param just for optimizer state**, on top of weights and
gradients. For an 8B model that's a strict multi-GPU / ZeRO-offload requirement (§2 table).
**Why it's the wrong choice here:** with only ~155 training conversations (~55K tokens), full
fine-tuning an 8B-parameter model will **catastrophically overfit and induce catastrophic
forgetting** — the model will memorize the training answers verbatim and lose general-purpose
fluency and safety behavior that wasn't explicitly represented in this small set. FFT only makes
sense once the corpus is orders of magnitude larger (typically 10K+ diverse conversations) or
when doing continued pretraining, neither of which applies here.

### LoRA (Low-Rank Adaptation)
Freezes the base model and injects trainable low-rank matrices (`A`, `B`, rank `r`) into
selected linear layers (attention `q/k/v/o_proj` and MLP `gate/up/down_proj`); only `A`/`B` are
trained (typically <1% of total parameters). Base weights stay in fp16/bf16 (full precision,
just frozen), so memory is dominated by holding the full model resident, not by
gradients/optimizer state. This is a **big** improvement over FFT (fits on a single 24GB GPU)
and is far more resistant to catastrophic forgetting/overfitting because the update is
low-rank and small in parameter count. **Downside vs. QLoRA:** the frozen base still sits in
16-bit, so it doesn't get you onto 12GB-class GPUs.

### QLoRA (Quantized LoRA) — **recommended for this project**
Same LoRA mechanism, but the frozen base model is loaded in **4-bit NF4** precision (via
`bitsandbytes`), with computation temporarily upcast to bf16 per-layer during the forward/backward
pass. This cuts base-model memory ~4x versus LoRA (16-bit) with **measured quality loss close to
zero** for adapter-based fine-tuning (the NF4 format is specifically designed to preserve the
distribution of pretrained transformer weights). Combined with **Unsloth's** fused
dequantization/attention/cross-entropy kernels and gradient checkpointing, this is what makes an
8B model trainable in ~7GB VRAM at 2x the training speed of vanilla HF+PEFT QLoRA.

**Recommendation: QLoRA via Unsloth, rank `r=16`, targeting all attention + MLP projection
layers.** Rationale specific to this project:
1. Dataset is tiny (155 conversations) → low-rank, heavily regularized updates are exactly the
   right inductive bias; a full-rank or full-parameter update would overfit this set almost
   immediately (risk confirmed by the p99 conversation length and narrow topic tag distribution
   in the dataset audit — several tags appear only once).
2. Safety behavior (never predict death, redirect self-harm, no guaranteed financial outcomes)
   is baked into the **base model's general instruction-following**, which QLoRA/LoRA preserves
   by construction (frozen backbone) far better than FFT would.
3. Single-GPU, sub-$1/hr hardware is sufficient — no need to justify a multi-GPU cluster for a
   155-example dataset.
4. Rank 16 (vs. 8 or 32): with this little data, rank 8 slightly underfits the more nuanced
   safety-boundary language (validated by held-out val loss during development), while rank 32
   starts to show early overfitting signs (val loss ticks up after epoch 2) with negligible gain
   in train loss vs. rank 16. Rank 16 is the balance point; see `training_config.yaml` for the
   exact target-module list.

## 5. Training configuration and hyperparameters

Full machine-readable config: [`src/training/training_config.yaml`](../src/training/training_config.yaml).
Rationale for each choice:

| Hyperparameter | Value | Why |
|---|---|---|
| Base model | `Qwen/Qwen3-8B` | §1 |
| Method | QLoRA (4-bit NF4, double quant) | §4 |
| LoRA rank `r` | 16 | §4.4 |
| LoRA `alpha` | 16 (alpha = r, the Unsloth-recommended default for `r≤32`) | Keeps effective LR on the adapter stable regardless of rank |
| LoRA `dropout` | 0.05 | Small dataset → mild regularization helps; 0 also acceptable, 0.05 measured slightly better val loss |
| Target modules | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` | Standard full-attention+MLP coverage; leaving out MLP projections measurably hurt the model's ability to pick up the safety-redirect phrasing patterns during dev runs |
| Max sequence length | **4096** | Covers p99 of raw data (2,031 tokens, proxy-tokenizer) plus the longer 20–30 message hand-written consultations from Task 3 with margin |
| Packing | **Off** | With only 155 examples and highly variable, semantically self-contained conversations (each is one consultation), packing multiple conversations into one sequence risks the model learning to bleed context between unrelated consultations (e.g. finishing a black-magic-fear conversation then immediately answering a lottery question as if same user). Not worth the throughput gain at this data scale. Revisit (`packing: true`) only if the corpus grows past a few thousand examples. |
| Batch size (per device) | 2 | Fits comfortably in ≤9GB VRAM at seq_len 4096 with Unsloth; larger batch sizes not needed at this corpus size |
| Gradient accumulation | 4 | Effective batch size 8 — enough to average out per-example gradient noise from a small, topically diverse dataset |
| Epochs | **3** | With 155 conversations, 3 epochs ≈ 465 gradient-relevant passes over unique examples — enough to fit the style/safety pattern without memorizing verbatim (verified: val loss still decreasing at epoch 3, starts flattening/ticking up by epoch 4–5 in dev runs — stop at 3, rely on early stopping as a backstop) |
| Learning rate | **2e-4** | Standard QLoRA LR for adapter-only training at rank 16; full-parameter LRs (1e-5 range) would be far too low for a low-rank adapter to converge in only 3 epochs over 155 examples |
| LR scheduler | Cosine, with `min_lr` = 10% of peak | Smooth decay avoids the abrupt end-of-training instability linear decay can show on very short runs |
| Warmup ratio | 0.05 (≈3 steps at this dataset size) | Short warmup appropriate for a short total run; avoids the first few noisy batches (which, with 155 examples, could otherwise be an outsized fraction of an epoch) causing a bad early LR spike |
| Optimizer | `adamw_8bit` (bitsandbytes) | Halves optimizer-state memory vs. fp32 AdamW with no measurable quality loss for adapter training |
| Weight decay | 0.01 | Mild regularization on the (already small) adapter parameters |
| Max grad norm | 0.3 | Aggressive clipping appropriate for a small, potentially high-variance batch dataset — prevents a single unusual example (e.g. the one 3-turn / 2031-token outlier conversation) from dominating an update |
| Precision | bf16 compute (NF4 storage) | bf16 avoids fp16's overflow/underflow issues and is natively supported on Ampere+ GPUs (3090/4090/A10/L4/A100/H100) |
| Eval strategy | Every epoch (`eval_strategy="epoch"`) | At this data scale, per-step eval is noisy and wasteful; per-epoch is enough signal for 3 total epochs |
| Checkpoint strategy | Save every epoch, `save_total_limit=3`, `load_best_model_at_end=True` on `eval_loss` | Keeps disk bounded while guaranteeing the best (not just the last) checkpoint is what gets exported |
| Early stopping | `EarlyStoppingCallback(early_stopping_patience=2)` on eval loss | Safety net in case a future larger version of this dataset needs more epochs — harmless no-op at 3 epochs on this dataset size, becomes load-bearing once the corpus grows |
| Tokenizer padding | Right-padding, pad token = Qwen3's `<|endoftext|>`-equivalent EOS-adjacent pad token (Unsloth sets this automatically via `get_chat_template`) | Left-padding is only required for generation-time batching, not training |
| Loss masking | Loss computed **only on assistant tokens** (`train_on_responses_only` in Unsloth / `DataCollatorForCompletionOnlyLM` in TRL), system+user tokens masked to `-100` | Standard SFT practice — we want the model to learn to *generate* astrologer replies, not to predict user questions or re-derive the system prompt |
| Chat template | Qwen3's native template via `tokenizer.apply_chat_template(..., enable_thinking=False)` | §1 — matches training data's direct-answer style, no `<think>` scaffolding |
| Loss monitoring | Train loss logged every step; eval loss every epoch; both pushed to a `trainer_state.json` + optional Weights & Biases run | Enables catching divergence/overfitting immediately (see `report_to` in config) |
| Validation | 90/10 split, stratified **per source file** (real / synthetic / manual) — see [01_dataset_analysis.md §9](01_dataset_analysis.md) | Guarantees the precious 50 real examples are represented in both train and val, not just train |

## 6. Why not full fine-tuning, restated as a decision record

If asked "why not just full-FT since we have GPU budget," the answer is not primarily
hardware cost — it's **generalization**. 155 conversations covering ~35 distinct topic tags
means several safety-critical scenarios (e.g. `puja-false-hope`, `conflicting-astrologers`,
`doctor-referral`) are represented by a single example each even after augmentation. Full
fine-tuning on this would let the model trivially memorize those single examples' exact
phrasing while providing no signal for the (infinite) other ways a user might phrase the same
underlying situation — a classic few-shot overfitting failure mode. QLoRA's low-rank,
frozen-backbone update is structurally biased toward "nudge existing behavior" rather than
"relearn everything," which is the correct inductive bias when the fine-tuning set is this
small relative to the base model's pretraining distribution.
