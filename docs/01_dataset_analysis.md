# Dataset Analysis — "Chat Data for Assessment of Applicants"

This document is a full audit of the raw file provided at
`data/raw/chat_data_for_assessment_of_applicants.json` (115,140 bytes / 112 KB, single
JSON file, 55 top-level conversation records). All numbers below are computed by
[`src/data_prep/analyze_and_clean.py`](../src/data_prep/analyze_and_clean.py) and stored
machine-readably in [`data/processed/analysis_report.json`](../data/processed/analysis_report.json)
— nothing here is eyeballed or estimated by hand.

## 1. Dataset format

**The file is not valid JSON and not strict JSONL.** It is 55 pretty-printed JSON objects
concatenated back-to-back with no enclosing `[ ... ]` array, and with inconsistent
separators — most object boundaries are a bare `\n`, but the file is clearly the result of
pasting together at least two export batches, because one boundary run has trailing commas
(`...}]},\n\n{"messages"...`) that look like array-fragment leftovers.

Practically this means:
- `json.load(f)` fails (`json.decoder.JSONDecodeError: Extra data`).
- Reading it as JSONL (`for line in f: json.loads(line)`) also fails, because several
  objects are pretty-printed across many lines rather than minified to one line each.
- The only robust way to parse it is a streaming approach —
  `json.JSONDecoder.raw_decode` in a loop, skipping whitespace/commas between objects —
  which is what `analyze_and_clean.py` does. This is the **first cleaning step** and is a
  hard requirement before anything else can run, including `datasets.load_dataset("json", ...)`,
  which would also choke on this file as-is.

**Schema is inconsistent across records** — two variants are present:

| Schema | Count | Description |
|---|---:|---|
| `{"messages": [...]}` | 35 | Bare conversation, no metadata |
| `{"id": ..., "tags": [...], "messages": [...]}` | 20 | Has a string id and a topic-tag list |

This looks like two authoring batches were merged: an earlier batch without metadata and a
later, more deliberately red-teamed batch with `id`/`tags`. The training pipeline treats
`id`/`tags` as optional metadata and only requires `messages`, so this inconsistency does not
block training, but it does mean **topic tags are only available for 20/55 (36%) of raw
records** — see §7.

## 2. Conversation format

Every record is a single JSON object with a `messages` array of role/content pairs, i.e. the
standard OpenAI/HF chat-format:

```json
{"messages": [
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

- 100% of conversations begin with exactly one `system` message (never zero, never more than
  one).
- After the system message, every conversation strictly alternates `user → assistant → user →
  assistant → ...` with **zero exceptions** (verified programmatically, not just spot-checked).
- Conversation length distribution (number of messages, including the system turn), on the
  **50 unique** conversations after dedup:

  | Messages | Turns (user+assistant pairs) | Count |
  |---:|---:|---:|
  | 3 | 1 | 39 |
  | 5 | 2 | 10 |
  | 7 | 3 | 1 |

  Mean 3.48 messages/conversation, median 3. **78% of conversations are single-turn** (one
  question, one answer, no follow-up). Only one conversation in the entire dataset goes to a
  third turn. This is the single most important structural fact for the fine-tuning strategy
  (§ "Implications for fine-tuning" below): the raw data barely teaches the model to sustain a
  multi-turn back-and-forth consultation, which is exactly what a real astrology chat session
  looks like. This directly motivates writing long, naturalistic 20–30 message consultations by
  hand (Task 3) and generating more multi-turn synthetic data (Task 4) rather than shipping the
  raw 50 conversations alone.

## 3. Language

The corpus is **code-mixed Hindi/English**, not monolingual:

- Devanagari-script characters: 17,610
- Latin-script characters: 33,633
- Devanagari share: **34.4%** of all script characters

Of the 20 tagged conversations, the self-reported language tags split as `hinglish` (6),
`hindi` (6), `english` (3) — i.e. even the "Hindi" and "Hinglish" turns often mix in Latin
script (English loanwords, Romanized Hindi), which is why the raw Devanagari ratio (34%) is
lower than the tag counts alone would suggest. This matches the system prompt instruction found
in the most common system prompt: *"Always reply in the same language and register the user
uses (Hindi, Hinglish, or English)."* The model must therefore be strong at **script-switching
and code-mixing**, not just bilingual translation — this is a first-class requirement, not an
edge case, and is the primary reason Qwen3 is recommended over Qwen2.5 in §
[02_finetuning_strategy.md](02_finetuning_strategy.md).

## 4. Response style

Assistant turns are long-form, structured, empathetic explanations, not short chat replies:

| | min | max | mean | median |
|---|---:|---:|---:|---:|
| Assistant message tokens (cl100k proxy) | 80 | 1,119 | 391.9 | ~360 |
| User message tokens | 9 | 226 | 58.5 | ~40 |
| System prompt tokens | 23 | 371 | 122.1 | — |

Assistant replies average **~6.7x longer** than user messages. Qualitatively, assistant turns
consistently follow a recognizable structure: (1) acknowledge the question/emotion, (2) give a
grounded astrological read (ascendant/moon sign/dasha/transit framed as *tendencies*, never
certainties), (3) explicit hedging language ("astrology shows trends, not guarantees"), (4) a
practical next step or remedy, and (5) for sensitive topics, an explicit safety redirect. This
is a **deliberately safety-tuned persona**, not a generic chatbot — see §5.

## 5. Is this "just astrology Q&A"? No — it's a safety-alignment dataset for an astrology assistant

This is the most important qualitative finding and changes the fine-tuning objective. Looking at
the 20 tagged conversations' tags:

```
career(2) lottery(2) money(2) fear(2) safety(2) marriage(2) black-magic(1) death(1)
demanding-guarantee(1) scammed(1) money-waste(1) emotional-crisis(1) breakup(1)
trust-boundary(1) free-will(1) health-boundary(1) medical-concern(1) puja-false-hope(1)
accident(1) refusal(1) pregnancy(1) doctor-referral(1) financial-safety(1) skeptic(1)
conflicting-astrologers(1) trust(1) ...
```

and the per-conversation system prompts, which are **not one fixed persona prompt** but 46
distinct variants tailored per scenario (guardrail language changes based on what the user is
about to ask — death, lottery numbers, black magic, medical diagnoses, guaranteed marriage
outcomes), the dataset is clearly constructed as **red-team / boundary-setting training data**
for a product astrology assistant ("Vedaz's AI Vedic astrologer"), where the hard requirements
are:

- Never predict death, illness, or lifespan.
- Never guarantee financial outcomes or give lottery numbers.
- Never validate black-magic/curse fears without pushing back gently.
- Redirect self-harm / acute emotional crisis to professional helplines instead of doing a
  reading.
- Frame all astrological statements as tendencies/probabilities, never certainties.
- Encourage practical action (job search, doctor visit, documentation) alongside astrology.

**Consequence for Task 3/4:** the 5 manually written and 100 synthetic conversations must
preserve this safety posture, not just mimic surface style. A model fine-tuned only for
"sounds like an astrologer" while dropping these guardrails would be a regression, not an
improvement, on the actual product requirement. This is enforced in the synthetic-data
generation guidelines (`data/synthetic/GENERATION_NOTES.md`) and in the manually written
conversations (`data/manual_conversations/`).

## 6. Data quality issues found and fixed

| Issue | Finding | Fix applied |
|---|---|---|
| Invalid file format | 55 concatenated JSON objects, no array wrapper, inconsistent separators | Custom streaming parser (`raw_decode` loop) in `analyze_and_clean.py` |
| **Exact duplicate conversations** | **5 of 55 (9%)** records are byte-identical duplicates: `conv_016_career_abroad`, `conv_017_lottery_prediction`, `conv_018_black_magic`, `conv_019_career_choice`, `conv_020_death_prediction` each appear twice with identical `id`, `tags`, and `messages` | Content-hash (SHA-256 over the `messages` array) dedup, keeping first occurrence → 55 → **50 unique** |
| **ID collisions** | The id scheme `conv_016`…`conv_020` is reused across two *different* batches of conversations with completely different content (e.g. one `conv_016` is about career-abroad, another unrelated `conv_016` batch covers accident prediction) | IDs are **not** treated as unique keys anywhere in the pipeline; dedup and joins use a content hash instead |
| Malformed messages (missing role/content, non-string content, empty content) | **0 found** — message-level structure is clean | No fix needed |
| Missing assistant replies | **0 found** — every conversation has ≥1 assistant turn | No fix needed |
| Inconsistent roles / broken alternation | **0 found** — 100% of conversations strictly alternate `system → (user → assistant)+` | No fix needed |
| Schema drift | 35 records have only `messages`; 20 also have `id`+`tags` | Pipeline treats `id`/`tags` as optional; final training JSONL is normalized to `{"messages": [...]}` only so the trainer never depends on optional fields |
| Single-turn skew | 78% of conversations are 1 user/assistant turn; only 1 conversation reaches turn 3 | Addressed via Task 3 (hand-written 20–30 message multi-turn consultations) and Task 4 (synthetic data explicitly includes multi-turn samples) |
| Duplicate individual messages (not whole conversations) | Only surfaced as a byproduct of the 5 exact-duplicate conversations above; after dedup, **0 residual duplicate individual messages** | Resolved by conversation-level dedup |

## 7. Token statistics (full detail)

Computed with `cl100k_base` (tiktoken) as a **BPE proxy** — the real Qwen tokenizer was not
available in the offline analysis environment. This proxy is known to **over-tokenize
Devanagari** relative to Qwen's tokenizer (which has dedicated Indic-language BPE merges from
its 100+-language pretraining), so treat these as **upper-bound / conservative** estimates; real
Qwen3 token counts for the Hindi-script portions will be somewhat lower. Re-run
`analyze_and_clean.py` with `transformers` installed to get exact Qwen3 counts (the script
prefers the real tokenizer automatically when available — see `get_tokenizer()`).

| Metric | Value |
|---|---:|
| Total tokens, full deduped corpus | 34,028 |
| Conversation length (tokens) — min / mean / median / max | 123 / 680.6 / 474.5 / 2,031 |
| Conversation length — p90 / p95 / p99 | 1,503 / 1,672 / 2,031 |
| System prompt tokens — min / mean / max | 23 / 122.1 / 371 |
| User message tokens — min / mean / max | 9 / 58.5 / 226 |
| Assistant message tokens — min / mean / max | 80 / 391.9 / 1,119 |

**Sizing implication:** a `max_seq_length` of **2048 tokens** covers 100% of the raw dataset
with headroom (p99 is 2,031 even under the conservative proxy tokenizer). Once merged with the
longer hand-written 20–30 message consultations from Task 3 (which run considerably longer,
multi-turn), the effective training context needs to extend to **4096 tokens** — see
`training_config.yaml`.

## 8. Average conversation length — summary

- **By messages:** 3.48 messages / conversation (median 3) → **1.24 user/assistant turn-pairs**
  on average.
- **By tokens:** 680.6 tokens / conversation (median 474.5).
- **By characters:** 361.4 chars/message overall; assistant messages average 665 chars vs. 124
  chars for user messages (assistant:user length ratio ≈ 5.4:1 by characters, 6.7:1 by tokens).

## 9. Preprocessing strategy (executive summary)

1. **Parse** the concatenated-JSON raw file with a streaming decoder (`analyze_and_clean.py`) —
   never assume standard JSON/JSONL for files claimed to be "chat data" exports; always sniff
   the format first.
2. **Validate** every conversation for role-set correctness, content type, non-empty content,
   and strict `system → (user → assistant)+` alternation. (0 violations found here, but the
   check is in the pipeline permanently since it is cheap and this dataset will grow.)
3. **Deduplicate** at the conversation level using a content hash of the `messages` array (not
   the `id`, which collides). Removes 5/55 (9%) exact duplicates.
4. **Normalize schema** to `{"messages": [...]}` — `id`/`tags` are useful for analysis
   (topic-balance auditing, see §5–7) but are stripped from the final training file so the
   trainer has one consistent shape regardless of source batch.
5. **Augment** for the two structural gaps this audit found: (a) single-turn skew — add
   multi-turn data (Task 3 hand-written + Task 4 synthetic multi-turn samples); (b) small
   absolute size (50 usable real examples is far too little to fine-tune a 7–8B model on alone
   without severe overfitting risk) — add the 100 synthetic + 5 manual conversations, giving a
   final corpus of **155 conversations** before train/val split (see
   `data/processed/split_summary.json` after running `split_dataset.py`).
6. **Split** train/val at 90/10 **per source file**, not globally, so the tiny-but-precious real
   dataset is guaranteed representation in both splits (`split_dataset.py`).
7. **Tokenize** at train time with the Qwen chat template (`apply_chat_template`), not a custom
   template — the system/user/assistant structure already matches Qwen's expected format
   1:1, so no reformatting is needed beyond what `TrainingArguments`/`SFTTrainer` do
   automatically.
