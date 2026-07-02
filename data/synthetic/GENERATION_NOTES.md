# Synthetic Data Generation Notes (Task 4)

`synthetic_astrology_100.jsonl` contains 100 independently written conversations, generated from
[`_build_synthetic_conversations.py`](_build_synthetic_conversations.py) (the source of truth —
re-run it to regenerate the JSONL deterministically).

## Design goals

1. **Balanced topic distribution.** 20 topics x 5 conversations each = 100, an explicit, even
   split rather than an organic/random distribution. Topics were chosen to (a) cover every item
   in the assignment's required topic list (career confusion, love marriage, arranged marriage,
   business loss, government job, abroad studies, relationship issues, health anxiety, financial
   stress, divorce, second marriage, late marriage, startup success, promotion, career switch,
   pregnancy) and (b) reflect the safety-boundary categories found in the real dataset audit
   (see `docs/01_dataset_analysis.md` section 5) — lottery/guaranteed-outcome refusal, black
   magic fear, health-anxiety/medical-boundary, and skeptic/trust scenarios — so the augmented
   corpus reinforces the product's safety posture instead of diluting it.
2. **No duplicated conversations.** Every entry is independently authored (not template +
   name-substitution), and the build script verifies zero duplicates via SHA-256 content hashing
   of the full `messages` array — the same technique used to find the 5 real duplicate
   conversations during dataset cleaning. The script prints a duplicate warning and drops any
   collision automatically if one is ever introduced by a future edit.
3. **Natural wording.** Each of the 5 conversations within a topic bucket varies sentence
   structure, opening style, and specific astrological terms used (different houses/planets/Dasha
   references), not just swapped names — the goal was to avoid the "obviously templated" feel a
   naive slot-filling generator produces.
4. **Language mix.** Within each 5-conversation topic bucket, at least Hindi, Hinglish, and
   English are represented, consistent with the ~34% Devanagari / 66% Latin script mix measured
   in the real dataset (`docs/01_dataset_analysis.md` section 3).
5. **Specific future prediction windows**, not vague statements — e.g. "September 2026 –
   January 2027", "Q3 2027", "around Diwali 2026" — mirroring the assignment's explicit
   instruction and the real dataset's pattern of concrete Dasha/transit-based timing.
6. **Schema match.** Every record is exactly `{"id", "tags", "messages": [system, user,
   assistant]}` — all 100 are single-turn (3 messages), matching both the assignment's literal
   example schema and the dominant pattern in the real dataset (78% of real conversations are
   single-turn, per `docs/01_dataset_analysis.md` section 2).

## Topic buckets (5 conversations each)

`career_confusion`, `government_job`, `promotion`, `career_switch`, `love_marriage`,
`arranged_marriage`, `late_marriage`, `business_loss`, `startup_success`, `financial_stress`,
`lottery_refusal` (safety), `abroad`, `relationship`, `health_anxiety` (safety), `pregnancy`,
`divorce`, `second_marriage`, `black_magic` (safety), `family_property`, `skeptic` (safety/trust).

## Validation performed

- Build script asserts zero duplicate conversations by content hash (0 found, confirmed at
  generation time — see script output).
- `system → user → assistant` role order and non-empty string content validated for all 100
  records using the same alternation check as `analyze_and_clean.py`.
- All 100 conversations parsed successfully as strict JSONL (one JSON object per line — unlike
  the raw real dataset, this file does NOT need the custom streaming parser).

## How this feeds the training pipeline

`synthetic_astrology_100.jsonl` is one of the three inputs to
[`src/data_prep/split_dataset.py`](../../src/data_prep/split_dataset.py), which merges it with
the cleaned real data (`data/processed/clean_dataset.jsonl`) and the 5 manual consultations
(`data/manual_conversations/manual_conversations.jsonl`) into the final `train.jsonl`/`val.jsonl`
used by `src/training/train_unsloth.py`.
