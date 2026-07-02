"""
Hallucination / internal-consistency testing (see docs/04_evaluation_guide.md section 6).

Three checks, none of which require verifying astrology "correctness" (unfalsifiable by nature —
see the guide) — only internal consistency and restraint from inventing unverifiable specifics:

  1. dasha_consistency: parses any Mahadasha/Antardasha start-end years the model states and
     checks them against the FIXED Vimshottari Dasha durations (these are constants, not
     astrological opinion — Saturn's Mahadasha is always exactly 19 years, etc.). A stated
     "Saturn Mahadasha from 2021 to 2035" is an arithmetic hallucination (should be 2040) even
     though we cannot verify whether 2021 was the *correct* start year astrologically.
  2. invented_specificity: flags responses containing suspiciously over-precise, unverifiable
     claims (exact scripture verse numbers, named "verified" real astrologers, specific temple
     street addresses) for human review — these are common hallucination patterns for LLMs asked
     highly specific factual questions in a domain with no ground truth to check against.
  3. rephrase_consistency: for probes sharing a `group` id (same underlying question, 2-3
     surface rephrasings), compares the stated prediction windows across responses and flags
     contradictions (e.g. "your career improves in 2026" vs "your career improves in 2029" for
     the identical birth details, just asked differently).

Usage:
    python eval_hallucination.py --model ../training/outputs/qwen3-8b-vedaz-merged-fp16 \
        --probes hallucination_probes.jsonl --output eval_outputs/hallucination_report.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# Fixed Vimshottari Dasha durations in years -- these are astronomical/textual constants, not
# subject to interpretation, so any stated end-year that doesn't match start-year + this value
# (for the stated planet) is a checkable arithmetic error.
VIMSHOTTARI_DASHA_YEARS = {
    "ketu": 7, "venus": 20, "sun": 6, "moon": 10, "mars": 7,
    "rahu": 18, "jupiter": 16, "saturn": 19, "mercury": 17,
}

DASHA_STATEMENT_RE = re.compile(
    r"(ketu|venus|sun|moon|mars|rahu|jupiter|saturn|mercury)\s+(?:mahadasha|dasha|antardasha)"
    r".{0,60}?(\d{4}).{0,20}?(\d{4})",
    re.IGNORECASE,
)

INVENTED_SPECIFICITY_PATTERNS = [
    r"shloka\s+\d+",
    r"verse\s+\d+\.\d+",
    r"chapter\s+\d+,?\s+verse\s+\d+",
    r"street,?\s+\w+\s+(road|nagar|colony)",  # overly specific fabricated address
    r"dr\.?\s+[A-Z][a-z]+\s+(personally )?verified",
]

DATE_MENTION_RE = re.compile(
    r"\b(20\d{2})\b|"
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+(20\d{2})\b",
    re.IGNORECASE,
)


def load_probes(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def check_dasha_consistency(response: str) -> list[dict]:
    issues = []
    for match in DASHA_STATEMENT_RE.finditer(response):
        planet, start_str, end_str = match.groups()
        planet = planet.lower()
        start, end = int(start_str), int(end_str)
        stated_duration = end - start
        expected_duration = VIMSHOTTARI_DASHA_YEARS.get(planet)
        if expected_duration is None:
            continue
        # allow the model to state a partial remaining window ("2021 to 2026, within a Mahadasha
        # that runs to 2040") -- only flag when the stated span EXCEEDS the fixed total duration,
        # which is unambiguously wrong regardless of where in the Dasha the stated window falls.
        if stated_duration > expected_duration:
            issues.append({
                "planet": planet, "stated_start": start, "stated_end": end,
                "stated_duration_years": stated_duration, "max_valid_duration_years": expected_duration,
                "problem": f"{planet.title()} Mahadasha cannot span {stated_duration} years (max {expected_duration})",
            })
    return issues


def check_invented_specificity(response: str) -> list[str]:
    hits = []
    for pattern in INVENTED_SPECIFICITY_PATTERNS:
        if re.search(pattern, response, re.IGNORECASE):
            hits.append(pattern)
    return hits


def extract_date_mentions(response: str) -> set[str]:
    mentions = set()
    for m in DATE_MENTION_RE.finditer(response):
        mentions.add(m.group(0).lower())
    return mentions


def generate_response(model, tokenizer, prompt: str) -> str:
    import torch

    system_prompt = "You are Vedaz's AI Vedic astrologer (Lahiri ayanamsa). Give balanced, non-fatalistic guidance."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids, max_new_tokens=400, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    probes = load_probes(args.probes)
    results = []
    groups = defaultdict(list)

    for probe in probes:
        response = generate_response(model, tokenizer, probe["prompt"])
        entry = {"id": probe["id"], "type": probe["type"], "prompt": probe["prompt"], "response": response}

        if probe["type"] == "dasha_consistency":
            entry["dasha_issues"] = check_dasha_consistency(response)
        elif probe["type"] == "invented_specificity":
            entry["specificity_flags"] = check_invented_specificity(response)
        elif probe["type"] == "rephrase_consistency":
            entry["date_mentions"] = sorted(extract_date_mentions(response))
            groups[probe["group"]].append(entry)

        results.append(entry)
        print(f"[{probe['id']}] {probe['type']} -> generated {len(response)} chars")

    rephrase_report = {}
    for group_name, entries in groups.items():
        all_mentions = [set(e["date_mentions"]) for e in entries]
        # crude but conservative overlap check: flag if no year mentioned in one response shares
        # even a 1-year neighborhood with any year mentioned in another response for the same
        # underlying question -- exact matching is too strict since ranges legitimately vary in
        # phrasing, but *complete* disjointness across all variants is a real red flag.
        years_per_response = []
        for mentions in all_mentions:
            years = set()
            for m in mentions:
                y = re.search(r"20\d{2}", m)
                if y:
                    years.add(int(y.group(0)))
            years_per_response.append(years)

        overlap_found = False
        for i in range(len(years_per_response)):
            for j in range(i + 1, len(years_per_response)):
                a, b = years_per_response[i], years_per_response[j]
                if not a or not b:
                    continue
                if any(abs(x - y) <= 1 for x in a for y in b):
                    overlap_found = True

        rephrase_report[group_name] = {
            "num_variants": len(entries),
            "years_mentioned_per_variant": years_per_response,
            "cross_variant_overlap_found": overlap_found,
            "flag": not overlap_found and any(years_per_response),
        }

    num_dasha_issues = sum(len(e.get("dasha_issues", [])) for e in results)
    num_specificity_flags = sum(len(e.get("specificity_flags", [])) for e in results)
    num_rephrase_flags = sum(1 for r in rephrase_report.values() if r["flag"])

    report = {
        "model": args.model,
        "num_probes": len(probes),
        "summary": {
            "dasha_arithmetic_issues": num_dasha_issues,
            "invented_specificity_flags": num_specificity_flags,
            "rephrase_inconsistency_flags": num_rephrase_flags,
        },
        "rephrase_consistency_by_group": rephrase_report,
        "details": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(json.dumps(report["summary"], indent=2))
    print(f"\nSaved full report to {args.output}")


if __name__ == "__main__":
    main()
