"""
Parses the raw applicant-assessment chat export, validates every conversation,
deduplicates, and writes a clean JSONL file plus a machine-readable analysis report.

The raw file is NOT valid JSON and is NOT strict JSONL: it is a sequence of
55 pretty-printed JSON objects concatenated back-to-back, separated inconsistently
by bare newlines in some places and by "," in others, with no enclosing array.
A standard `json.load()` or `for line in f: json.loads(line)` will both fail on it.
We parse it with `json.JSONDecoder.raw_decode` in a loop, which is robust to both
separator styles and to objects that span multiple lines.

Usage:
    python analyze_and_clean.py \
        --input ../../data/raw/chat_data_for_assessment_of_applicants.json \
        --out-jsonl ../../data/processed/clean_dataset.jsonl \
        --out-report ../../data/processed/analysis_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path

VALID_ROLES = {"system", "user", "assistant"}


def parse_concatenated_json(text: str) -> list[dict]:
    """Split a string containing back-to-back JSON objects (no array wrapper,
    inconsistent/absent separators) into a list of Python objects."""
    decoder = json.JSONDecoder()
    objects = []
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n:
            break
        obj, end = decoder.raw_decode(text, i)
        objects.append(obj)
        i = end
    return objects


def get_tokenizer():
    """Prefer the real Qwen tokenizer if transformers+network are available;
    fall back to a BPE proxy (tiktoken cl100k_base); fall back to a word-count
    heuristic (tokens ~= words * 1.3) if neither is installed. The proxy tends
    to OVER-count Devanagari script relative to Qwen's tokenizer, which has
    dedicated Indic-language merges, so treat proxy counts as an upper bound."""
    try:
        from transformers import AutoTokenizer  # type: ignore

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        return lambda s: len(tok.encode(s)), "Qwen/Qwen3-8B (exact)"
    except Exception:
        pass
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")
        return lambda s: len(enc.encode(s)), "cl100k_base (BPE proxy, upper bound for Hindi)"
    except Exception:
        pass
    return lambda s: max(1, int(len(s.split()) * 1.3)), "whitespace heuristic (rough)"


def script_ratio(text: str) -> tuple[int, int]:
    devanagari = len(re.findall(r"[ऀ-ॿ]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return devanagari, latin


def conv_hash(messages: list[dict]) -> str:
    return hashlib.sha256(
        json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def validate_conversation(idx: int, obj: dict) -> list[str]:
    """Return a list of validation problems (empty list == fully valid)."""
    problems = []
    msgs = obj.get("messages")
    if not isinstance(msgs, list) or len(msgs) == 0:
        return [f"conv[{idx}]: 'messages' missing/empty/not-a-list"]

    roles_seen = []
    for j, m in enumerate(msgs):
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            problems.append(f"conv[{idx}].messages[{j}]: missing role/content keys")
            continue
        if m["role"] not in VALID_ROLES:
            problems.append(f"conv[{idx}].messages[{j}]: unknown role '{m['role']}'")
        if not isinstance(m["content"], str):
            problems.append(f"conv[{idx}].messages[{j}]: content is not a string")
        elif m["content"].strip() == "":
            problems.append(f"conv[{idx}].messages[{j}]: empty content")
        roles_seen.append(m["role"])

    if roles_seen.count("assistant") == 0:
        problems.append(f"conv[{idx}]: no assistant reply present")
    if roles_seen.count("system") > 1:
        problems.append(f"conv[{idx}]: multiple system messages")

    body = [r for r in roles_seen if r != "system"]
    expected = "user"
    for r in body:
        if r != expected:
            problems.append(f"conv[{idx}]: role sequence breaks strict user/assistant alternation")
            break
        expected = "assistant" if expected == "user" else "user"

    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-report", required=True)
    args = ap.parse_args()

    raw_text = Path(args.input).read_text(encoding="utf-8")
    objects = parse_concatenated_json(raw_text)

    all_problems = []
    for idx, obj in enumerate(objects):
        all_problems.extend(validate_conversation(idx, obj))

    seen_hashes: dict[str, int] = {}
    unique_objects = []
    exact_duplicate_of = {}
    for idx, obj in enumerate(objects):
        h = conv_hash(obj.get("messages", []))
        if h in seen_hashes:
            exact_duplicate_of[idx] = seen_hashes[h]
            continue
        seen_hashes[h] = idx
        unique_objects.append(obj)

    token_fn, tokenizer_name = get_tokenizer()

    role_counts = Counter()
    conv_len_dist = Counter()
    conv_token_totals = []
    assistant_tok_lens, user_tok_lens, system_tok_lens = [], [], []
    dev_chars = lat_chars = 0
    tag_counter = Counter()
    schema_variants = Counter()
    msg_dupes = Counter()

    for obj in unique_objects:
        schema_variants[tuple(sorted(obj.keys()))] += 1
        msgs = obj["messages"]
        conv_len_dist[len(msgs)] += 1
        conv_tok = 0
        for m in msgs:
            role_counts[m.get("role", "?")] += 1
            content = m.get("content", "")
            if not isinstance(content, str):
                continue
            t = token_fn(content)
            conv_tok += t
            if m["role"] == "assistant":
                assistant_tok_lens.append(t)
            elif m["role"] == "user":
                user_tok_lens.append(t)
            elif m["role"] == "system":
                system_tok_lens.append(t)
            d, l = script_ratio(content)
            dev_chars += d
            lat_chars += l
            if m["role"] in ("user", "assistant"):
                msg_dupes[(m["role"], content.strip())] += 1
        conv_token_totals.append(conv_tok)
        for t in obj.get("tags", []):
            tag_counter[t] += 1

    def pct(sorted_vals, p):
        if not sorted_vals:
            return 0
        idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * p))
        return sorted_vals[idx]

    srt = sorted(conv_token_totals)

    report = {
        "tokenizer_used_for_stats": tokenizer_name,
        "raw_object_count": len(objects),
        "unique_conversation_count": len(unique_objects),
        "exact_duplicate_conversations_removed": len(exact_duplicate_of),
        "duplicate_pairs": [
            {"kept_index": kept, "dropped_index": dropped}
            for dropped, kept in exact_duplicate_of.items()
        ],
        "schema_variants": {str(k): v for k, v in schema_variants.items()},
        "validation_problems": all_problems,
        "role_message_counts": dict(role_counts),
        "conversation_length_distribution_num_messages": dict(conv_len_dist),
        "conversation_length_messages_stats": {
            "min": min(conv_len_dist.elements()),
            "max": max(conv_len_dist.elements()),
            "mean": round(statistics.mean(list(conv_len_dist.elements())), 2),
            "median": statistics.median(list(conv_len_dist.elements())),
        },
        "token_stats": {
            "total_tokens": sum(conv_token_totals),
            "conversation_tokens": {
                "min": min(conv_token_totals),
                "max": max(conv_token_totals),
                "mean": round(statistics.mean(conv_token_totals), 1),
                "median": statistics.median(conv_token_totals),
                "p90": pct(srt, 0.90),
                "p95": pct(srt, 0.95),
                "p99": pct(srt, 0.99),
            },
            "system_prompt_tokens": {
                "min": min(system_tok_lens), "max": max(system_tok_lens),
                "mean": round(statistics.mean(system_tok_lens), 1),
            },
            "user_message_tokens": {
                "min": min(user_tok_lens), "max": max(user_tok_lens),
                "mean": round(statistics.mean(user_tok_lens), 1),
            },
            "assistant_message_tokens": {
                "min": min(assistant_tok_lens), "max": max(assistant_tok_lens),
                "mean": round(statistics.mean(assistant_tok_lens), 1),
            },
        },
        "script_mix": {
            "devanagari_chars": dev_chars,
            "latin_chars": lat_chars,
            "devanagari_ratio": round(dev_chars / (dev_chars + lat_chars + 1e-9), 3),
        },
        "tag_distribution": dict(tag_counter.most_common()),
        "duplicate_individual_messages": {
            f"{role}::{content[:80]}": cnt
            for (role, content), cnt in msg_dupes.items() if cnt > 1
        },
    }

    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for obj in unique_objects:
            clean = {"messages": obj["messages"]}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    print(f"Parsed {len(objects)} raw objects -> {len(unique_objects)} unique conversations")
    print(f"Removed {len(exact_duplicate_of)} exact duplicate conversations")
    print(f"Validation problems found: {len(all_problems)}")
    print(f"Wrote clean JSONL: {args.out_jsonl}")
    print(f"Wrote analysis report: {args.out_report}")


if __name__ == "__main__":
    main()
