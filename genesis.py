"""Simic Stage 1 — genesis: single-command agent memory pipeline.

Takes a hand-cleaned interview transcript and produces a complete agent memory:
  - System prompt (summary cache)
  - Expert behavioral observations (4 experts in parallel)
  - Coverage gaps and cross-expert conflicts
  - Full transcript preserved as reference

Usage:
    python genesis.py transcript.md --agent-id agent_1
    python genesis.py transcript.md --agent-id agent_1 --skip-gaps
    python genesis.py transcript.md --agent-id agent_1 --skip-cache
    python genesis.py transcript.md --agent-id agent_1 --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import yaml
from anthropic import Anthropic
from dotenv import load_dotenv

from expert_prompts import (
    COVERAGE_GAPS_PROMPT,
    EXPERTS,
    SUMMARY_CACHE_PROMPT,
    WRAPPER_PROMPT,
)

SONNET = "claude-sonnet-4-5"
AGENT_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def validate_agent_id(agent_id: str) -> None:
    if not AGENT_ID_RE.match(agent_id):
        print(
            f"Error: agent_id '{agent_id}' is invalid. "
            f"Allowed characters: letters, digits, underscore, hyphen.",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Demographic extraction
# ---------------------------------------------------------------------------

FIELD_MAP = {
    "full name": "name",
    "current company": "company",
    "currently live": "city",
    "hometown": "hometown",
    "how old": "age",
    "gender": "gender",
    "marital": "marital_status",
    "relationship status": "marital_status",
    "lives with you": "living_with",
    "income": "income_range",
    "occupation": "occupation",
    "religion": "religion",
    "languages": "languages",
}


def parse_demographics(transcript_text: str) -> dict:
    """Extract structured demographic fields from the intake section of a transcript.

    Handles formats:
      - Question? Answer
      - Label: Answer
      - Question? - Answer  (dash-separated variant)
    """
    match = re.search(r"^#\s*Demographic intake\s*:?\s*$", transcript_text, re.IGNORECASE | re.MULTILINE)
    if not match:
        return {}

    start = match.end()
    end_match = re.search(r"^(?:#|\s*Q\d+[\.\s])", transcript_text[start:], re.MULTILINE)
    section = transcript_text[start:start + end_match.start()] if end_match else transcript_text[start:]

    demographics = {}
    for line in section.splitlines():
        line = line.strip()
        if not line:
            continue

        for sep in ["?", ":"]:
            if sep in line:
                label, _, value = line.partition(sep)
                value = value.strip()
                if value.startswith("- ") or value.startswith("-"):
                    value = value.lstrip("-").strip()
                if not value:
                    continue
                label_lower = label.lower().strip()
                for key, field in FIELD_MAP.items():
                    if key in label_lower:
                        demographics[field] = value
                        break
                break

    if "age" in demographics:
        try:
            demographics["age"] = int(demographics["age"])
        except ValueError:
            pass

    return demographics


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

def parse_transcript(path: Path) -> str:
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        print(f"Error: file is empty: {path}", file=sys.stderr)
        sys.exit(1)
    return text.strip()


# ---------------------------------------------------------------------------
# Expert execution
# ---------------------------------------------------------------------------

def build_system_prompt(expert_prompt: str) -> str:
    return WRAPPER_PROMPT + "\n\n" + expert_prompt


def run_expert(client: Anthropic, name: str, system_prompt: str, body: str) -> dict:
    response = client.messages.create(
        model=SONNET,
        max_tokens=4000,
        system=system_prompt,
        messages=[{"role": "user", "content": body}],
    )
    return {
        "name": name,
        "text": response.content[0].text.strip(),
        "tokens_in": response.usage.input_tokens,
        "tokens_out": response.usage.output_tokens,
    }


def run_all_experts(client: Anthropic, body: str) -> list:
    expert_list = list(EXPERTS.items())
    results = [None] * len(expert_list)

    print(f"  Running {len(expert_list)} experts in parallel...")
    start = time.time()
    total_in = total_out = 0

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for i, (name, prompt) in enumerate(expert_list):
            system_prompt = build_system_prompt(prompt)
            future = executor.submit(run_expert, client, name, system_prompt, body)
            futures[future] = i

        for done_count, future in enumerate(as_completed(futures), 1):
            idx = futures[future]
            try:
                result = future.result()
            except Exception as e:
                name = expert_list[idx][0]
                print(f"    WARNING: {name} failed: {e}", file=sys.stderr)
                result = {"name": name, "text": "", "tokens_in": 0, "tokens_out": 0}
            results[idx] = result
            total_in += result["tokens_in"]
            total_out += result["tokens_out"]
            print(f"    [{done_count}/{len(expert_list)}] {result['name']}", flush=True)

    elapsed = time.time() - start
    print(f"  Experts done in {elapsed:.0f}s ({total_in:,} in / {total_out:,} out)")
    return results


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_analysis_input(body: str, expert_results: list) -> str:
    obs = "\n\n---\n\n".join(
        f"## {r['name']}\n\n{r['text']}" for r in expert_results if r["text"]
    )
    return f"TRANSCRIPT:\n\n{body}\n\n---\n\nEXPERT OBSERVATIONS:\n\n{obs}"


def run_coverage_gaps(client: Anthropic, body: str, expert_results: list) -> dict:
    print("  Running coverage gaps analysis...")
    start = time.time()
    content = _build_analysis_input(body, expert_results)

    response = client.messages.create(
        model=SONNET,
        max_tokens=2000,
        system=COVERAGE_GAPS_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    elapsed = time.time() - start
    print(f"  Gaps done in {elapsed:.0f}s ({response.usage.input_tokens:,} in / {response.usage.output_tokens:,} out)")

    return {
        "text": response.content[0].text.strip(),
        "tokens_in": response.usage.input_tokens,
        "tokens_out": response.usage.output_tokens,
    }


def generate_summary_cache(client: Anthropic, body: str, expert_results: list) -> dict:
    print("  Generating summary cache (system prompt)...")
    start = time.time()
    content = _build_analysis_input(body, expert_results)

    response = client.messages.create(
        model=SONNET,
        max_tokens=3000,
        system=SUMMARY_CACHE_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    elapsed = time.time() - start
    print(f"  Cache done in {elapsed:.0f}s ({response.usage.input_tokens:,} in / {response.usage.output_tokens:,} out)")

    return {
        "text": response.content[0].text.strip(),
        "tokens_in": response.usage.input_tokens,
        "tokens_out": response.usage.output_tokens,
    }


def validate_cache(text: str) -> list:
    warnings = []
    if not text.startswith("You are"):
        warnings.append("Summary cache does not start with 'You are'")
    rules = re.findall(r"^\d+\.\s+[A-Z]", text, re.MULTILINE)
    if len(rules) < 5:
        warnings.append(f"Only {len(rules)} behavioral rules found (expected 8-12)")
    if "[Participant]" in text or "[Company]" in text:
        warnings.append("Anonymization tokens leaked into summary cache")
    return warnings


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------

def count_observations(text: str) -> int:
    return len(re.findall(r"^\d+\.\s", text, re.MULTILINE))


def assemble_memory(agent_id: str, body: str, expert_results: list,
                    gaps_result: dict | None, cache_result: dict | None,
                    today: str, demographics: dict | None = None) -> str:
    """Build the memory/ file — the complete agent memory.

    Ordering: YAML frontmatter → system prompt → transcript → expert notes → gaps.
    """
    parts = []

    if demographics:
        parts.append("---")
        parts.append(yaml.dump(demographics, default_flow_style=False, allow_unicode=True, sort_keys=False).rstrip())
        parts.append("---\n")

    parts.append(f"# Agent Memory — {agent_id}\n")
    parts.append(f"Generated: {today}")
    parts.append(f"Last updated: {today}")
    parts.append("\n---\n")

    if cache_result and cache_result["text"]:
        parts.append("# System Prompt\n")
        parts.append(cache_result["text"])
        parts.append("\n---\n")

    parts.append("# Interview Transcript\n")
    parts.append(body)
    parts.append("\n---\n")

    parts.append("# Expert Observations\n")
    for result in expert_results:
        if not result["text"]:
            continue
        parts.append(f"## {result['name']}\n")
        parts.append(result["text"])
        parts.append("")

    if gaps_result and gaps_result["text"]:
        parts.append("---\n")
        parts.append("## Coverage Gaps & Conflicts\n")
        parts.append(gaps_result["text"])
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_genesis(transcript_path: Path, agent_id: str, skip_gaps: bool = False,
                skip_cache: bool = False, dry_run: bool = False) -> Path | None:
    """Run the genesis pipeline. Returns path to memory file on success."""
    validate_agent_id(agent_id)
    body = parse_transcript(transcript_path)
    demographics = parse_demographics(body)

    print(f"Agent: {agent_id}")
    print(f"Transcript: {len(body):,} chars (~{len(body) // 4:,} tokens)")
    if demographics:
        print(f"Demographics: {len(demographics)} fields extracted")
    else:
        print("WARNING: No demographic intake section found")

    today = date.today().isoformat()
    transcript_out = Path("transcripts") / f"{agent_id}_{today}_transcript.md"
    transcript_out.parent.mkdir(exist_ok=True)
    transcript_out.write_text(body, encoding="utf-8")
    print(f"Stored: {transcript_out}")

    if dry_run:
        return None

    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    client = Anthropic(api_key=api_key, max_retries=3)
    pipeline_start = time.time()

    expert_results = run_all_experts(client, body)
    live_experts = [r for r in expert_results if r["text"]]
    if len(live_experts) < 3:
        print(f"FATAL: only {len(live_experts)}/4 experts succeeded. Aborting.", file=sys.stderr)
        sys.exit(1)
    total_obs = sum(count_observations(r["text"]) for r in expert_results)
    print(f"  {total_obs} observations")

    gaps_result = None
    cache_result = None
    run_gaps = not skip_gaps
    run_cache = not skip_cache

    if run_gaps or run_cache:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            if run_gaps:
                futures[executor.submit(run_coverage_gaps, client, body, expert_results)] = "gaps"
            if run_cache:
                futures[executor.submit(generate_summary_cache, client, body, expert_results)] = "cache"
            for future in as_completed(futures):
                label = futures[future]
                if label == "gaps":
                    gaps_result = future.result()
                else:
                    cache_result = future.result()

    if cache_result:
        warnings = validate_cache(cache_result["text"])
        for w in warnings:
            print(f"  WARNING: {w}", file=sys.stderr)

    elapsed = time.time() - pipeline_start
    print(f"\nDone in {elapsed:.0f}s — {total_obs} observations")

    memory_path = Path("memory") / f"{agent_id}_{today}_memory.md"
    memory_path.parent.mkdir(exist_ok=True)
    memory = assemble_memory(agent_id, body, expert_results, gaps_result, cache_result, today, demographics)
    memory_path.write_text(memory, encoding="utf-8")
    print(f"Memory: {memory_path}")

    return memory_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate agent memory from a clean interview transcript.",
        usage="genesis.py <transcript> --agent-id <id> [--skip-gaps] [--skip-cache] [--dry-run]",
    )
    parser.add_argument("file", type=Path, help="Path to hand-cleaned transcript (.md)")
    parser.add_argument("--agent-id", required=True, help="Agent identifier (e.g., agent_1)")
    parser.add_argument("--skip-gaps", action="store_true", help="Skip coverage gaps analysis.")
    parser.add_argument("--skip-cache", action="store_true", help="Skip summary cache generation.")
    parser.add_argument("--dry-run", action="store_true", help="Read transcript and print stats, no API calls.")
    args = parser.parse_args()

    run_genesis(args.file, args.agent_id, args.skip_gaps, args.skip_cache, args.dry_run)


if __name__ == "__main__":
    main()
