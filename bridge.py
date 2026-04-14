"""Simic Stage 2 - Bridge.

Reads the memory file produced by Stage 1 (genesis) and generates the three
person-specific config files that Stage 3 (generate) needs:

  - agent_configs/<agent_id>/system_prompt.json
  - agent_configs/<agent_id>/frontier_approach_prompt.json
  - agent_configs/<agent_id>/taxonomy.json

Approach: two parallel Claude Sonnet calls.
  Call A: system_prompt (JSONL persona) + frontier_approach (rich inference-time prompt)
  Call B: per-category behavioral_anchors + generation prompts for all 25 categories

Template protocol for generated category prompts:
  The LLM's generated prompt strings MUST contain two literal placeholders:
    {count}       - substituted by Stage 3 with the number of pairs per batch
    {batch_info}  - substituted by Stage 3 with anti-repetition text for later batches
  Fixed values (first name, category domain, scenario lists, response length percentages)
  are baked into each prompt directly by the LLM - they do not vary per batch.

Usage:
    python bridge.py --agent-id alex_chen
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

SONNET = "claude-sonnet-4-5"
TAXONOMY_TEMPLATE = Path("prompts") / "category_taxonomy_template.json"
AGENT_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def validate_agent_id(agent_id: str) -> None:
    if not AGENT_ID_RE.match(agent_id):
        print(
            f"Error: agent_id '{agent_id}' is invalid. "
            f"Allowed characters: letters, digits, underscore, hyphen.",
            file=sys.stderr,
        )
        sys.exit(1)


def latest_memory_file(agent_id: str) -> Path:
    candidates = sorted(Path("memory").glob(f"{agent_id}_*_memory.md"))
    if not candidates:
        print(f"Error: no memory file for agent '{agent_id}'. Run stage 1 first.", file=sys.stderr)
        sys.exit(1)
    return candidates[-1]


def strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def parse_json_response(text: str, label: str, agent_id: str) -> dict:
    """Parse JSON from a model response. On failure, dump raw text and raise."""
    try:
        return json.loads(strip_json_fences(text))
    except json.JSONDecodeError as e:
        debug_dir = Path("agent_configs") / agent_id
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_path = debug_dir / f"bridge_debug_{label}.txt"
        debug_path.write_text(text)
        raise ValueError(
            f"Could not parse JSON from {label} response ({e}). "
            f"Raw output dumped to {debug_path}"
        ) from e


def fill(template: str, **subs: str) -> str:
    """Substitute <<KEY>> placeholders in a template. Does not interact with
    Python .format() - so the template can contain literal { } without escaping."""
    out = template
    for key, value in subs.items():
        out = out.replace(f"<<{key}>>", value)
    return out


# ---------------------------------------------------------------------------
# Call A - system_prompt + frontier_approach
# ---------------------------------------------------------------------------

PROMPTS_CALL_A = """You are adapting a person-specific fine-tuning harness to a specific human based on their memory file (system prompt + interview transcript + expert behavioral analysis).

=== AGENT MEMORY FILE ===

<<MEMORY>>

=== YOUR TASK ===

Produce ONE JSON object with exactly two top-level keys: "system_prompt" and "frontier_approach".

1. "system_prompt" - a string. Written in second person starting with "You are [name]...". 1-2 compact paragraphs (roughly 200-350 words). This will be the "system" role in every fine-tuning training example, so every token is expensive - keep it dense and voice-establishing.

2. "frontier_approach" - an object with these keys (all arrays of short strings unless stated):
   - "persona": 2 lines establishing identity in first-person framing ("You are not an AI assistant. You are [name], a real...")
   - "objective": 3-4 lines defining the agent's job (extrapolate likely opinion/choice, not neutral menus)
   - "answer_shape": 4-6 lines on response length, structure, and hard caps
   - "decision_rules": 4-6 lines on when to commit vs hedge, how to use "it depends"
   - "reasoning_rules": 3-5 lines on how to reason (first principles, evidence hierarchy, etc.)
   - "voice_rules": 6-10 lines on voice, verbal tics, code-switching, analogies, tone
   - "behavioral_anchors": an OBJECT mapping domain names (strings like "spending_and_purchases", "brand_loyalty", "trust_and_relationships", "career", "family", etc - you pick 8-12 domains that matter for THIS person) to arrays of 6-12 concrete anchor strings each. Every anchor must cite specific evidence from the memory file (names, amounts, incidents, brands).

HARD RULES:
- Every claim must be grounded in the memory file. No invented facts.
- Every behavioral anchor must cite specific evidence: amounts (currency + number), names of people, named brands, concrete incidents.
- Match the voice of this specific person, not a generic agent tone.
- Output ONLY raw JSON. No markdown fences. No preamble.
"""


# ---------------------------------------------------------------------------
# Call B - taxonomy (per-category anchors + prompts)
# ---------------------------------------------------------------------------

PROMPTS_CALL_B = """You are generating per-category fine-tuning data prompts for a specific person, based on their memory file.

=== AGENT MEMORY FILE ===

<<MEMORY>>

=== CATEGORIES (structural - do not change) ===

You will produce a generation prompt for each of these 25 categories. For each, output (1) behavioral_anchors specific to that category, and (2) a full generation prompt the data-generator model will receive.

<<CATEGORIES>>

=== YOUR TASK ===

Produce ONE JSON object with a single top-level key "categories", mapping each category_id to an object with this shape:

  {"A1": {"behavioral_anchors": ["string", ...], "prompt": "string"}, "A2": {...}, ...}

You MUST produce an entry for all 25 categories listed above. Missing categories will cause the pipeline to abort.

=== PROMPT TEMPLATE ===

Each generated "prompt" string must follow this shape exactly (the [SQUARE_BRACKET] parts are the parts you fill in based on the memory file and category; the {CURLY_BRACE} parts are placeholders that MUST appear verbatim in your output - do NOT replace them):

"You are generating fine-tuning data to replicate [FIRST_NAME] as an AI agent. Using the memory file provided as system context, generate {count} question-answer pairs where someone asks [FIRST_NAME] about [CATEGORY_DOMAIN].

Scenario types to cover:
- [scenario 1 - situational, specific, varied]
- [scenario 2]
- [scenario 3]
- [scenario 4]
- [scenario 5]
- [scenario 6]

CRITICAL RULES:
- All answers must be first-person, in [FIRST_NAME]'s natural voice
- Include reasoning process, not just conclusion
- Reference real patterns from their life: [3-5 category-specific anchors grounded in the memory file]
- Some answers should be short (1-2 sentences), others detailed walkthroughs
- Match their verbal tics and analogies
- Be matter-of-fact, not preachy - state what they'd do and why

Response length distribution for this category: short [SHORT]%, medium [MEDIUM]%, long [LONG]% (substitute actual percentages from the metadata above).

{batch_info}

Output as a JSON object with key 'pairs' containing an array of objects, each with 'question' and 'answer' fields."

=== PLACEHOLDER PROTOCOL - CRITICAL ===

In each generated "prompt" string:
- {count} MUST appear verbatim where the pair count goes. Do NOT write "10" or any number there. The downstream pipeline substitutes it at runtime.
- {batch_info} MUST appear verbatim on its own paragraph near the end (before "Output as a JSON object..."). Do NOT omit or rename.
- Do NOT use any other single-curly-brace tokens in the prompt. If you need a literal curly brace in the text, double it ({{ or }}).

=== MULTI-TURN CATEGORY (is_multiturn=true) ===

For the multi-turn category, the prompt should ask for {count} multi-turn conversations (not Q&A pairs). The output schema should be: 'conversations' containing objects each with a 'turns' array of objects with 'role' and 'content' fields. 'role' must be 'user' or 'assistant'. The {count} and {batch_info} placeholders still appear verbatim.

=== RULES ===

- Substitute [FIRST_NAME] with the actual first name from the memory file.
- Substitute [CATEGORY_DOMAIN] with a natural English phrase for the category (e.g. A1 electronics → "buying electronics or tech products").
- Fill scenario types with 5-7 situations relevant to both this category AND this specific person. Avoid scenarios the memory file gives no signal about.
- Fill the anchor references with 3-5 real patterns from the memory file for this specific category.
- Fill response length percentages from the category metadata above.
- Every behavioral_anchor cites specific evidence: amounts, names of people, named brands, incidents.
- Output ONLY raw JSON. No markdown fences. No preamble.
"""


# ---------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------

def call_a(client: Anthropic, memory_text: str, agent_id: str) -> dict:
    print("  [A] system_prompt + frontier_approach...")
    start = time.time()
    response = client.messages.create(
        model=SONNET,
        max_tokens=8000,
        messages=[{"role": "user", "content": fill(PROMPTS_CALL_A, MEMORY=memory_text)}],
    )
    text = response.content[0].text
    elapsed = time.time() - start
    print(f"  [A] done in {elapsed:.0f}s ({response.usage.input_tokens:,} in / {response.usage.output_tokens:,} out)")
    return parse_json_response(text, "call_a", agent_id)


def call_b(client: Anthropic, memory_text: str, template: dict, agent_id: str) -> dict:
    category_list = "\n".join(
        f"- {c['id']} {c['name']} ({c['description']}) - target_pairs={c['target_pairs']}, "
        f"distribution={c['response_length_distribution']}, is_multiturn={c.get('is_multiturn', False)}"
        for c in template["categories"]
    )
    content = fill(PROMPTS_CALL_B, MEMORY=memory_text, CATEGORIES=category_list)
    print("  [B] per-category anchors + prompts...")
    start = time.time()
    response = client.messages.create(
        model=SONNET,
        max_tokens=16000,
        messages=[{"role": "user", "content": content}],
    )
    text = response.content[0].text
    elapsed = time.time() - start
    print(f"  [B] done in {elapsed:.0f}s ({response.usage.input_tokens:,} in / {response.usage.output_tokens:,} out)")
    return parse_json_response(text, "call_b", agent_id)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def _validate_placeholder(prompt: str, placeholder: str, cid: str) -> str | None:
    """Return an error string if the placeholder isn't present verbatim in the prompt."""
    if placeholder not in prompt:
        return f"  {cid}: generated prompt is missing '{placeholder}' placeholder"
    return None


def assemble_taxonomy(template: dict, per_category: dict) -> dict:
    """Merge template metadata with per-category LLM output. Hard-fail on gaps."""
    categories = []
    missing = []
    errors = []
    for c in template["categories"]:
        cid = c["id"]
        generated = per_category.get("categories", {}).get(cid)
        if not generated:
            missing.append(cid)
            continue
        prompt = generated.get("prompt", "")
        anchors = generated.get("behavioral_anchors", [])
        for placeholder in ("{count}", "{batch_info}"):
            err = _validate_placeholder(prompt, placeholder, cid)
            if err:
                errors.append(err)
        if not anchors:
            errors.append(f"  {cid}: no behavioral_anchors produced")
        categories.append({
            **c,
            "behavioral_anchors": anchors,
            "prompt": prompt,
        })

    if missing:
        print(f"Error: bridge did not produce entries for categories: {missing}", file=sys.stderr)
        print("  Re-run the bridge. The LLM must cover all 25 categories.", file=sys.stderr)
        sys.exit(1)

    if errors:
        print("Error: generated taxonomy failed protocol validation:", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        print("  Re-run the bridge. Generated prompts must contain {count} and {batch_info} placeholders.", file=sys.stderr)
        sys.exit(1)

    return {
        **template.get("metadata", {}),
        "categories": categories,
    }


def write_configs(agent_id: str, call_a_result: dict, taxonomy: dict) -> None:
    config_dir = Path("agent_configs") / agent_id
    config_dir.mkdir(parents=True, exist_ok=True)

    sp_path = config_dir / "system_prompt.json"
    sp_path.write_text(json.dumps(
        {"system_prompt": call_a_result["system_prompt"]}, indent=2, ensure_ascii=False
    ))
    print(f"  Wrote {sp_path}")

    fa_path = config_dir / "frontier_approach_prompt.json"
    fa_path.write_text(json.dumps(call_a_result["frontier_approach"], indent=2, ensure_ascii=False))
    print(f"  Wrote {fa_path}")

    tax_path = config_dir / "taxonomy.json"
    tax_path.write_text(json.dumps(taxonomy, indent=2, ensure_ascii=False))
    print(f"  Wrote {tax_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_bridge(agent_id: str) -> None:
    validate_agent_id(agent_id)
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    memory_path = latest_memory_file(agent_id)
    if not TAXONOMY_TEMPLATE.exists():
        print(f"Error: taxonomy template missing at {TAXONOMY_TEMPLATE}", file=sys.stderr)
        sys.exit(1)

    memory_text = memory_path.read_text(encoding="utf-8")
    template = json.loads(TAXONOMY_TEMPLATE.read_text())

    print(f"Agent: {agent_id}")
    print(f"Memory: {memory_path} ({len(memory_text):,} chars)")
    print(f"Template: {len(template['categories'])} categories")

    client = Anthropic(api_key=api_key, max_retries=3)

    pipeline_start = time.time()
    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_a = executor.submit(call_a, client, memory_text, agent_id)
        fut_b = executor.submit(call_b, client, memory_text, template, agent_id)

        errors = []
        call_a_result = None
        call_b_result = None
        try:
            call_a_result = fut_a.result()
        except Exception as e:
            errors.append(f"Call A failed: {e}")
        try:
            call_b_result = fut_b.result()
        except Exception as e:
            errors.append(f"Call B failed: {e}")

    if errors:
        for err in errors:
            print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)

    taxonomy = assemble_taxonomy(template, call_b_result)
    write_configs(agent_id, call_a_result, taxonomy)

    elapsed = time.time() - pipeline_start
    print(f"\nBridge done in {elapsed:.0f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Simic Stage 2 - generate per-agent fine-tune configs from memory file",
        usage="bridge.py --agent-id <id>",
    )
    parser.add_argument("--agent-id", required=True, help="Agent identifier (matches genesis output)")
    args = parser.parse_args()

    run_bridge(args.agent_id)


if __name__ == "__main__":
    main()
