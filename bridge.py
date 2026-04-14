"""Simic Stage 2 — Bridge.

Reads the memory file produced by Stage 1 (genesis) and generates the three
person-specific config files that Stage 3 (generate) needs:

  - agent_configs/<agent_id>/system_prompt.json
  - agent_configs/<agent_id>/frontier_approach_prompt.json
  - agent_configs/<agent_id>/taxonomy.json

Approach: two parallel Claude Sonnet calls.
  Call A: system_prompt (JSONL persona) + frontier_approach (rich inference-time prompt)
  Call B: per-category behavioral_anchors + generation prompts for all 25 categories

Usage:
    python bridge.py --agent-id alex_chen
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

SONNET = "claude-sonnet-4-5"
TAXONOMY_TEMPLATE = Path("prompts") / "category_taxonomy_template.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def parse_json_response(text: str, label: str) -> dict:
    try:
        return json.loads(strip_json_fences(text))
    except json.JSONDecodeError as e:
        print(f"Error: could not parse JSON from {label} call: {e}", file=sys.stderr)
        # Dump raw for debugging
        debug_path = Path(f"bridge_debug_{label}.txt")
        debug_path.write_text(text)
        print(f"  Raw response written to {debug_path}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Call A — system_prompt + frontier_approach
# ---------------------------------------------------------------------------

PROMPTS_CALL_A = """You are adapting a person-specific fine-tuning harness to a specific human based on their memory file (system prompt + interview transcript + expert behavioral analysis).

=== AGENT MEMORY FILE ===

{memory_text}

=== YOUR TASK ===

Produce ONE JSON object with exactly two top-level keys: "system_prompt" and "frontier_approach".

1. "system_prompt" — a string. Written in second person starting with "You are [name]...". 1-2 compact paragraphs (roughly 200-350 words). This will be the "system" role in every fine-tuning training example, so every token is expensive — keep it dense and voice-establishing.

2. "frontier_approach" — an object with these keys (all arrays of short strings unless stated):
   - "persona": 2 lines establishing identity in first-person framing ("You are not an AI assistant. You are [name], a real...")
   - "objective": 3-4 lines defining the agent's job (extrapolate likely opinion/choice, not neutral menus)
   - "answer_shape": 4-6 lines on response length, structure, and hard caps
   - "decision_rules": 4-6 lines on when to commit vs hedge, how to use "it depends"
   - "reasoning_rules": 3-5 lines on how to reason (first principles, evidence hierarchy, etc.)
   - "voice_rules": 6-10 lines on voice, verbal tics, code-switching, analogies, tone
   - "behavioral_anchors": an OBJECT mapping domain names (strings like "spending_and_purchases", "brand_loyalty", "trust_and_relationships", "career", "family", etc — you pick 8-12 domains that matter for THIS person) to arrays of 6-12 concrete anchor strings each. Every anchor must cite specific evidence from the memory file (names, amounts, incidents, brands).

HARD RULES:
- Every claim must be grounded in the memory file. No invented facts.
- Every behavioral anchor must cite specific evidence: amounts (₹, $), names of people, named brands, concrete incidents.
- Match the voice of this specific person, not a generic agent tone.
- Output ONLY raw JSON. No markdown fences. No preamble.
"""


# ---------------------------------------------------------------------------
# Call B — taxonomy (per-category anchors + prompts)
# ---------------------------------------------------------------------------

PROMPTS_CALL_B = """You are generating per-category fine-tuning data prompts for a specific person, based on their memory file.

=== AGENT MEMORY FILE ===

{memory_text}

=== CATEGORIES (structural — do not change) ===

You will produce a generation prompt for each of these 25 categories. For each, output (1) behavioral_anchors specific to that category, and (2) a full generation prompt the data-generator model will receive.

{category_list}

=== YOUR TASK ===

Produce ONE JSON object with a single top-level key "categories", mapping category_id to this shape:

{{
  "A1": {{
    "behavioral_anchors": ["string", ...],   // 6-10 anchors specific to this category, each citing evidence
    "prompt": "string"                        // the full generation prompt, see template below
  }},
  "A2": {{ ... }},
  ...
}}

=== PROMPT TEMPLATE (follow this shape for each category's "prompt") ===

"You are generating fine-tuning data to replicate [FIRST_NAME] as an AI agent. Using the memory file provided as system context, generate 10 question-answer pairs where someone asks [FIRST_NAME] about [CATEGORY_DOMAIN].

Scenario types to cover:
- [scenario 1 — situational, specific, varied]
- [scenario 2]
- [scenario 3]
- [scenario 4]
- [scenario 5]
- [scenario 6]

CRITICAL RULES:
- All answers must be first-person, in [FIRST_NAME]'s natural voice
- Include reasoning process, not just conclusion
- Reference real patterns from their life: [3-5 category-specific anchors, e.g. 'asking Sarah for recommendations', 'using their Amex for groceries', 'avoiding subscriptions']
- Some answers should be short (1-2 sentences), others detailed walkthroughs (per response_length_distribution)
- Match their verbal tics and analogies
- Be matter-of-fact, not preachy — state what they'd do and why

Response length distribution for this category: short {short_pct}%, medium {medium_pct}%, long {long_pct}%.

Output as a JSON object with key 'pairs' containing an array of objects, each with 'question' and 'answer' fields."

(For the multi-turn category where is_multiturn is true, replace "10 question-answer pairs" with "5 multi-turn conversations", and replace the output schema with 'conversations' containing objects each with a 'turns' array of {{role, content}} objects. role may be 'user' or 'assistant'.)

=== RULES ===

- Substitute [FIRST_NAME] with the actual first name from the memory file.
- Substitute [CATEGORY_DOMAIN] with a natural English phrase for the category (e.g. A1 electronics → "buying electronics or tech products"; E2 values_religion_spirituality → "religion, spirituality, and belief").
- Fill [3-5 category-specific anchors] with specific grounded references from THIS person's memory file for THIS category.
- Fill response_length_distribution percentages from the category metadata provided above.
- Scenario types must be RELEVANT to both the category AND this specific person. Avoid scenarios the transcript gives no signal about.
- Every behavioral_anchor cites specific evidence: amounts, names of people, named brands, incidents.
- Output ONLY raw JSON. No markdown fences. No preamble.
"""


# ---------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------

def call_a(client: Anthropic, memory_text: str) -> dict:
    print("  [A] system_prompt + frontier_approach...")
    start = time.time()
    response = client.messages.create(
        model=SONNET,
        max_tokens=8000,
        messages=[{"role": "user", "content": PROMPTS_CALL_A.format(memory_text=memory_text)}],
    )
    text = response.content[0].text
    elapsed = time.time() - start
    print(f"  [A] done in {elapsed:.0f}s ({response.usage.input_tokens:,} in / {response.usage.output_tokens:,} out)")
    return parse_json_response(text, "call_a")


def call_b(client: Anthropic, memory_text: str, template: dict) -> dict:
    category_list = "\n".join(
        f"- {c['id']} {c['name']} ({c['description']}) — target_pairs={c['target_pairs']}, "
        f"distribution={c['response_length_distribution']}, is_multiturn={c.get('is_multiturn', False)}"
        for c in template["categories"]
    )
    print("  [B] per-category anchors + prompts...")
    start = time.time()
    response = client.messages.create(
        model=SONNET,
        max_tokens=16000,
        messages=[{
            "role": "user",
            "content": PROMPTS_CALL_B.format(memory_text=memory_text, category_list=category_list),
        }],
    )
    text = response.content[0].text
    elapsed = time.time() - start
    print(f"  [B] done in {elapsed:.0f}s ({response.usage.input_tokens:,} in / {response.usage.output_tokens:,} out)")
    return parse_json_response(text, "call_b")


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def assemble_taxonomy(template: dict, per_category: dict) -> dict:
    categories = []
    missing = []
    for c in template["categories"]:
        cid = c["id"]
        generated = per_category.get("categories", {}).get(cid)
        if not generated:
            missing.append(cid)
            continue
        categories.append({
            **c,
            "behavioral_anchors": generated.get("behavioral_anchors", []),
            "prompt": generated.get("prompt", ""),
        })

    if missing:
        print(f"  WARNING: bridge did not produce entries for: {missing}", file=sys.stderr)

    return {
        "metadata": {
            **template.get("metadata", {}),
            "bridge_version": "1.0",
        },
        "categories": categories,
    }


def write_configs(agent_id: str, call_a_result: dict, taxonomy: dict) -> None:
    config_dir = Path("agent_configs") / agent_id
    config_dir.mkdir(parents=True, exist_ok=True)

    # system_prompt.json
    sp_path = config_dir / "system_prompt.json"
    sp_path.write_text(json.dumps(
        {"system_prompt": call_a_result["system_prompt"]}, indent=2, ensure_ascii=False
    ))
    print(f"  Wrote {sp_path}")

    # frontier_approach_prompt.json
    fa_path = config_dir / "frontier_approach_prompt.json"
    fa_path.write_text(json.dumps({
        "metadata": {
            "version": "1.0",
            "purpose": "Rich inference-time system prompt for the frontier baseline or for testing the fine-tuned model with a richer persona.",
            "usage": "Load this file, assemble sections into a system prompt, and append the memory file if desired.",
            "notes": "Generated by simic bridge from the memory file.",
        },
        **call_a_result["frontier_approach"],
    }, indent=2, ensure_ascii=False))
    print(f"  Wrote {fa_path}")

    # taxonomy.json
    tax_path = config_dir / "taxonomy.json"
    tax_path.write_text(json.dumps(taxonomy, indent=2, ensure_ascii=False))
    print(f"  Wrote {tax_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_bridge(agent_id: str) -> None:
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    memory_path = latest_memory_file(agent_id)
    template_path = TAXONOMY_TEMPLATE
    if not template_path.exists():
        print(f"Error: taxonomy template missing at {template_path}", file=sys.stderr)
        sys.exit(1)

    memory_text = memory_path.read_text(encoding="utf-8")
    template = json.loads(template_path.read_text())

    print(f"Agent: {agent_id}")
    print(f"Memory: {memory_path} ({len(memory_text):,} chars)")
    print(f"Template: {len(template['categories'])} categories")

    client = Anthropic(api_key=api_key, max_retries=3)

    pipeline_start = time.time()
    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_a = executor.submit(call_a, client, memory_text)
        fut_b = executor.submit(call_b, client, memory_text, template)
        call_a_result = fut_a.result()
        call_b_result = fut_b.result()

    taxonomy = assemble_taxonomy(template, call_b_result)
    write_configs(agent_id, call_a_result, taxonomy)

    elapsed = time.time() - pipeline_start
    print(f"\nBridge done in {elapsed:.0f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Simic Stage 2 — generate per-agent fine-tune configs from memory file",
        usage="bridge.py --agent-id <id>",
    )
    parser.add_argument("--agent-id", required=True, help="Agent identifier (matches genesis output)")
    args = parser.parse_args()

    run_bridge(args.agent_id)


if __name__ == "__main__":
    main()
