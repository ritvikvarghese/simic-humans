"""Simic - unified CLI for the three-stage pipeline.

Stage 1: genesis      - transcript → memory file (the simagent)
Stage 2: bridge       - memory file → per-agent fine-tune configs
Stage 3: generate     - configs + memory → JSONL fine-tune dataset

Default: Stage 1 only (fast, ~1 min, produces a working prompt-engineered agent).
--finetune: all three stages (slow, ~2600 API calls for stage 3, produces JSONL).

Usage:
    python simic.py transcripts/alex.md --agent-id alex_chen
    python simic.py transcripts/alex.md --agent-id alex_chen --finetune
    python simic.py transcripts/alex.md --agent-id alex_chen --finetune --category A1
    python simic.py transcripts/alex.md --agent-id alex_chen --dry-run

For finer control, invoke each stage directly: genesis.py / bridge.py / generate.py.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from genesis import run_genesis
from bridge import run_bridge
from generate import DEFAULT_CONFIG, compile_jsonl, run_generation

AGENT_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def validate_agent_id(agent_id: str) -> None:
    if not AGENT_ID_RE.match(agent_id):
        print(
            f"Error: agent_id '{agent_id}' is invalid. "
            f"Allowed characters: letters, digits, underscore, hyphen.",
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Simic - transcript to simulated agent (prompt + optional fine-tune).",
        usage="simic.py <transcript.md> --agent-id <id> [--finetune] [options]",
    )
    parser.add_argument("file", type=Path, nargs="?", help="Path to hand-cleaned transcript (.md). Required unless --compile is passed.")
    parser.add_argument("--agent-id", required=True, help="Agent identifier")

    # Stage 1 pass-through
    parser.add_argument("--skip-gaps", action="store_true", help="Stage 1: skip coverage gaps analysis")
    parser.add_argument("--skip-cache", action="store_true", help="Stage 1: skip summary cache generation")
    parser.add_argument("--dry-run", action="store_true", help="Stage 1: parse transcript only, no API calls")

    # Stage 3 toggle + pass-through
    parser.add_argument("--finetune", action="store_true", help="Run all 3 stages (default: stage 1 only)")
    parser.add_argument("--bridge-only", action="store_true", help="Run stages 1+2 (skip fine-tune data generation)")
    parser.add_argument("--model", type=str, help="Stage 3: override generator model name")
    parser.add_argument("--base-url", type=str, help="Stage 3: override API base URL")
    parser.add_argument("--api-key-env", type=str, help="Stage 3: override API key env var")
    parser.add_argument("--concurrency", type=int, help="Stage 3: max concurrent requests")
    parser.add_argument("--rpm", type=int, help="Stage 3: requests per minute")
    parser.add_argument("--category", type=str, help="Stage 3: run only a specific category (e.g. A1)")
    parser.add_argument("--compile", action="store_true", help="Stage 3: compile existing batches into JSONL then exit")

    args = parser.parse_args()
    validate_agent_id(args.agent_id)

    if args.compile:
        compile_jsonl(args.agent_id)
        return

    if args.file is None:
        parser.error("the 'file' argument is required (unless --compile is passed)")

    # Stage 1: genesis
    print("\n" + "="*60)
    print(" STAGE 1 - genesis")
    print("="*60)
    memory_path = run_genesis(
        transcript_path=args.file,
        agent_id=args.agent_id,
        skip_gaps=args.skip_gaps,
        skip_cache=args.skip_cache,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        return

    if memory_path is None:
        print("Stage 1 did not produce a memory file. Aborting.", file=sys.stderr)
        sys.exit(1)

    if not (args.finetune or args.bridge_only):
        print("\nStage 1 complete. To also produce fine-tune data, re-run with --finetune.")
        return

    # Stage 2: bridge
    print("\n" + "="*60)
    print(" STAGE 2 - bridge")
    print("="*60)
    run_bridge(args.agent_id)

    if args.bridge_only:
        print("\nStages 1+2 complete. Configs written to agent_configs/. Run generate.py to produce JSONL.")
        return

    # Stage 3: generate
    print("\n" + "="*60)
    print(" STAGE 3 - generate fine-tune data")
    print("="*60)

    load_dotenv()
    config = dict(DEFAULT_CONFIG)
    if args.model:
        config["model"] = args.model
    if args.base_url:
        config["base_url"] = args.base_url
    if args.api_key_env:
        config["api_key_env"] = args.api_key_env
    if args.concurrency:
        config["max_concurrent_requests"] = args.concurrency
    if args.rpm:
        config["requests_per_minute"] = args.rpm
    if args.category:
        config["category_filter"] = args.category
    config["api_key"] = os.environ.get(config["api_key_env"], "")

    asyncio.run(run_generation(args.agent_id, config))

    print("\nAll stages complete.")
    print(f"  Memory file:   {memory_path}")
    print(f"  Configs:       agent_configs/{args.agent_id}/")
    print(f"  Batches:       db/{args.agent_id}/")
    print(f"  To compile JSONL: python simic.py {args.file} --agent-id {args.agent_id} --compile")


if __name__ == "__main__":
    main()
