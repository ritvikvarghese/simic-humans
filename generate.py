#!/usr/bin/env python3
"""Simic Stage 3 — Synthetic fine-tune data generator.

Takes a per-agent memory file + per-agent taxonomy (produced by bridge.py)
and generates a JSONL training dataset to fine-tune a model to be this person.

Resumable, parallel, with progress tracking per agent.

Usage:
    python generate.py --agent-id alex_chen
    python generate.py --agent-id alex_chen --status
    python generate.py --agent-id alex_chen --compile
    python generate.py --agent-id alex_chen --reset
    python generate.py --agent-id alex_chen --reset-category A1
    python generate.py --agent-id alex_chen --model "anthropic/claude-sonnet-4-5" --concurrency 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import httpx
except ImportError:
    httpx = None

AGENT_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def validate_agent_id(agent_id: str) -> None:
    if not AGENT_ID_RE.match(agent_id):
        print(
            f"Error: agent_id '{agent_id}' is invalid. "
            f"Allowed characters: letters, digits, underscore, hyphen.",
            file=sys.stderr,
        )
        sys.exit(1)


# ============================================================
# CONFIG — defaults. Override via CLI.
# ============================================================

DEFAULT_CONFIG = {
    # API (defaults target Z.AI GLM-4.6; cheap at scale for 2000+ generations)
    "api_key_env": "ZAI_API_KEY",
    "base_url": "https://api.z.ai/api/paas/v4/chat/completions",
    "model": "glm-4.6",
    "json_mode": True,

    # Rate limiting
    "max_concurrent_requests": 2,
    "requests_per_minute": 9999,
    "retry_attempts": 3,
    "retry_delay_seconds": 5,

    # Generation
    "batch_size": 10,
    "temperature": 0.85,
    "max_tokens": 16000,
}


# ============================================================
# PATHS (per-agent)
# ============================================================

def agent_paths(agent_id: str) -> dict:
    base = Path(".")
    config_dir = base / "agent_configs" / agent_id
    return {
        "taxonomy": config_dir / "taxonomy.json",
        "system_prompt": config_dir / "system_prompt.json",
        "memory_glob": str(base / "memory" / f"{agent_id}_*_memory.md"),
        "db_dir": base / "db" / agent_id,
        "output_dir": base / "output" / agent_id,
    }


def latest_memory_file(agent_id: str) -> Path:
    candidates = sorted(Path("memory").glob(f"{agent_id}_*_memory.md"))
    if not candidates:
        raise FileNotFoundError(
            f"No memory file for agent '{agent_id}'. Run stage 1 first: "
            f"python simic.py <transcript.md> --agent-id {agent_id}"
        )
    return candidates[-1]


# ============================================================
# PROGRESS DB (JSON files per agent)
# ============================================================

class ProgressDB:
    def __init__(self, db_dir: Path):
        self.db_dir = db_dir
        self.batches_dir = db_dir / "raw_batches"
        self.progress_file = db_dir / "progress.json"
        self.errors_file = db_dir / "errors.json"
        self.run_log_file = db_dir / "run_log.json"

        db_dir.mkdir(parents=True, exist_ok=True)
        self.batches_dir.mkdir(parents=True, exist_ok=True)

        self.progress = self._load(self.progress_file, {
            "created_at": datetime.now().isoformat(),
            "batches": {},
        })
        self.errors = self._load(self.errors_file, {"errors": []})
        self.run_log = self._load(self.run_log_file, {"runs": []})

    @staticmethod
    def _load(path: Path, default: dict) -> dict:
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return default

    def _save(self, path: Path, data: dict):
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def is_batch_done(self, batch_id: str) -> bool:
        return self.progress["batches"].get(batch_id, {}).get("status") == "done"

    def mark_batch_done(self, batch_id: str, pair_count: int):
        self.progress["batches"][batch_id] = {
            "status": "done",
            "pair_count": pair_count,
            "completed_at": datetime.now().isoformat(),
        }
        self._save(self.progress_file, self.progress)

    def mark_batch_failed(self, batch_id: str, error: str):
        self.progress["batches"][batch_id] = {
            "status": "failed",
            "error": error,
            "failed_at": datetime.now().isoformat(),
        }
        self._save(self.progress_file, self.progress)
        self.errors["errors"].append({
            "batch_id": batch_id,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        })
        self._save(self.errors_file, self.errors)

    def save_batch_data(self, batch_id: str, data: list):
        with open(self.batches_dir / f"{batch_id}.json", "w") as f:
            json.dump(data, f, indent=2)

    def load_batch_data(self, batch_id: str) -> list:
        f = self.batches_dir / f"{batch_id}.json"
        if f.exists():
            with open(f) as fp:
                return json.load(fp)
        return []

    def log_run(self, stats: dict):
        self.run_log["runs"].append({"timestamp": datetime.now().isoformat(), **stats})
        self._save(self.run_log_file, self.run_log)

    def get_stats(self) -> dict:
        batches = self.progress["batches"]
        done = sum(1 for b in batches.values() if b.get("status") == "done")
        failed = sum(1 for b in batches.values() if b.get("status") == "failed")
        total_pairs = sum(b.get("pair_count", 0) for b in batches.values() if b.get("status") == "done")
        return {
            "total_batches_tracked": len(batches),
            "done": done,
            "failed": failed,
            "total_pairs_generated": total_pairs,
        }

    def reset_all(self):
        import shutil
        if self.db_dir.exists():
            shutil.rmtree(self.db_dir)
        self.__init__(self.db_dir)

    def reset_category(self, category_id: str):
        to_remove = [bid for bid in self.progress["batches"] if bid.startswith(f"{category_id}_batch_")]
        for bid in to_remove:
            del self.progress["batches"][bid]
            f = self.batches_dir / f"{bid}.json"
            if f.exists():
                f.unlink()
        self._save(self.progress_file, self.progress)


# ============================================================
# BATCH PLANNING
# ============================================================

def plan_batches(taxonomy: dict, batch_size: int) -> list:
    """Break each category into batches and substitute runtime placeholders.

    The bridge-generated category prompts must contain literal `{count}` and
    `{batch_info}` placeholders. This function populates them per-batch.
    """
    batches = []
    for category in taxonomy["categories"]:
        cat_id = category["id"]
        target = category["target_pairs"]
        num_batches = (target + batch_size - 1) // batch_size

        template = category["prompt"]
        for placeholder in ("{count}", "{batch_info}"):
            if placeholder not in template:
                print(
                    f"Error: taxonomy prompt for category {cat_id} is missing required "
                    f"placeholder {placeholder!r}. Re-run bridge.py to regenerate the taxonomy.",
                    file=sys.stderr,
                )
                sys.exit(1)

        for i in range(num_batches):
            pairs_in_batch = min(batch_size, target - i * batch_size)
            batch_id = f"{cat_id}_batch_{i:03d}"

            batch_info = ""
            if num_batches > 1:
                batch_info = (
                    f"IMPORTANT: This is batch {i+1} of {num_batches} for this category. "
                    f"Previous batches have already covered some scenarios. Generate COMPLETELY "
                    f"DIFFERENT questions and scenarios than what a previous batch might have "
                    f"produced. Be creative and varied — explore edge cases, unusual angles, "
                    f"and specific situations. Do NOT repeat common or obvious questions."
                )

            try:
                prompt = category["prompt"].format(
                    count=pairs_in_batch,
                    batch_info=batch_info,
                )
            except (KeyError, IndexError) as e:
                print(
                    f"Error: taxonomy prompt for category {cat_id} is missing a required "
                    f"placeholder ({e}). Re-run bridge.py to regenerate the taxonomy.",
                    file=sys.stderr,
                )
                sys.exit(1)

            batches.append({
                "batch_id": batch_id,
                "category_id": cat_id,
                "category_name": category["name"],
                "batch_index": i,
                "total_batches": num_batches,
                "target_pairs": pairs_in_batch,
                "prompt": prompt,
                "is_multiturn": category.get("is_multiturn", False),
            })

    return batches


# ============================================================
# API CALLING
# ============================================================

class RateLimiter:
    def __init__(self, requests_per_minute: int):
        self.rpm = requests_per_minute
        self.interval = 60.0 / requests_per_minute
        self.last_request_time = 0
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_request_time
            if elapsed < self.interval:
                await asyncio.sleep(self.interval - elapsed)
            self.last_request_time = time.monotonic()


async def call_api(
    client: "httpx.AsyncClient",
    memory_text: str,
    generation_prompt: str,
    config: dict,
    semaphore: asyncio.Semaphore,
    rate_limiter: RateLimiter,
) -> dict:
    """Call the generator model with the agent's memory file as system context."""

    system_message = (
        "You are an expert at creating realistic, high-fidelity synthetic "
        "training data for fine-tuning language models. You have been given "
        "a comprehensive memory file for a specific person, including their "
        "system prompt, interview transcript, and expert behavioral analysis. "
        "Your job is to generate training pairs that accurately replicate this "
        "person's voice, reasoning patterns, values, and decision-making processes.\n\n"
        "=== AGENT MEMORY FILE ===\n\n"
        f"{memory_text}"
    )

    effective_prompt = generation_prompt
    if not config.get("json_mode"):
        effective_prompt += (
            "\n\nIMPORTANT: Respond ONLY with a valid JSON object. "
            "No markdown fences, no preamble, no explanation — just the raw JSON."
        )

    async with semaphore:
        await rate_limiter.acquire()

        for attempt in range(config["retry_attempts"]):
            try:
                request_body = {
                    "model": config["model"],
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": effective_prompt},
                    ],
                    "temperature": config["temperature"],
                    "max_tokens": config["max_tokens"],
                }

                if config.get("json_mode"):
                    request_body["response_format"] = {"type": "json_object"}

                # Z.AI requires thinking disabled explicitly
                if "z.ai" in config["base_url"]:
                    request_body["thinking"] = {"type": "disabled"}

                response = await client.post(
                    config["base_url"],
                    headers={
                        "Authorization": f"Bearer {config['api_key']}",
                        "Content-Type": "application/json",
                    },
                    json=request_body,
                    timeout=120.0,
                )

                if response.status_code == 429:
                    retry_after = int(response.headers.get("retry-after", 30))
                    print(f"    Rate limited. Waiting {retry_after}s...")
                    await asyncio.sleep(retry_after)
                    continue

                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                return json.loads(content)

            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                wait_time = config["retry_delay_seconds"] * (2 ** attempt)
                print(f"    Attempt {attempt+1} failed: {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

            except json.JSONDecodeError as e:
                print(f"    JSON parse error: {e}.")
                wait_time = config["retry_delay_seconds"] * (2 ** attempt)
                await asyncio.sleep(wait_time)

        raise RuntimeError(f"All {config['retry_attempts']} attempts failed")


# ============================================================
# PAIR EXTRACTION
# ============================================================

def extract_pairs(raw_response: dict, batch: dict, system_prompt_text: str) -> list:
    formatted = []

    if batch["is_multiturn"]:
        conversations = raw_response.get("conversations", [])
        for conv in conversations:
            turns = conv.get("turns", [])
            messages = [{"role": "system", "content": system_prompt_text}]
            for turn in turns:
                role = turn.get("role", "")
                content = turn.get("content", "")
                if role in ("questioner", "user", "interviewer"):
                    messages.append({"role": "user", "content": content})
                elif role in ("assistant", "agent", "subject"):
                    messages.append({"role": "assistant", "content": content})
            if len(messages) > 2:
                formatted.append({"messages": messages})
    else:
        pairs = raw_response.get("pairs", [])
        for pair in pairs:
            question = pair.get("question", "")
            answer = pair.get("answer", "")
            if question and answer:
                formatted.append({
                    "messages": [
                        {"role": "system", "content": system_prompt_text},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ]
                })

    return formatted


# ============================================================
# MAIN LOOP
# ============================================================

async def process_batch(
    batch: dict,
    client: "httpx.AsyncClient",
    memory_text: str,
    system_prompt_text: str,
    config: dict,
    db: ProgressDB,
    semaphore: asyncio.Semaphore,
    rate_limiter: RateLimiter,
):
    batch_id = batch["batch_id"]
    if db.is_batch_done(batch_id):
        return

    try:
        raw_response = await call_api(
            client=client,
            memory_text=memory_text,
            generation_prompt=batch["prompt"],
            config=config,
            semaphore=semaphore,
            rate_limiter=rate_limiter,
        )
        pairs = extract_pairs(raw_response, batch, system_prompt_text)

        if not pairs:
            db.mark_batch_failed(batch_id, "No pairs extracted from response")
            print(f"  ✗ {batch_id} — no pairs extracted")
            return

        db.save_batch_data(batch_id, pairs)
        db.mark_batch_done(batch_id, len(pairs))
        print(f"  ✓ {batch_id} — {len(pairs)} pairs")

    except Exception as e:
        db.mark_batch_failed(batch_id, str(e))
        print(f"  ✗ {batch_id} — {e}")


async def run_generation(agent_id: str, config: dict):
    validate_agent_id(agent_id)
    if httpx is None:
        print("ERROR: httpx is required. Install it: pip install httpx")
        return

    paths = agent_paths(agent_id)
    memory_path = latest_memory_file(agent_id)

    for label, path in [("taxonomy", paths["taxonomy"]), ("system prompt", paths["system_prompt"])]:
        if not path.exists():
            print(f"ERROR: {label} file not found at {path}")
            print(f"  Run the bridge first: python bridge.py --agent-id {agent_id}")
            return

    if not config["api_key"]:
        print(f"ERROR: API key not set. export {config['api_key_env']}='your-key-here'")
        return

    print(f"Agent: {agent_id}")
    print(f"Memory: {memory_path}")

    with open(memory_path) as f:
        memory_text = f.read()
    with open(paths["taxonomy"]) as f:
        taxonomy = json.load(f)
    with open(paths["system_prompt"]) as f:
        system_prompt_text = json.load(f)["system_prompt"]

    print(f"  Memory: {len(memory_text.split())} words")
    print(f"  Categories: {len(taxonomy['categories'])}")

    all_batches = plan_batches(taxonomy, config["batch_size"])

    category_filter = config.get("category_filter")
    if category_filter:
        all_batches = [b for b in all_batches if b["category_id"] == category_filter]
        if not all_batches:
            print(f"ERROR: No category '{category_filter}'")
            print(f"  Available: {', '.join(c['id'] for c in taxonomy['categories'])}")
            return
        print(f"  Filtered to category: {category_filter} ({len(all_batches)} batches)")

    print(f"  Total batches planned: {len(all_batches)}")

    db = ProgressDB(paths["db_dir"])
    stats_before = db.get_stats()

    pending = [b for b in all_batches if not db.is_batch_done(b["batch_id"])]
    print(f"  Already completed: {stats_before['done']}")
    print(f"  Pending: {len(pending)}")

    if not pending:
        print("\nAll batches already completed! Use --compile to build the JSONL.")
        return

    semaphore = asyncio.Semaphore(config["max_concurrent_requests"])
    rate_limiter = RateLimiter(config["requests_per_minute"])

    print(f"\nStarting generation...")
    print(f"  Model: {config['model']}")
    print(f"  Base URL: {config['base_url']}")
    print(f"  Concurrency: {config['max_concurrent_requests']}")
    print(f"  Rate limit: {config['requests_per_minute']} req/min")
    print()

    start_time = time.time()
    async with httpx.AsyncClient() as client:
        tasks = [
            process_batch(
                batch=batch,
                client=client,
                memory_text=memory_text,
                system_prompt_text=system_prompt_text,
                config=config,
                db=db,
                semaphore=semaphore,
                rate_limiter=rate_limiter,
            )
            for batch in pending
        ]
        await asyncio.gather(*tasks)

    elapsed = time.time() - start_time
    stats_after = db.get_stats()
    new_pairs = stats_after["total_pairs_generated"] - stats_before["total_pairs_generated"]
    new_batches = stats_after["done"] - stats_before["done"]

    print(f"\n{'='*50}")
    print(f"Run complete in {elapsed:.1f}s")
    print(f"  New batches: {new_batches}, new pairs: {new_pairs}")
    print(f"  Failed: {stats_after['failed']}, total pairs: {stats_after['total_pairs_generated']}")
    print(f"{'='*50}")

    db.log_run({
        "elapsed_seconds": round(elapsed, 1),
        "new_batches": new_batches,
        "new_pairs": new_pairs,
        "failed": stats_after["failed"],
        "total_pairs": stats_after["total_pairs_generated"],
    })

    if stats_after["failed"] > 0:
        print(f"\n{stats_after['failed']} batches failed. Re-run to retry.")


# ============================================================
# COMPILE
# ============================================================

def compile_jsonl(agent_id: str):
    validate_agent_id(agent_id)
    paths = agent_paths(agent_id)
    db = ProgressDB(paths["db_dir"])
    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "training_data.jsonl"
    all_examples = []
    batch_ids = sorted([
        bid for bid, info in db.progress["batches"].items()
        if info.get("status") == "done"
    ])
    for batch_id in batch_ids:
        all_examples.extend(db.load_batch_data(batch_id))

    with open(output_file, "w") as f:
        for example in all_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Compiled {len(all_examples)} examples from {len(batch_ids)} batches")
    print(f"Output: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

    summary_file = output_dir / "training_data_summary.json"
    category_counts = {}
    for batch_id in batch_ids:
        cat_id = batch_id.split("_batch_")[0]
        category_counts[cat_id] = category_counts.get(cat_id, 0) + len(db.load_batch_data(batch_id))

    with open(summary_file, "w") as f:
        json.dump({
            "agent_id": agent_id,
            "total_examples": len(all_examples),
            "total_batches": len(batch_ids),
            "compiled_at": datetime.now().isoformat(),
            "by_category": category_counts,
        }, f, indent=2)

    print(f"Summary: {summary_file}")


# ============================================================
# STATUS
# ============================================================

def show_status(agent_id: str, config: dict):
    validate_agent_id(agent_id)
    paths = agent_paths(agent_id)
    if not paths["taxonomy"].exists():
        print(f"No taxonomy for agent '{agent_id}'. Run the bridge first.")
        return

    db = ProgressDB(paths["db_dir"])
    with open(paths["taxonomy"]) as f:
        taxonomy = json.load(f)

    all_batches = plan_batches(taxonomy, config["batch_size"])
    overall = db.get_stats()

    print(f"\n{'='*65}")
    print(f"  SIMIC GENERATION STATUS — {agent_id}")
    print(f"{'='*65}")
    print(f"  Total pairs generated: {overall['total_pairs_generated']}")
    print(f"  Batches done/total: {overall['done']}/{len(all_batches)}")
    print(f"  Failed: {overall['failed']}\n")

    print(f"  {'Category':<42} {'Done':>6} {'Target':>8} {'Status':>10}")
    print(f"  {'-'*42} {'-'*6} {'-'*8} {'-'*10}")

    for category in taxonomy["categories"]:
        cat_id = category["id"]
        target = category["target_pairs"]
        cat_batches = [b for b in all_batches if b["category_id"] == cat_id]
        done_count = sum(1 for b in cat_batches if db.is_batch_done(b["batch_id"]))
        pairs_done = sum(
            db.progress["batches"].get(b["batch_id"], {}).get("pair_count", 0)
            for b in cat_batches
        )
        if done_count == len(cat_batches):
            status = "done"
        elif done_count > 0:
            status = f"~{done_count}/{len(cat_batches)}"
        else:
            status = "pending"
        name_display = f"{cat_id} {category['name']}"
        if len(name_display) > 40:
            name_display = name_display[:40] + ".."
        print(f"  {name_display:<42} {pairs_done:>6} {target:>8} {status:>10}")

    print(f"{'='*65}\n")


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Simic Stage 3 — Fine-tune data generator")
    parser.add_argument("--agent-id", required=True, help="Agent identifier")
    parser.add_argument("--status", action="store_true", help="Show generation progress")
    parser.add_argument("--compile", action="store_true", help="Compile batches into JSONL")
    parser.add_argument("--reset", action="store_true", help="Reset ALL progress for this agent")
    parser.add_argument("--reset-category", type=str, help="Reset a specific category")

    parser.add_argument("--model", type=str, help="Override model name")
    parser.add_argument("--base-url", type=str, help="Override API base URL")
    parser.add_argument("--api-key-env", type=str, help="Override API key env var name")
    parser.add_argument("--concurrency", type=int, help="Override max concurrent requests")
    parser.add_argument("--rpm", type=int, help="Override requests per minute")
    parser.add_argument("--category", type=str, help="Run only a specific category (e.g. A1)")

    args = parser.parse_args()
    validate_agent_id(args.agent_id)

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

    if args.status:
        show_status(args.agent_id, config)
    elif args.compile:
        compile_jsonl(args.agent_id)
    elif args.reset:
        confirm = input(f"Delete ALL progress for '{args.agent_id}'? Type 'yes' to confirm: ")
        if confirm.strip().lower() == "yes":
            db = ProgressDB(agent_paths(args.agent_id)["db_dir"])
            db.reset_all()
            print("Reset.")
        else:
            print("Cancelled.")
    elif args.reset_category:
        db = ProgressDB(agent_paths(args.agent_id)["db_dir"])
        db.reset_category(args.reset_category)
        print(f"Category {args.reset_category} reset.")
    else:
        asyncio.run(run_generation(args.agent_id, config))


if __name__ == "__main__":
    main()
