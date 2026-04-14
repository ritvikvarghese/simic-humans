# AGENTS.md

Shared rules for any agent (Claude Code, Codex, etc.) working in this repo. For humans, start with [README.md](README.md). For research context, see [docs/RESEARCH.md](docs/RESEARCH.md).

## What this is

Simic is a three-stage Python pipeline that turns an interview transcript into an AI agent (prompt-engineered or fine-tuned). Archived research artifact — open-source, not actively maintained.

- **Stage 1** (`genesis.py`) — transcript → memory file via 4-expert panel (Anthropic Claude Sonnet)
- **Stage 2** (`bridge.py`) — memory file → per-agent fine-tune configs (2 parallel Claude calls)
- **Stage 3** (`generate.py`) — configs + memory → JSONL fine-tune dataset (~2,600 calls via Z.AI GLM-4.6 by default)
- **Runtime** (`serve.py`) — FastAPI over memory files (Approach A only; does not serve fine-tuned models)

Unified CLI: `simic.py`.

## How to verify

No API key is required for these checks. Run them before touching anything.

```bash
# 1. All Python files compile
python3 -m py_compile simic.py genesis.py bridge.py generate.py serve.py expert_prompts.py

# 2. CLI help renders
python3 simic.py --help

# 3. Dry-run parses a transcript end-to-end (no API calls)
python3 simic.py transcripts/Arjun_DSouza.md --agent-id arjun_dsouza --dry-run

# 4. agent_id validator rejects path traversal
python3 simic.py bogus.md --agent-id "alex/../evil" --dry-run   # should exit 1
```

Full live run (requires `ANTHROPIC_API_KEY`, ~1 min, a few cents): drop the `--dry-run` flag on step 3.

## Conventions

**agent_id** — must match `^[a-zA-Z0-9_-]+$`. Validated at CLI entry in every stage. Lowercase snake-case (e.g., `arjun_dsouza`) is conventional.

**Taxonomy placeholder protocol** — bridge-generated `prompt` strings in `agent_configs/<agent_id>/taxonomy.json` MUST contain two literal placeholders:
- `{count}` — replaced at Stage 3 with per-batch pair count
- `{batch_info}` — replaced at Stage 3 with anti-repetition text for later batches

Both `bridge.assemble_taxonomy` and `generate.plan_batches` hard-fail if either placeholder is missing. Don't hand-edit taxonomy prompts unless you preserve both placeholders.

**Memory file schema** — `memory/<agent_id>_<date>_memory.md` is the handoff artifact between stages. Structure:
1. YAML frontmatter (demographic profile)
2. `# System Prompt` section (synthesized persona)
3. `# Interview Transcript` section (raw)
4. `# Expert Observations` section (4 experts)
5. `## Coverage Gaps & Conflicts` section (optional)

`serve.py` skips files missing YAML frontmatter — re-run genesis if that happens.

**Bridge regenerates configs deterministically** — `agent_configs/<agent_id>/` is bridge output, not hand-maintained config. Re-running `bridge.py --agent-id X` will overwrite.

## What not to touch

**`prompts/category_taxonomy_template.json`** — the 25 category IDs (A1–K1), `target_pairs` values, and `response_length_distribution` are a stable contract the bridge relies on. Bridge assumes these exactly. Don't rename, merge, or drop categories without also updating `bridge.py`'s validation logic.

**Expert lane-keeping rules in `expert_prompts.py`** — each expert has explicit "DO NOT write about" boundaries to prevent cross-expert redundancy. Removing them produces four experts that all say the same thing.

**`generate.py` resumability** — `db/<agent_id>/progress.json` tracks batch completion. Don't refactor `ProgressDB` without preserving this contract — a mid-run crash needs to resume from where it stopped.

## Costs to be aware of

- Stage 1 — ~6 Claude Sonnet calls per agent (~$0.05–0.15)
- Stage 2 — 2 Claude Sonnet calls per agent (~$0.10–0.30)
- Stage 3 — ~2,600 GLM-4.6 calls per agent (~$5–20, takes hours)

Never run Stage 3 in a test loop.

## Secrets

- `ANTHROPIC_API_KEY` — Stages 1, 2, and serve.py
- `ZAI_API_KEY` — Stage 3 (default generator)
- `API_AUTH_TOKEN` — optional Bearer token for serve.py `/query`

All via `.env` (gitignored). `.env.example` is the template.

## When making changes

- Use the Edit tool on a specific file. Avoid wholesale rewrites of `genesis.py` / `bridge.py` / `generate.py` — they share implicit contracts (memory file schema, taxonomy placeholders, `agent_id` validation) that are easy to break.
- After any change to Python files, run the four verification steps above.
- For bridge or taxonomy changes, unit-test `plan_batches()` logic with a synthetic taxonomy before committing (see the placeholder validation flow).
- The repo is archived — don't add new features, dependencies, or abstractions unless the user explicitly asks.
