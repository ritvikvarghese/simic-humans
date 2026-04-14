# Simic

Two methodologies for replicating a specific person as an AI agent — unified pipeline.

Given an interview transcript of a real human, Simic produces either:
1. **A prompt-engineered agent** (fast, cheap) — a memory file loaded as context at query time, or
2. **A synthetic dataset for fine-tuning** (slow, expensive) — a JSONL that trains a model to be the person at the weight level.

Both approaches take the same input (a transcript) and share Stage 1. You choose how deep to go.

## The Two Approaches

### Approach A — prompt + memory file
- **Stage 1 only.** Four expert personas (psychologist, consumer behavior, cultural-demographic, social network) analyze the transcript in parallel. Their observations are compressed into a system prompt and packaged with the transcript into a single memory file. `serve.py` loads memory files and exposes a FastAPI endpoint that sends `{system: memory_file, user: question}` to Claude.
- **Cost:** ~6 API calls per agent, ~1 minute, a few cents.
- **Strength:** easy to iterate, voice fidelity preserved, works zero-shot on any transcript.
- **Weakness:** context tax on every query, limited by the memory file quality, model is still Claude playing a role.

### Approach B — fine-tune on synthetic Q&A
- **Stages 1 + 2 + 3.** After genesis, a bridge script derives per-person fine-tune configs (system prompt, frontier inference prompt, 25-category generation taxonomy) from the memory file. A generator model (default: GLM-4.6 via Z.AI) then produces ~2,600 Q&A pairs across 25 behavioral categories. The output is an OpenAI-chat-format JSONL ready for fine-tuning.
- **Cost:** ~2,600+ API calls per agent, hours, dollars.
- **Strength:** no context tax at inference, voice baked into the weights, runs on your own infra.
- **Weakness:** only as good as the synthetic data quality, requires hosting the fine-tuned model, harder to iterate.

## Quickstart

```bash
# 1. Setup
git clone [this repo]
cd simic
pip install -r requirements.txt
cp .env.example .env      # add your ANTHROPIC_API_KEY (and ZAI_API_KEY if fine-tuning)

# 2. Produce a simagent (Approach A)
python simic.py transcripts/alex_chen.md --agent-id alex_chen

# 3. Query it
uvicorn serve:app --reload
curl -X POST localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "What would you do if you got a 2x raise?", "agent_ids": ["alex_chen"]}'

# 4. Or produce fine-tune data (Approach B)
python simic.py transcripts/alex_chen.md --agent-id alex_chen --finetune
python simic.py transcripts/alex_chen.md --agent-id alex_chen --compile
# → output/alex_chen/training_data.jsonl
```

## Pipeline Stages

```
transcript.md
    │
    ▼
[Stage 1: genesis.py]  ─────▶  memory/<agent_id>_<date>_memory.md
    │                           (YAML frontmatter + system prompt + transcript + expert notes)
    │
    ▼
[Stage 2: bridge.py]   ─────▶  agent_configs/<agent_id>/
    │                           ├── system_prompt.json
    │                           ├── frontier_approach_prompt.json
    │                           └── taxonomy.json (25 categories, person-specific)
    │
    ▼
[Stage 3: generate.py] ─────▶  output/<agent_id>/training_data.jsonl
                                (~2,600 Q&A pairs, OpenAI chat format)

[Stage 4: serve.py]   ─────▶  HTTP API over memory files (for Approach A)
```

## CLI Reference

### Unified entrypoint

```bash
python simic.py <transcript> --agent-id <id> [options]

# Stage 1 only (default — Approach A)
python simic.py transcripts/alex.md --agent-id alex

# All three stages (Approach B)
python simic.py transcripts/alex.md --agent-id alex --finetune

# Stages 1+2 only (produce configs, don't generate 2600 pairs yet)
python simic.py transcripts/alex.md --agent-id alex --bridge-only

# Test one category during stage 3
python simic.py transcripts/alex.md --agent-id alex --finetune --category A1

# Compile existing stage 3 batches into final JSONL
python simic.py transcripts/alex.md --agent-id alex --compile

# Parse transcript, show stats, no API calls
python simic.py transcripts/alex.md --agent-id alex --dry-run
```

### Per-stage scripts (finer control)

```bash
python genesis.py transcripts/alex.md --agent-id alex              # Stage 1
python bridge.py --agent-id alex                                    # Stage 2
python generate.py --agent-id alex                                  # Stage 3
python generate.py --agent-id alex --status                         # Stage 3 progress
python generate.py --agent-id alex --compile                        # Stage 3 compile
python generate.py --agent-id alex --reset                          # Stage 3 reset
python generate.py --agent-id alex --model "anthropic/claude-sonnet-4-5" \
    --base-url "https://api.anthropic.com/v1/messages" \
    --api-key-env ANTHROPIC_API_KEY                                 # Use Claude as generator
```

### Serve API

```bash
uvicorn serve:app --reload

# Health + agent list
curl localhost:8000/health
curl localhost:8000/agents

# Query one or more agents
curl -X POST localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "How do you decide what to eat?", "agent_ids": ["alex"]}'

# With format preset (brief | detailed | structured | quantitative)
curl -X POST localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "Rate how much you trust social media ads 1-10", "format": "quantitative"}'
```

## Transcript Format

A transcript is a markdown file in any Q&A format. The only required structure: a `# Demographic intake:` section at the top. The pipeline uses it to extract structured profile metadata (name, age, occupation, city, etc.) as YAML frontmatter in the memory file.

Example:

```markdown
# Demographic intake:

Full name: Alex Chen
Currently live in: San Francisco, California
Hometown: Taipei, Taiwan
How old are you: 32
Gender: male
Marital status: single
Occupation: senior product manager at a B2B SaaS company
Languages: English, Mandarin

# Q1. Walk me through a typical weekday.

I wake up around 6:30, drink coffee while reading...
```

**What makes a good transcript:**
- Rich voice signal — verbal tics, how they actually talk, natural flow
- Specific details — named brands, named people, amounts, dates, incidents
- Contradictions — where stated beliefs and actual behavior diverge (experts latch onto these)
- Depth over breadth — 60-90 min conversation is better than a wide survey

See `transcripts/` for synthetic samples demonstrating the expected format.

## Directory Layout

```
simic/
├── simic.py                    # Unified CLI (stages 1+2+3 wired together)
├── genesis.py                  # Stage 1: transcript → memory file
├── expert_prompts.py           # Prompts for the 4-expert panel + gaps + system prompt synth
├── bridge.py                   # Stage 2: memory file → per-agent fine-tune configs
├── generate.py                 # Stage 3: configs → JSONL fine-tune data (resumable, parallel)
├── serve.py                    # Stage 4: FastAPI over memory files (Approach A runtime)
├── history.py                  # Query history SQLite store
├── prompts/
│   └── category_taxonomy_template.json  # 25-category structural template (bridge fills it in)
├── transcripts/                # Synthetic sample transcripts (sealed, tracked)
├── memory/                     # Generated agent memories (gitignored)
├── expert_notes/               # Debug: raw expert observations (gitignored)
├── agent_configs/              # Per-agent bridge output (gitignored)
├── db/                         # Per-agent stage 3 progress + raw batches (gitignored)
├── output/                     # Per-agent stage 3 final JSONL (gitignored)
├── requirements.txt
├── .env.example
└── docs/
    └── RESEARCH.md             # Methodology writeup + why this direction was paused
```

## Research Context

See `docs/RESEARCH.md` for the methodology deep-dive: what the two approaches are trying to prove, what we learned, and why this direction is being paused.

## License

[TODO — add LICENSE file before going public]

## Credits

Simic unifies two prior research projects into a single pipeline:
- the prompt-engineered memory-file approach (Stage 1)
- the synthetic fine-tune dataset approach (Stages 2 + 3)
