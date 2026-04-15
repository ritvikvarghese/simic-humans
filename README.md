# Simic

Can an AI agent convincingly replicate a specific real person? simic humans is a multi-agent research project to see if it can be used to predict human choice and decisions for market research. 

Two methodologies for replicating a specific person as an AI agent - unified pipeline.

Given an interview transcript of a real human, Simic produces either:
1. **A prompt-engineered agent** (fast, cheap) - a memory file loaded as context at query time, or
2. **A synthetic dataset for fine-tuning** (slow, expensive) - a JSONL that trains a model to be the person at the weight level.

### Approach A - prompt + memory file
- **Stage 1 only.** Four expert personas (psychologist, consumer behavior, cultural-demographic, social network) analyze the transcript in parallel. Their observations are compressed into a system prompt and packaged with the transcript into a single memory file. `serve.py` loads memory files and exposes a FastAPI endpoint that sends `{system: memory_file, user: question}` to Claude.
- **Cost:** ~6 API calls per agent, ~1 minute, a few cents.

### Approach B - fine-tune on synthetic Q&A
- **Stages 1 + 2 + 3.** After genesis, a bridge script derives per-person fine-tune configs (system prompt, frontier inference prompt, 25-category generation taxonomy) from the memory file. A generator model (default: GLM-4.6 via Z.AI) then produces ~2,600 Q&A pairs across 25 behavioral categories. The output is an OpenAI-chat-format JSONL ready for fine-tuning.
- **Cost:** ~2,600+ API calls per agent, hours, dollars.

## Sample Transcripts

Five synthetic interview transcripts are bundled in `transcripts/` so you can run the pipeline end-to-end: 

| File | Persona |
|---|---|
| `Arjun_DSouza.md` | Freelance video editor & YouTuber, Bengaluru (via Mangaluru), 3-6L, Catholic |
| `Deepika_Kaur.md` | Product Marketing Manager at Zoho, Chennai, 15-25L, Sikh |
| `Mohammed_Irfan.md` | Operations Manager at Lenskart, Hyderabad, 12-18L, Muslim |
| `Priya_Sharma.md` | Junior Data Analyst at Razorpay, Bengaluru, 5-8L, Hindu |
| `Rahul_Nambiar.md` | Franchise owner (Chai Point) + tutoring centre, Kochi, 8-15L, Hindu (Nair) |

## Quickstart

```bash
# 1. Setup
git clone https://github.com/ritvikvarghese/simic-humans.git
cd simic-humans
pip install -r requirements.txt
cp .env.example .env      # add your ANTHROPIC_API_KEY (and ZAI_API_KEY if fine-tuning)

# 2. Produce a simagent (Approach A) - ~1 min, a few cents
python simic.py transcripts/Arjun_DSouza.md --agent-id arjun_dsouza

# 3. Query it
uvicorn serve:app --reload
curl -X POST localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "What would you do if you got a 2x raise?", "agent_ids": ["arjun_dsouza"]}'

# 4. Or produce fine-tune data (Approach B) - hours, $5-20 per agent
python simic.py transcripts/Arjun_DSouza.md --agent-id arjun_dsouza --finetune
python simic.py transcripts/Arjun_DSouza.md --agent-id arjun_dsouza --compile
# → output/arjun_dsouza/training_data.jsonl

# 5. Generate agents for all 5 samples at once (Approach A)
for f in transcripts/*.md; do
  id=$(basename "$f" .md | tr '[:upper:]' '[:lower:]')
  python simic.py "$f" --agent-id "$id"
done
# Then: curl localhost:8000/agents  ← lists all 5
# Or query all in parallel: curl -X POST localhost:8000/query -d '{"query":"..."}'
```

**Naming convention:** the `--agent-id` is arbitrary but must be `[a-zA-Z0-9_-]+`. Lowercase snake-case is conventional (e.g., `arjun_dsouza`), and makes the memory filename predictable: `memory/<agent_id>_<date>_memory.md`.

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

```bash
python simic.py <transcript> --agent-id <id> [options]

# Stage 1 only (default - Approach A)
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

```

Responses are brief (2-3 sentences, plain text) by design. For longer or structured answers, edit `serve.py`'s `BRIEF_INSTRUCTION` - the format-preset machinery was removed as speculative.

## Transcript Format

A transcript is a markdown file in any Q&A format. The only required structure: a `# Demographic intake:` section at the top. The pipeline uses it to extract structured profile metadata (name, age, occupation, city, etc.) as YAML frontmatter in the memory file.

Example:

```markdown
# Demographic intake:

Full name: Ritvik Varghese
Currently live in: Bengaluru, Karnataka
Hometown: Chennai, Tamil Nadu
How old are you: 25
Gender: male
Marital status: single
Occupation: senior product manager at a B2B SaaS company
Languages: English, Hindi, German, French

# Q1. Walk me through a typical weekday.

I wake up around 6:30, drink coffee while reading...
```
Place your own transcripts in `transcripts/`. The directory is tracked so you can commit synthetic samples alongside the code if you want them to ship with the repo.

## Directory Layout

```
simic-humans/
├── simic.py                    # Unified CLI (stages 1+2+3 wired together)
├── genesis.py                  # Stage 1: transcript → memory file
├── expert_prompts.py           # Prompts for the 4-expert panel + gaps + system prompt synth
├── bridge.py                   # Stage 2: memory file → per-agent fine-tune configs
├── generate.py                 # Stage 3: configs → JSONL fine-tune data (resumable, parallel)
├── serve.py                    # Stage 4: FastAPI over memory files (Approach A runtime)
├── prompts/
│   └── category_taxonomy_template.json  # 25-category structural template (bridge fills it in)
├── transcripts/                # User-provided transcripts (tracked - place samples here)
├── memory/                     # Generated agent memories (gitignored, created at runtime)
├── agent_configs/              # Per-agent bridge output (gitignored, created at runtime)
├── db/                         # Per-agent stage 3 progress + raw batches (gitignored, created at runtime)
├── output/                     # Per-agent stage 3 final JSONL (gitignored, created at runtime)
├── requirements.txt
├── .env.example
├── LICENSE
└── RESEARCH.md                 # Methodology writeup + why this direction was paused
```

## Research Context

See [RESEARCH.md](RESEARCH.md) for the methodology deep-dive: what the two approaches are trying to prove, what we learned, and why this direction is being paused.

## License

MIT - see [LICENSE](LICENSE).
