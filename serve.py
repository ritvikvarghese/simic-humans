"""Simic API - serves pre-generated agent memory files as callable agents.

Loads memory files from memory/ at startup, exposes REST + SSE endpoints
for querying agents individually or all in parallel via the Claude API.

Usage:
    uvicorn serve:app --reload
    uvicorn serve:app --host 0.0.0.0 --port $PORT  # production

Auth:
    Set API_AUTH_TOKEN in .env to require Bearer token on /query.
    If unset, /query is open (local dev mode).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time

import yaml
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SONNET = "claude-sonnet-4-5"
MEMORY_DIR = Path("memory")
MAX_CONCURRENT = 25
QUERY_TIMEOUT = 60.0

BRIEF_INSTRUCTION = (
    "Answer in 2-3 sentences. Be direct, no preamble. "
    "Write in plain text only - no markdown, no headers, no bullet points, no bold or italic."
)

SCENARIO_GUARD = (
    "Important: If the question mentions unnamed people ('a friend', 'your partner', "
    "'someone you trust', 'a colleague'), treat them as generic - do not assume they are "
    "a specific person from your life unless explicitly named."
)

load_dotenv()

ALLOWED_ORIGINS = os.environ.get(
    "CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
).split(",")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Memory file parsing
# ---------------------------------------------------------------------------

AGENTS: dict[str, dict] = {}


def parse_agent_id(filename: str) -> tuple[str, str]:
    stem = filename.removesuffix("_memory.md")
    match = re.match(r"^(.+)_(\d{4}-\d{2}-\d{2})$", stem)
    if match:
        return match.group(1), match.group(2)
    return stem, ""


def parse_frontmatter(text: str) -> tuple[dict | None, str]:
    if not text.startswith("---"):
        return None, text
    end = text.find("\n---", 3)
    if end == -1:
        return None, text
    raw = text[3:end].strip()
    try:
        data = yaml.safe_load(raw)
        if isinstance(data, dict):
            remaining = text[end + 4:].lstrip("\n")
            return data, remaining
    except yaml.YAMLError:
        pass
    return None, text


def build_profile(agent_id: str, frontmatter: dict) -> dict:
    profile = {"agent_id": agent_id}
    profile.update(frontmatter)
    age = frontmatter.get("age", "")
    occupation = frontmatter.get("occupation", "")
    city = frontmatter.get("city", "")
    city_short = city.split(",")[0].strip() if city else ""
    parts = [str(age), occupation, city_short]
    profile["summary"] = ", ".join(p for p in parts if p)
    return profile


def load_memory_files() -> dict[str, dict]:
    """Scan memory/ and parse all agent memory files.

    Memory files must have YAML frontmatter (produced by genesis.py). Files
    without frontmatter are skipped with a warning.
    """
    agents = {}
    if not MEMORY_DIR.exists():
        log.warning("memory/ directory not found")
        return agents

    files = sorted(MEMORY_DIR.glob("*_memory.md"))
    if not files:
        log.warning("No memory files found in memory/")
        return agents

    candidates: dict[str, tuple[str, Path]] = {}
    for path in files:
        agent_id, date_str = parse_agent_id(path.name)
        if agent_id not in candidates or date_str > candidates[agent_id][0]:
            candidates[agent_id] = (date_str, path)

    for agent_id, (_date_str, path) in candidates.items():
        text = path.read_text(encoding="utf-8")
        frontmatter, text_without_fm = parse_frontmatter(text)
        if not frontmatter:
            log.warning(f"Skipping {path.name}: no YAML frontmatter. Re-run genesis.py to regenerate.")
            continue

        marker = "# System Prompt"
        idx = text_without_fm.find(marker)
        memory = text_without_fm[idx:] if idx != -1 else text_without_fm

        profile = build_profile(agent_id, frontmatter)
        agents[agent_id] = {"memory": memory, "profile": profile}
        log.info(f"Loaded {agent_id}: {profile.get('name', agent_id)} ({len(memory):,} chars)")

    log.info(f"Total agents loaded: {len(agents)}")
    return agents


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

client: Optional[AsyncAnthropic] = None
semaphore: Optional[asyncio.Semaphore] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global AGENTS, client, semaphore
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("ANTHROPIC_API_KEY not set - queries will fail")
    client = AsyncAnthropic(max_retries=2)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    AGENTS = load_memory_files()
    yield
    if client is not None:
        await client.close()


app = FastAPI(title="Simic API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

AUTH_TOKEN = os.environ.get("API_AUTH_TOKEN")


def check_auth(request: Request):
    if not AUTH_TOKEN:
        return
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {AUTH_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid or missing auth token")


class QueryRequest(BaseModel):
    query: str = Field(..., max_length=10000)
    agent_ids: Optional[list[str]] = None


@app.get("/health")
async def health():
    return {
        "status": "ok" if AGENTS else "no_agents",
        "agents_loaded": len(AGENTS),
    }


@app.get("/agents")
async def list_agents():
    return [agent["profile"] for agent in AGENTS.values()]


@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    agent = AGENTS.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return agent["profile"]


@app.post("/query")
async def query_agents(req: QueryRequest, request: Request):
    check_auth(request)

    if not AGENTS:
        raise HTTPException(status_code=503, detail="No agents loaded")

    user_message = f"{BRIEF_INSTRUCTION}\n\n{SCENARIO_GUARD}\n\n{req.query}"

    if req.agent_ids:
        missing = [aid for aid in req.agent_ids if aid not in AGENTS]
        if missing:
            raise HTTPException(status_code=404, detail=f"Agents not found: {missing}")
        targets = {aid: AGENTS[aid] for aid in req.agent_ids}
    else:
        targets = AGENTS

    async def run_one(agent_id: str, agent_data: dict) -> dict:
        async with semaphore:
            t0 = time.monotonic()
            try:
                response = await asyncio.wait_for(
                    client.messages.create(
                        model=SONNET,
                        max_tokens=4000,
                        system=[{
                            "type": "text",
                            "text": agent_data["memory"],
                            "cache_control": {"type": "ephemeral"},
                        }],
                        messages=[{"role": "user", "content": user_message}],
                    ),
                    timeout=QUERY_TIMEOUT,
                )
                latency = int((time.monotonic() - t0) * 1000)
                cached = response.usage.cache_read_input_tokens or 0
                log.info(f"  {agent_id}: {response.usage.input_tokens} in / {response.usage.output_tokens} out (cached: {cached})")
                return {
                    "agent_id": agent_id,
                    "name": agent_data["profile"]["name"],
                    "response": response.content[0].text.strip(),
                    "status": "success",
                    "latency_ms": latency,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cached_tokens": cached,
                }
            except asyncio.TimeoutError:
                latency = int((time.monotonic() - t0) * 1000)
                log.error(f"  {agent_id} timed out after {QUERY_TIMEOUT}s")
                return {
                    "agent_id": agent_id,
                    "name": agent_data["profile"]["name"],
                    "response": "",
                    "status": "error",
                    "error": "Query timed out",
                    "latency_ms": latency,
                }
            except Exception as e:
                latency = int((time.monotonic() - t0) * 1000)
                log.exception(f"  {agent_id} failed: {e}")
                return {
                    "agent_id": agent_id,
                    "name": agent_data["profile"]["name"],
                    "response": "",
                    "status": "error",
                    "error": "Agent query failed",
                    "latency_ms": latency,
                }

    async def event_stream():
        tasks = [asyncio.create_task(run_one(aid, adata)) for aid, adata in targets.items()]
        succeeded = 0
        failed = 0

        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result["status"] == "success":
                succeeded += 1
            else:
                failed += 1
            yield f"event: agent_response\ndata: {json.dumps(result)}\n\n"

        summary = {"total": len(targets), "succeeded": succeeded, "failed": failed}
        yield f"event: done\ndata: {json.dumps(summary)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
