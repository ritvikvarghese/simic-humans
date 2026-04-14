# Simic — Research Notes

## The thesis

Can an AI agent convincingly replicate a specific real person — their decisions, voice, reasoning, and contradictions — from a single interview transcript? And if so, which is the better approach: load the person into the context window at query time, or bake them into the weights?

Simic tests both.

## The two approaches

### Approach A — prompt engineering over a rich memory file

Take a 60–90 minute interview transcript. Run four expert personas in parallel (psychologist, consumer-behavior analyst, cultural-demographic analyst, social-network analyst). Each expert produces 8–18 grounded, inferential observations about the person — not what they said, but what their answers reveal about how they work. Merge everything into a single memory file: YAML demographic frontmatter at top, a synthesized system prompt in the middle, the full transcript as evidence at the bottom, and the expert notes as an analytical layer.

At query time, load the memory file as the system prompt and send questions to Claude Sonnet. The model now has everything: the voice evidence (transcript), the distilled identity (system prompt), the behavioral patterns (expert notes). No training required.

**What the four experts do that a single expert could not:**
- **Psychologist** — core drives, self-awareness gaps, identity arc. Writes about the psyche, not about decisions.
- **Consumer Behavior Analyst** — mental accounting, price sensitivity, brand loyalty mechanisms. Writes about purchases and financial behavior.
- **Cultural-Demographic Analyst** — SEC classification, community identity, family hierarchy, the invisible cultural rules. Writes about the social operating system.
- **Social Network Analyst** — trust circles, influence sources, decision-flow diagrams. Writes about who shapes what and how information travels.

Each expert stays in their lane (enforced via explicit "DO NOT write about" boundaries in the prompts), which reduces redundancy. A coverage-gaps pass then identifies transcript thinness and cross-expert contradictions — the latter being the most predictive data points, because internal contradictions reveal which identity wins in which context.

### Approach B — fine-tune on synthetic Q&A grounded in the memory file

Take the memory file from Approach A. A bridge script derives three person-specific artifacts from it:
1. A short **system prompt** suitable for JSONL training (used as the `system` role in every training example)
2. A rich **frontier inference prompt** with decision rules, answer-shape rules, voice rules, and behavioral anchors grouped by domain (spending, trust, career, family, etc.)
3. A populated **25-category taxonomy** where each category has category-specific behavioral anchors and a full generation prompt

A generator model (default: GLM-4.6 via Z.AI — cheap at the 2,600-call scale this pipeline requires) is then run over the taxonomy. The system message for every call is the full memory file. The user message is the category's generation prompt, which asks for 10 Q&A pairs (or 5 multi-turn conversations for the voice-calibration-multiturn category) in a specific response-length distribution, grounded in the person's voice and real patterns.

The result is a JSONL of ~2,600 training examples in OpenAI chat format, ready to upload to any fine-tuning endpoint. Once fine-tuned, the target model can respond as the person at inference time with no context tax.

### The 25-category taxonomy

The categories are structured to cover a whole person:

| Domain | Categories | Target pairs |
|---|---|---|
| Purchases | A1 electronics, A2 food/grocery, A3 clothing/lifestyle, A4 SaaS/AI, A5 home/household, A6 transport, A7 brand loyalty | 710 |
| Finance | B1 investment, B2 spending/negotiation, B3 lending/trust | 280 |
| Work | C1 entrepreneurship/business, C2 career/work philosophy | 300 |
| Relationships | D1 family, D2 friendships/trust | 200 |
| Values | E1 politics/society, E2 religion/spirituality, E3 ethics/dilemmas | 260 |
| Life | F1 health/fitness, F2 daily routines | 160 |
| Worldview | G1 AI/technology/future, H1 identity/self-reflection | 200 |
| Voice | I1 casual calibration, I2 multi-turn conversations, J1 contradiction/edge cases | 450 |
| Facts | K1 biographical | 200 |

Categories are weighted by importance — entrepreneurship (200) gets much more training data than religion (60), reflecting behavioral signal density. Each category's generation prompt is tailored per-person by the bridge, so an electronics prompt for a 25-year-old entrepreneur differs from one for a 58-year-old retiree.

## Why these categories, in this structure

The taxonomy is opinionated. It assumes:
- **Behavior is domain-specific.** A person's financial logic and their food logic run on different rules. Training on a flat "be this person" prompt produces a bland composite; training on domain-tagged data forces the model to learn the seams.
- **Voice requires dedicated calibration.** I1 (200 casual exchanges), I2 (100 multi-turn conversations), J1 (150 contradictions) are ~17% of the dataset — pure voice scaffolding. Without these, the model sounds right on "what phone do you use" and robotic on "how's your day."
- **Biographical facts (K1) prevent hallucination drift.** 200 concrete Q&As grounded in the transcript keep the fine-tuned model from inventing new facts about its own life.

## Why both? The actual research question

The interesting question isn't "does this work" (it does, in both cases). It's **"which approach replicates a person better, and at what cost?"**

Hypothesized tradeoffs:
| | Approach A (prompt) | Approach B (fine-tune) |
|---|---|---|
| Setup time | ~1 minute | Hours + tuning |
| Setup cost | ~5¢ | ~$5–20 per agent |
| Per-query cost | High (full memory in context) | Low (weights baked in) |
| Voice fidelity | Moderate — model is playing a role | High — voice in weights |
| Fact fidelity | High — transcript is in context | Moderate — risk of drift on long convos |
| Iteration speed | Fast — edit prompt, re-run | Slow — retrain to change anything |
| Works for N agents | Linear — just load more memory files | Linear — train N models |

The fair comparison runs the fine-tuned model and the prompt+memory model against the same battery of questions and a blind human rater (or the real person). That comparison is what the `testapp/` folder in `simic-finetuned` was designed for — it wasn't ported into this unified repo because it had Mongo + RunPod dependencies that don't generalize, but the eval harness is the missing piece of this research.

## What worked

[TODO — rivar fills in]

- The 4-expert panel structure produced observations that surprised even the people being interviewed. Cross-expert conflict detection (e.g., "Psychologist says autonomy-seeking, Cultural-Demographic says family-compliant — which wins when?") surfaced the most predictive features.
- ...

## What didn't

[TODO — rivar fills in]

- The fine-tune pipeline's 25-category taxonomy was originally hardcoded to one person. Making it person-agnostic (via the bridge) was a late realization — the first version of simic-finetuned only worked for its author.
- ...

## Why this direction is being paused

[TODO — rivar fills in — the honest postmortem]

## Where this could go

[TODO — rivar fills in — or leave blank]

Possible next steps if someone picked this up:
- **Eval harness** — port testapp-style three-way comparison (fine-tuned vs. prompt+memory vs. frontier baseline) into this repo, with SQLite instead of Mongo.
- **Active elicitation** — use the coverage-gaps output from Stage 1 to drive a follow-up interview, iterating memory quality.
- **Multi-agent simulation** — use `serve.py`'s parallel-query endpoint to simulate a panel (20 agents from 20 interviews) answering the same market research question.
- **Evaluation metric** — not "does the agent sound right" but "does the real person rate the agent's answer as something they'd plausibly say?" Run against the real subject, not a blind rater.

## Credits & Inspiration

- The 4-expert pattern was inspired by [cite if applicable — or just list prior art in qualitative research and agent design]
- The 25-category taxonomy was developed iteratively during simic-finetuned

---

*Simic is a research artifact by @ritvikvarghese. Not maintained. Open-sourced as a reference implementation.*
