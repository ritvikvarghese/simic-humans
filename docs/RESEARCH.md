# Simic Research Notes

Can an AI agent convincingly replicate a specific real person? 
their decisions, voice, reasoning, and contradictions, can this be done from a single interview transcript? 

Simic was a multi-agent research project to test that and see if it can be used to predict human choice and decisions for market research.

## The two approaches

### Approach A - prompt engineering over a rich memory file

1. Take a 60–90 minute interview transcript. 
2. Run four expert personas in parallel (psychologist, consumer-behavior analyst, cultural-demographic analyst, social-network analyst). 
3. Each expert produces 8–18 grounded, inferential observations about the person.
4. Merge everything into a single memory file. 

At query time, load the memory file as the system prompt and send questions to Claude Sonnet. The model uses the memory to understand and answer questions you ask the agent. 

### Approach B - fine-tune on synthetic Q&A grounded in the memory file

Take the memory file from Approach A. A bridge script derives three person-specific artifacts from it:
1. A short **system prompt** suitable for JSONL training (used as the `system` role in every training example)
2. A rich **frontier inference prompt** with decision rules, answer-shape rules, voice rules, and behavioral anchors grouped by domain (spending, trust, career, family, etc.)
3. A populated **25-category taxonomy** where each category has category-specific behavioral anchors and a full generation prompt

A generator model (default: GLM-4.6 via Z.AI) runs over the taxonomy at ~2,600 calls. Each call uses the full memory file as system and the category prompt as user, producing 10 Q&A pairs (or 5 multi-turn chats for voice calibration) across set lengths, grounded in the person’s voice and patterns.

The result is a JSONL of ~2,600 training examples in OpenAI chat format, ready to upload to any fine-tuning endpoint. Once fine-tuned, the target model can respond as the person at inference time with no context tax.

## What you can use this for: 
1. Creating your own agent
2. use `serve.py`'s parallel-query endpoint to simulate a panel (20 agents from 20 interviews) answering the same market research question.

*Simic is a research artifact, maintained by Ritvik Varghese. Open-sourced as a reference implementation.*