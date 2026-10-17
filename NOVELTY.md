# Finding the Novelty: Research Findings

As of Sat Oct 17, 2026, about 1:20 PM PDT. Four parallel research passes. Every fact links to a page an agent opened; anything else is marked unverified.

## Recommendation

**Target the multiwide maximum-overhang problem (Paterson and Zwick).** Published optima exist for n = 2 to 20 and n = 30. A balance check runs in milliseconds as a linear program. As far as we found, no LLM system has attacked this version: AlphaEvolve did only the one-block-per-level variant, and that variant's answer is the textbook harmonic stack.

**Claim we can make, if it works:** "A 3B-active open model, run by its own agent team on Crusoe, rediscovers the counterweighted Paterson and Zwick stacks, which beat the textbook harmonic stack by up to 29% at n = 20."

**Runner-up:** FloorplanQA (ICML 2026), if the team would rather hill-climb a named benchmark. Details are below.

## Constraints that shape everything (Crusoe facts)

| Fact | Source |
|---|---|
| **Nemotron 3 Nano 30B-A3B is not fine-tunable.** The only fine-tunable Nemotron is `nvidia/nemotron-3.5-lightning-30b-a3b`. | [available models](https://docs.crusoecloud.com/serverless-fine-tuning/available-models) |
| Fine-tuning is LoRA only and supervised only. Data is JSONL or Parquet with a `messages` array; loss is on assistant turns. Lightning costs $0.40 per million training tokens. | [guide](https://docs.crusoecloud.com/serverless-fine-tuning/fine-tune-a-model), [pricing](https://www.crusoe.ai/cloud/pricing) |
| Crusoe's own blog: "Fine-tuning jobs run for hours". Self-Serve Deployment takes "up to 40 minutes". | [blog](https://www.crusoe.ai/resources/blog/a-practical-guide-to-serverless-fine-tuning), [quickstart](https://docs.crusoecloud.com/self-serve-deployments/quickstart) |
| Fine-tuning API is OpenAI-compatible at `https://api.intelligence.crusoecloud.com/v1`. The base model takes an ID from `models.list()`. | [guide](https://docs.crusoecloud.com/serverless-fine-tuning/fine-tune-a-model) |
| Inference base URL is `https://api.inference.crusoecloud.com/v1`. Nemotron IDs: `nvidia/Nemotron-3-Nano-30B-A3B`, `nvidia/Nemotron-3-Super-120B-A12B`, `nvidia/nemotron-3.5-lightning-30b-a3b`. Ultra's ID is not documented. | [serverless inference](https://docs.crusoecloud.com/serverless-inference) |
| **Rate limits: 30 requests/min per model without a payment method, 600 with one.** | [rate limits](https://docs.crusoecloud.com/serverless-inference/rate-limits) |
| Foundry may need approval through `console.crusoecloud.com/request-foundry`. | [llms.txt](https://www.crusoe.ai/llms.txt) |

**What this means:**
- **One fine-tuning round at most.** The headline cannot depend on repeated fine-tuning.
- **Self-improvement has to happen mainly at inference time.** One LoRA round on Lightning can be a bonus.
- **Add a payment method now, or ask for hackathon credits.** At 30 requests/min, a round of 2,400 attempts takes 80 minutes.

### Questions for the Crusoe table

1. Can Nano be fine-tuned, or only Lightning?
2. How long is the queue plus training for about 1,000 rows?
3. Is Foundry self-serve today, or does it need approval?
4. Which API field attaches a checkpoint to a deployment, and what is `flavor_id`?
5. Can a deployment come up in less than 40 minutes?
6. Are there hackathon credits?
7. Can an adapter be used on per-token serverless inference, without a dedicated deployment?
8. What is Ultra's model ID?

## Option A: multiwide maximum overhang (recommended)

**The problem.** Stack n identical blocks on a table so the top reaches as far past the edge as possible. Blocks may sit side by side, and counterweights are allowed. The model is 2D, with frictionless blocks of length 1 and weight 1.

**Records** ([SODA paper](https://www.cs.tau.ac.il/~zwick/papers/overhang-SODA.pdf), [arXiv](https://arxiv.org/pdf/0710.2357)):

| n | Paterson and Zwick optimum | Harmonic (textbook) | Gain |
|---|---|---|---|
| 10 | 1.78713 | 1.464 | +22% |
| 20 | 2.32014 | 1.799 | +29% |
| 30 | 2.70909 | 1.997 | +36% |

- The n = 20 optimum is **non-spinal** (2.32014). The best spinal stack reaches 2.31899, and spinal stacks are optimal only for n ≤ 19.
- Optimal "loaded" stacks are only "believed to be close to optimal" (weight 100: 4.23897). That is a stretch goal where we could set a genuinely new record.

**Verifier.** Balance is LP feasibility over the contact forces: per block, the vertical forces and torques must balance, and every force must be non-negative ([Paterson and Zwick §2 and §5](https://ar5iv.labs.arxiv.org/html/0710.2357)). With scipy `linprog` this takes milliseconds. **Do not verify in PyBullet:** optimal stacks sit exactly on the edge of balance, so friction and contact tolerance make a rigid-body sim unreliable there (inferred). Use PyBullet only to render the demo, with a small safety margin.

**Prior work and what we must not claim** ([AlphaEvolve problems paper §14](https://arxiv.org/html/2511.02864), [ThetaEvolve](https://arxiv.org/html/2511.23473), [EinsteinArena](https://arxiv.org/html/2606.10402)):
- "First LLM on block stacking": AlphaEvolve did the one-on-one variant and rediscovered ½·H_n.
- "First evolutionary search on overhang": Hohm 2007 did it without an LLM (unverified; from a search snippet).
- "First small or open model to set a math record": ThetaEvolve did it with an 8B model on circle packing and autocorrelation, and TTT-Discover with gpt-oss-120b.
- "Beats the harmonic stack" as news: that is Ainley 1979 and Paterson and Zwick 2006.
- "Proved optimal": Paterson and Zwick's optima are numerical, so any match is a numerical match.
- Any record on circle packing (n = 26), autocorrelation, kissing numbers, Heilbronn, Thomson, Tammes or hexagon packing: these are already taken.

**Disclosure.** If an LLM proposes the stack structure and a numerical optimizer polishes the block positions, say so. Paterson and Zwick found their optima the same way: enumerate structures, then optimize numerically.

**Feasibility.** n ≤ 10 is likely in 3 hours. n = 20 is hard.

**Why it wins.**
- The records are real, not invented.
- Stacks with counterweights look strange, and the difference from the textbook stack is visible at a glance.
- It works with no fine-tuning, so Crusoe's turnaround can't break it.
- It fits SA's dashboard exactly: an open-ended leaderboard benchmark with an outside verifier.

## Option B: FloorplanQA (runner-up)

- **Links:** [paper](https://arxiv.org/abs/2507.07644v3) (ICML 2026), [code](https://github.com/OldDeLorean/FloorplanQA).
- **Task:** room layouts as JSON with geometric questions (distance, placement, free space, max box, shortest path). Answers are computed exactly from the geometry.
- **Data:** 1,800 synthetic layouts and 200 semi-real ones. Its generators can make unlimited training questions.
- **Scores:** 75 to 95% on synthetic layouts falls to 35 to 60% on real ones, averaged across models. Max Box and Shortest Path are the hardest.
- **Prior work:** no fine-tuning result found.
- **Claim:** "First self-improvement result on FloorplanQA: trained only on synthetic layouts, tested on held-out real ones."
- **Weakness:** it leans on fine-tuning, which Crusoe limits today. Mac install unverified.

## Rejected targets

| Target | Reason |
|---|---|
| [NewtonBench](https://arxiv.org/html/2510.07172v2) | Each task is a multi-turn agent run, too slow for today. Its LLM judge defaults to Nemotron Ultra, which is nice but not enough |
| SnorkelSpatial | Archived, and GPT-5.4 is at 99% |
| DecompSR | Extends StepGame |
| GSU | Already LoRA-tuned |
| PHYRE | Linux-only wheels, Python ≤3.9 |
| PIQA | Saturated |
| Vision-only benchmarks | Need a vision model |

## Twists to the loop itself (secondary)

The most novel twist found is the model **predicting simulator outcomes and using those predictions to pick its own next tasks**, then taking over task selection from the larger model once it predicts better. Prior work:
- [Early Experience](https://arxiv.org/abs/2510.08558)
- [WorldLLM](https://arxiv.org/abs/2506.06725)
- [R-Zero](https://arxiv.org/abs/2508.05004)
- [Multi-Agent Evolve](https://arxiv.org/abs/2510.23595)

It needs several fine-tuning rounds, so it doesn't fit today. Proposer/solver self-play would read as "Absolute Zero for blocks" ([Absolute Zero](https://arxiv.org/abs/2505.03335)).
