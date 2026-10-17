# 60-second video

## Theme

**Optimizing every layer of an AI data center, with each layer framed as an optimization problem that code can check.**

| Layer | Case study | Verifier (no LLM) | In the video |
|---|---|---|---|
| 1. Hall layout | GPU data-hall layout (DoneTwin) | Rikin's scorer, 0 to 100 | **Hero**, 20 s |
| 2. Equipment placement | QAPLIB facility layout (quadratic assignment) | Exact assignment cost vs the best-known answer | 8 s |
| 3. GPU kernels | KernelBench mini, 12 Level-1 CUDA problems on a Crusoe L40S | Correctness and speed vs PyTorch, with reward-hack guards | 7 s |
| 4. Inference | Data-center operations questions with code-checked exact answers (`benchmarks/inference_opt/`) | Exact-answer check plus measured Crusoe cost | 10 s |

The hall is the hero for three reasons:
- It is Crusoe's own business.
- It has a 3D twin to show on screen.
- It has the only finished live run.

## Script (about 150 spoken words)

| Time | Layer | Spoken |
|---|---|---|
| 0:00 to 0:07 | Intro | "An AI data center is a stack of optimization problems: where the racks go, where the equipment goes, the kernels on the GPUs, and the inference bill. Code can check every one." |
| 0:07 to 0:15 | Harness | "Our harness turns each into a Nemotron team on Crusoe that proposes changes to itself and keeps only what a verifier measures." |
| 0:15 to 0:35 | 1. Hall layout | "Hall layout. Our cold team starts at 71 out of 100. In run one, it proposed 9 changes to itself, and the gate rejected all 9. Then we tested settings. The only one that beat the cold start was the small model allowed to think." **Outcome A:** "In run two the team turned reasoning on by itself, and [N_HALLS] unseen halls went from 71 to [RUN2_SCORE]." **Outcome B:** "In run two, the gate still found nothing worth keeping on unseen halls." |
| 0:35 to 0:43 | 2. Equipment | "Equipment placement, same harness: QAPLIB, [QAP_RESULT]." |
| 0:43 to 0:50 | 3. Kernels | "GPU kernels: 12 CUDA problems with reward-hack guards, built and ready for a Crusoe GPU." (Not run: our project has zero GPU quota.) |
| 0:50 to 1:00 | 4. Inference | "And serving: the team re-routes its own Nemotron calls, keeping answers right at [INF_COST_PCT]% of the cost. Bring a verifier. Get a team that keeps only what it measures." |

**Placeholders to fill from `reports/RESULTS.md` before recording:**

| Placeholder | Example wording | Fallback if nothing is measured |
|---|---|---|
| [RUN2_SCORE] | "73.5" | Use Outcome B |
| [QAP_RESULT] | "from [QAP_V0] to [QAP_VN] of the best-known answers on 6 unseen instances" | "the gate kept nothing that did not measure better" |
| [INF_COST_PCT] | "40" | "the gate cut [INF_CALLS] calls per question" if accuracy held. If no version passed: "the gate kept the expensive version, because nothing cheaper held accuracy" |

## Shot list

| Time | On screen | Source | Exists? |
|---|---|---|---|
| 0:00 to 0:07 | Four stacked layer cards: hall, equipment, kernels, inference | Slide | To make |
| 0:07 to 0:15 | Loop diagram (research agent, paper, verifier rerun, gate), then one paper's claim JSON | Slide, plus a version ledger in an editor | Ledger yes, diagram to make |
| 0:15 to 0:22 | DoneTwin control room: v0 cold's layout, score 71, resilience 0/20 | `donetwin/ui/index.html` at http://127.0.0.1:8000 | Yes |
| 0:22 to 0:28 | Terminal: the 9 rejected rows from run 1, then the settings table with the one winning row highlighted (Nano, reasoning on, 32k: +2.66) | `reports/datahall_live_1/version_ledger.jsonl`, `reports/RESULTS.md` | Yes |
| 0:28 to 0:35 | Outcome A: the 3D twin with vN's layout, plus a bar chart of v0 cold vs vN on 15 held-out halls, with greedy as a gray reference bar. Outcome B: the run 2 ledger | DoneTwin UI; chart from run 2's `before_after.json` | Chart to make |
| 0:35 to 0:43 | SA's dashboard, Benchmarks tab: the QAPLIB entry, v0 cold vs final, with 2-swap as a gray reference | http://127.0.0.1:5173 | Once run 2 publishes |
| 0:43 to 0:50 | The 12 problem names and the reward-hack guard list | `benchmarks/kernelbench_mini/` | Depends on the VM |
| 0:50 to 1:00 | Bar chart: $ per question, v0 (Super, reasoning on) vs vN, with accuracy above each bar. End card: tagline, "Crusoe Managed Inference + NVIDIA Nemotron" | Chart from the inference_opt run's `before_after.json`; slide | To make |

## Baselines

Every headline gain is **v0 cold to final** on held-out tasks: the same team before and after it edits itself. Greedy (hall) and 2-swap (QAPLIB) appear only as a gray "no-LLM reference" row or bar, and are never narrated.

## What is cut, and why

| Cut | Why |
|---|---|
| ALE-Bench | Not a data-center layer. Its run was stopped |
| SA's research harness (You.com researcher, attempt agent) | A second system in 60 seconds splits attention. It belongs in the README and the Q&A |
| Shared Brain internals (dedup, disputes, pruning) | Too fine-grained for video. The one memory fact that matters, 15% more tokens for no gain in run 1, is in `DEMO.md` |
| Per-layer numbers beyond one line each | Only the hall gets a before/after. The other three get one measured line each |
| The 2D model's limits | Stated in `DEMO.md` and the README, not in narration |
