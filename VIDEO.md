# 60-second video

## Theme

**Every layer of an AI data center is an optimization problem that code can score.**

| Layer | Case study | In the video |
|---|---|---|
| Where the GPU racks go | GPU data-hall layout (DoneTwin) | **Hero**, 40 s |
| Where the equipment goes | QAPLIB facility layout (quadratic assignment, Koopmans and Beckmann 1957) | One line, "same harness" proof |
| What runs on the GPUs | KernelBench mini (12 Level-1 CUDA problems) | One line, built but not yet run |
| What it costs on Crusoe | Inference cost per verified point | Woven through the hero beat, not its own beat |

## Why the data hall is the hero

- It is Crusoe's own business.
- It has a 3D twin to show on screen.
- It is the only benchmark with a finished live run.

## Script (about 150 spoken words)

| Time | Spoken |
|---|---|
| 0:00 to 0:08 | "Every layer of an AI data center is an optimization problem that code can score: where the racks go, where the equipment goes, what runs on the GPUs." |
| 0:08 to 0:20 | "Our harness turns each one into a Nemotron team on Crusoe that edits itself. A research agent proposes a change with a predicted gain. A verifier reruns it. No LLM decides what sticks." |
| 0:20 to 0:34 | "Our hero: GPU data halls. Greedy scores 71 out of 100. In run one, the team proposed 9 changes and the gate rejected all 9. None measured a gain, and memory cost 15% more tokens for nothing." |
| 0:34 to 0:47 | **Outcome A:** "Run two switched to [RUN2_CONFIG]. On 15 unseen halls: 71 to [RUN2_SCORE], at [RUN2_COST_PER_POINT] per point." **Outcome B:** "Run two tried [RUN2_CONFIG]. Still 71 on 15 unseen halls, and the gate shipped nothing it couldn't measure." |
| 0:47 to 1:00 | "Same harness, no new core code: QAPLIB facility layout, with 16 records still open. CUDA kernels are next, on a Crusoe GPU. Bring a verifier. Get a team that keeps only what it measures." |

Word count: about 150 with either outcome. Fill in the brackets from `reports/datahall_live_2/` (or wherever run 2 writes) before recording.

## Shot list

| Time | On screen | Source | Exists? |
|---|---|---|---|
| 0:00 to 0:08 | Title card listing the layers: racks, equipment, kernels, cost | Slide | To make |
| 0:08 to 0:20 | Loop diagram (research agent, paper, verifier rerun, gate), then one paper's JSON: change, `claim.dev_gain`, `claim.cost_change_pct` | Slide, plus `reports/datahall_live_1/version_ledger.jsonl` in an editor | Ledger yes, diagram to make |
| 0:20 to 0:27 | DoneTwin control room: greedy's layout for one hall, score 71, resilience 0/20 | `donetwin/ui/index.html` ("DoneTwin · Control room") | Yes |
| 0:27 to 0:34 | Terminal: the 9 ledger rows, each `accepted: false` with its reason (7 "gain +0.000 < 0.05" or "-0.001", 2 "step longer than 280 chars"). Cut to `before_after.md` showing tokens 223,921 vs 257,007 | `reports/datahall_live_1/version_ledger.jsonl`, `before_after.md` | Yes |
| 0:34 to 0:47 | Bar chart: greedy vs v0 vs vN on 15 held-out halls, with $ per point. Outcome A: the 3D twin with vN's layout. Outcome B: the run 2 ledger, all rejected | Chart from `before_after.json`; DoneTwin UI | Chart to make |
| 0:47 to 0:53 | SA's dashboard, Benchmarks tab: the QAPLIB leaderboard | http://127.0.0.1:5173 | Needs a QAPLIB entry with a verified score |
| 0:53 to 1:00 | End card: tagline, plus "Crusoe Managed Inference + NVIDIA Nemotron" | Slide | To make |

## What is cut, and why

| Cut | Why |
|---|---|
| ALE-Bench AHC039 ("Purse Seine Fishing") | Off-theme: a fishing-net polygon is not a data-center layer. The live smoke scored 0.35x greedy, which needs more explanation than 3 seconds allows. Kept in `DEMO.md` as an off-theme generality test |
| QAPLIB live numbers | The evolve run crashed on a Crusoe 429 and is relaunching. The smoke gap (13.9% vs 2-swap's 0.0% on els19) is worse than baseline, so it is not shown. If a relaunched run beats 2-swap, swap the 0:47 line for its number |
| KernelBench results | None exist: no Crusoe GPU VM has been created. It appears only as "next" |
| SA's research harness (You.com researcher, attempt agent) | A second system in 60 seconds splits attention. It belongs in the README and the Q&A |
| Shared Brain internals (dedup, disputes, pruning) | Too fine-grained for video. The one memory fact shown is the honest one: it cost 15% more tokens for no gain |
| The 2D model's limits | Stated in `DEMO.md` and the README, not in narration |
