# Demo: optimizing every layer of an AI data center

**Theme:** optimizing every layer of an AI data center, with each layer framed as an optimization problem that code can check. Four layers:
1. Hall layout (DoneTwin)
2. Equipment placement (QAPLIB)
3. GPU kernels (KernelBench)
4. Inference (data-center operations questions)

The video plan is in `VIDEO.md`.

**Claim:** one harness turns any problem with a programmatic verifier into an agent team that improves itself. It is not a general solver: each domain still needs its own task generator, its own non-LLM verifier, and its own role prompts. Each benchmark below is a case study of the same harness.

**What a domain supplies:**
1. Tasks split into train, dev and held-out.
2. A verifier that returns a number.
3. Role prompts.

**What the harness supplies, unchanged:** memory admission, dispute tests, the claimed-vs-measured papers ledger, the score gate, the cost gate and the held-out report. That is `team/brain.py` (390 lines), `team/evolve.py` (547) and `team/crusoe.py` (482). Each domain plugs in through one function, `run_team(version, brain, task, llm) -> RunResult`. The data-hall adapter is 484 lines; the QAPLIB adapter is 204.

## The loop (SA's framing)

A research agent writes papers. A solver team runs the benchmark. A verifier checks each paper before anyone trusts it.

| SA's role | Our component | What it does |
|---|---|---|
| Research agent | Reviewer (`team/evolve.py`) | Writes a **paper**: a hypothesis, one typed change, a predicted dev gain and cost change, and the run ids it is based on. A paper without numbers is dropped |
| Solver | The domain's team (`team/datahall.py`, `benchmarks/qaplib/team_runner.py`) | Nemotron roles on Crusoe |
| Verifier | Gate plus the domain's scorer | Reruns the paired dev tasks with and without the change. The non-LLM scorer measures both |
| Peer review | LLM reviewer | May only narrow a claim's scope ("3 halls" does not support "all halls"). It never accepts a claim |

A change is one of: a prompt edit, a playbook rule, a config change (model per role, max tokens, retries, thinking on/off), a tool on/off, a role added or removed, or promoting memory into a playbook.

**Acceptance rule:** dev score rises by 0.05 or more with no lost passes, or cost falls by 15% or more at equal score. Memory disputes use the same rule: one dev run with each side hidden.

**What happens to accepted papers:** they enter the Shared Brain with their measurement attached (Shared Brain track). Proven ones graduate into role playbooks as a new team version and leave memory (Version N+1 track).

**Why a verifier and not an LLM judge decides:** a judge lets the team learn what sounds right. That is the documented false-success failure mode.

## Layer 1, hall layout: GPU data-hall layout (hero, live)

- **Verifier:** Rikin's DoneTwin scorer (`donetwin/donetwin/score.py`), no LLM. Score 0 to 100 = 70 for density + 20 for resilience to a single CDU failure + 10 for hose length.
- **Hard rules, which score 0 if broken:** 2N power on A/B busways, 1.2 m hot/cold aisles, keep-outs, hose reach, CDU capacity, room air budget.
- **Baseline:** v0 cold (the original team, empty memory), 71.0 on held-out. **No-LLM reference:** greedy, also 71.0, usually 0 of the 20 resilience points. v0 starts from greedy's layout, which is why the two match.
- **Splits:** currently train 8, dev 6, held-out 10 halls (`team/datahall.py`). Train and dev use disjoint seeds in 0..9999, held-out uses seeds 10000..10049. Run 1 used 12 / 8 / 15.
- **Live run 1** (`reports/datahall_live_1/`): every role on Nemotron Nano with reasoning off.
  - Held-out on 15 halls: greedy 0.710, v0 with empty memory 0.710, v0 with learned memory 0.710.
  - The Reviewer proposed 9 changes and the gate rejected all 9: 7 measured a gain of +0.000 or -0.001 against the required 0.05, and 2 failed validation (a playbook step over 280 chars).
  - 475 calls, 0 errors, $0.173.
  - Memory raised tokens from 223,921 to 257,007 (+14.8%) for no gain. The cost gate now penalizes that.
- **Settings experiment** (4 train halls, Layout role only, greedy 0.695):
  - Nano with reasoning on and a 32k-token cap: +2.66 points. The only win: 2 of 4 halls improved, by +5.6 and +5.0, at $0.018 and 739 s per hall.
  - Reasoning off never produced a feasible layout: 0 of 12 attempts, for both Nano and Super.
  - Super with reasoning on stalled: every call was cut off at 16k tokens.
  - Run-to-run noise: about ±1.3 points.
  - In one line: the only setting that beat the cold start was the small model allowed to think. The cold start, Nano with reasoning off, scored exactly greedy.
- **Runs 2+** (live, latest `datahall-live-4`): v0 is run 1's config. The Reviewer can turn Layout reasoning on, or enable a verifier-guided polish tool (`team/datahall_polish.py`, Rikin's local search). Result: [RUN2_SCORE]
- **Required ablation for any polish gain:** offline, greedy + polish with no LLM scores 72.75 vs greedy 68.79 on 6 held-out halls. If the final version's gain comes from enabling polish, it must be reported next to that row. The LLM's contribution is then choosing the tool and the settings, not the layout itself.

Both outcomes are reportable:
- **If the final version beats v0 cold:** the table below shows the gain on unseen halls.
- **If none does:** the ledger shows every change the gate refused, with claimed vs measured. "The gate refused every change that did not help" is Version N+1 working as designed. A team that edits itself without a gate would have shipped those changes.

| Config (15 held-out halls) | Score /100 | Pass rate | Calls/task | Tokens/task | $/task | $ per verified point |
|---|---|---|---|---|---|---|
| v0 cold, empty memory (**baseline**) | 71.0 | [ ] | [ ] | [ ] | [ ] | [ ] |
| v0, learned memory (Shared Brain effect) | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| vN, learned memory (Version N+1 effect) | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| *Reference: greedy (no LLM)* | *71.0* | *100%* | 0 | 0 | 0 | n/a |
| *Reference: greedy + polish (no LLM)* | *[POLISH_REF]* | | 0 | 0 | 0 | n/a |

## Layer 2, equipment placement: QAPLIB facility layout (live)

- **Task:** 45 instances, split 18 / 9 / 18.
- **Verifier:** computes the exact cost of the returned assignment.
- **Score:** reported as the gap to the best-known answer. 16 of the best-known answers are not proven optimal, so a score above 1.0 on one of those would be a new record.
- **How the team works:** the solver role writes a heuristic program. It runs in a subprocess with a hard timeout (10 s in the baseline), and each attempt's verified gap is fed back.
- **Baselines** (`reports/qaplib_baseline.json`):
  - 2-swap: 0.989 mean score on held-out, 1.2% mean gap, at the best-known answer on 5 of 18.
  - Random: 0.707.
- **Live smoke:** Super wrote a runnable program with a 13.9% gap on els19, where 2-swap is at 0.0%. Cost $0.016.
- **Evolve run 1** (`runs/qaplib-evolve-1/`): crashed on a Crusoe 429 (`max_requests_limit`) before finishing.
- **Data-center QAP instances** (`benchmarks/qaplib/dc_instances.py`): GPU rack placement in DoneTwin halls, in QAPLIB's format. Facilities are one training job's racks, locations are the hall's rack slots, flow is GB/s between racks, and distance is cable-tray length. There is no published optimum, so the reference is the best of a time-boxed robust tabu search and 2-swap; 1.0 matches it. This is what ties the QAPLIB layer to real data-center data.
- **Held-out so far (v0 cold → final):**
  - QAPLIB (`qaplib-research-1`): final 0.966, 2.9 points below the 2-swap reference (0.995). v0 cold is [QAP_V0].
  - Data-center QAP (`qaplib-dc-research-1`): 0.477 → 0.802 (+0.325). The 2-swap reference is 0.967, so the final version is still below a textbook heuristic.
  - Still running: `qaplib-research-2` and `qaplib-dc-research-2`.
- **Evolve run 2** (`runs/qaplib-evolve-2/`): 2 generations on a 4 / 3 / 6 split, after the 429 client fix. Result: [QAP_RESULT]. Same table as layer 1: v0 cold is the baseline, with 2-swap as the gray no-LLM reference row.

## Layer 3, GPU kernels: KernelBench mini

- **Task:** 12 Level-1 CUDA problems (`benchmarks/kernelbench_mini/`).
- **Verifier:** correctness and speed against the PyTorch reference, with static reward-hack guards.
- **Status:** built and tested offline. **Not run:** our Crusoe project has zero GPU quota, so the L40S VM (`scripts/crusoe_gpu_vm.sh`) cannot be created.
- **Result:** [KB_RESULT]. If none: "built and tested offline with reward-hack guards; the live run is next, on a Crusoe GPU."

## Layer 4, inference: data-center operations questions (live)

- **Task:** a fixed workload of data-center operations questions drawn from DoneTwin halls: power budget per row, rack counts, CDU headroom, hose reach. One task is one hall with 12 questions (4 types × 3). Splits: 4 train, 3 dev, 4 held-out halls. Full write-up: `docs/INFERENCE_BENCHMARK.md`.
- **Verifier:** each answer is checked by code against an exact computed value. The harness also measures the Crusoe cost of every call.
- **Starting point:** v0 is deliberately expensive: Super for every type, reasoning on, max_tokens 4000, one question per call.
- **Knobs:** model per type (Super, Nano, Lightning), reasoning per type, token caps, batch size 1 to 12, temperature, prompts and playbooks.
- **Gate:** a version is kept if mean cost per hall falls by 15% or more with no loss in accuracy and no lost passes, or if accuracy rises by 0.05 or more.
- **Live run 1** (`runs/inference-opt-1/`): on dev, batch_size 3 cut cost by 30.5% at 100% accuracy. There is no held-out result yet, so it is not a headline. Run 2 (`inference-opt-2`) is running.
- **Limits:** this is our own workload, so there are no outside records. It tunes how we call the model, not Crusoe's serving stack (GPU scheduling, KV cache, quantization).
- **Headline:** same accuracy at [INF_COST_PCT]% of the cost.

**Measured live on 2026-10-17, which motivates this layer:**
- Nano is about 11x cheaper per call than Super, at 0.55 s per small call.
- Turning Nano's reasoning off took one call from 180 tokens and 2.0 s to 7 tokens and 0.2 s.
- In the hall layout, reasoning off never produced a feasible layout, so cheaper is not free. That is why the gate checks accuracy as well as cost.
- Crusoe reported 0 cached tokens, so we claim no cache savings.

Cost, tokens and calls per verified point are also reported on layers 1 to 3.

## Fits the same interface, not run

- AlgoTune: 77 numpy/scipy tasks with timing verifiers.
- ALE-Bench: heuristic contests with official scorers.

## What we show

1. **Per case study:** the held-out table above. Each row is reported as the mean plus the per-task spread (min, max).
2. **Version history:** v0 to vN, each step naming the accepted paper and its measured dev gain or cost cut.
3. **Papers ledger** (`reports/papers.md`):

| Paper | Change | Claimed | Measured | Verdict |
|---|---|---|---|---|
| [id] | [typed change] | [+x score, -y% cost] | [+x', -y'%] | reproduced / smaller than claimed / not reproduced |

4. **Generality evidence:** the same 3 core files run every layer. Only the adapter differs.

## What is distinctive

**Prior art:**
- ACE (Agentic Context Engineering, 2025) already has a curator with helpful/harmful counters.
- GEPA, DGM and ADAS optimize prompts or code against a score.

**What we add:**
- **(a) Reproduction before admission.** A claim enters memory only after an independent non-LLM measurement reproduces it. The ledger shows claimed vs measured for every paper.
- **(b) Disputes settled by a test run, not a judge.** Each side is hidden in turn on the same dev tasks.
- **(c) Graduation.** Proven lessons move from memory into role playbooks and then leave memory.
- **(d) One gate for quality and cost.** The same rule that keeps a better result keeps a cheaper team.
- **(e) One harness, many domains.** A new domain costs one adapter, not a new system.

**Strongest single piece of evidence:** the share of papers that did not reproduce. Each is a claim an LLM judge would likely have admitted.

**Stretch, if time allows:** three memory arms on held-out (ungated: admit every claim; gated; none). A 2026 paper (arXiv 2605.12978) found ungated memory can do worse than no memory.

## Honesty

- **Generality is limited:** it is claimed only for problems with a programmatic verifier. It is shown on [N_LIVE] of 4 layers with live results.
- **Run 1 tied greedy:** on 15 held-out halls the team scored 71.0, equal to greedy, and the gate rejected 9 of 9 changes. We report that as the result so far.
- **Small sample:** 15 held-out halls and one seed. The settings experiment used 4 halls with about ±1.3 points of noise, so +2.66 is a small effect. We report the mean with the per-task spread and make no significance claim.
- **Greedy is the floor by construction:** the data-hall team starts from greedy's layout and keeps its best layout, so it cannot score below greedy. The claim is the gain of the final version over v0 cold. For QAPLIB there is no such floor: the team writes programs from scratch, so v0 cold can score below 2-swap, and we show 2-swap so that is visible.
- **Limited model:** the data-hall model is 2D, built from published specs. No airflow, no structural design, and it is not an engineering tool. A high score here does not show real-world capability.
- **Not ours:** the data-hall scorer and hall generator are a teammate's code (Rikin's DoneTwin).
- **No cache savings claimed:** Crusoe reported 0 cached tokens.
- **Every number in brackets is a placeholder** until the live runs finish.

## Live demo run-of-show (3 minutes)

**Rule: verify live, replay the climbs.** Nothing on stage waits on a live LLM call. Reasoning calls take 100 to 340 s, and Crusoe returned 429s all day. What runs live is the verifier: the DoneTwin scorer recomputing a layout, and the dashboards reading finished runs.

**Before going on:** open these 4 tabs in order and check each one loads:
1. http://127.0.0.1:5173, Overview tab;
2. http://127.0.0.1:8000, DoneTwin, with the chosen held-out hall preloaded;
3. http://127.0.0.1:5173, AI Data Center tab, scrolled to the hill climb, with the run selected;
4. http://127.0.0.1:5173, Library tab, with the matching paper open.

Rebuild the data once after the last run lands (`scripts/build_datacenter_view.py`, `scripts/publish_results_to_dashboard.py`). Turn on Do Not Disturb.

| Time | On screen | Say (gist) | Fallback if it breaks |
|---|---|---|---|
| 0:00 to 0:20 | Overview tab: the thesis and the loop diagram | "An AI data center is a stack of optimization problems code can check. Our harness turns each one into a Nemotron team on Crusoe that edits itself and keeps a change only when the verifier measures a gain." | Hero screenshot, `docs/demo/hero-*.png` (to capture once the Overview tab ships) |
| 0:20 to 1:00 | DoneTwin control room: one held-out hall in 3D. Show v0 cold's layout (= greedy), then the team's best verified layout, and let the scorer recompute live | "This hall was never seen in training. Here is the cold start: [V0_HALL] out of 100. Here is the team's layout: [VN_HALL]. The score is recomputed right now by code: power, aisles, cooling, hose reach. No LLM grades it." If the gain came from the polish tool, add: "Polish alone, with no LLM, gets 72.75; the team's job was choosing to use it." | `docs/demo/donetwin-empty-hall.png` shows the empty hall; the layout shots are still to capture. If there is no hall gain, show the cold layout and say the gate kept nothing on this layer |
| 1:00 to 1:30 | AI Data Center tab: the four layer cards, v0 cold → final, with gray no-LLM references | "Four layers, one harness. Each card is the same team before and after it edits itself, on unseen tasks. Gray is the textbook method with no LLM, so you can see where we still lose." | `docs/demo/datacenter-1548.png` (15:48 snapshot from before the v0-cold reframe; re-shoot after the final runs) |
| 1:30 to 2:10 | Hill climb: best run (currently `qaplib-dc-research-1`), press Replay, then click one green dot | "Each dot is a change the team proposed to itself. Red, the gate refused it. Green, kept. This one: turn reasoning on for the solver. It claimed +0.03. The verifier measured +0.146, but cost went up 93%. The gate shows what it read: You.com sources." | `docs/demo/hillclimb.png`, then `docs/demo/hillclimb-detail.png` |
| 2:10 to 2:30 | Library tab: the same change written up as a paper, with the gate's verdict attached | "Every change is a paper with a falsifiable claim. The verifier is the reviewer. A claim enters memory only if it reproduces." | Read the claim and verdict from `hillclimb-detail.png` |
| 2:30 to 3:00 | Back to the AI Data Center tab | Caveats: "Classical references still win on quadratic assignment: our best is 0.802 against 2-swap's 0.967. KernelBench: [KB_STATUS]." Ask: "Bring a verifier, get a team that keeps only what it measures. We want problems from Crusoe's own build-out to run it on." | Same slide, spoken |

**Optional live moment.** At 0:00, start a one-hall Nano run in a terminal (`.venv/bin/python scripts/run_datahall.py --live --one`). Return to it at 2:30 only if it has printed its bill. If not, never mention it. It needs `CRUSOE_API_KEY` in that shell, and it competes with any runs still going for Crusoe's concurrency cap.

**Placeholders to fill at about 5:00 PM** from `reports/RESULTS.md`:
- [V0_HALL] and [VN_HALL]: one held-out hall from the best hall run.
- [KB_STATUS]: if an Apple MPS run finished, use "measured on a laptop GPU (Apple MPS), because our Crusoe GPU quota was zero". Otherwise: "built with reward-hack guards, not run: our Crusoe GPU quota was zero".
- Swap the hill-climb run if a later one beats `qaplib-dc-research-1`.

## 60-second demo script

See `VIDEO.md`: four layers, with the hall as the hero.
