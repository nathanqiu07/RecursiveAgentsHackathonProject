# Pitch

## 1. One line

One harness on Crusoe and Nemotron that turns any problem with a programmatic verifier into a self-improving agent team, keeping a change only when the verifier measures a gain.

**Tagline:** Bring a verifier. Get a team that improves.

## 2. What we do

The harness is general. The solver is not. A domain plugs in with 3 things: tasks split into train, dev and held-out; a verifier that returns a number; and role prompts. The harness supplies the rest.

A task arrives. Each role, a Nemotron model on Crusoe, reads its slice of the Shared Brain, then works, and the verifier scores the result. Lessons enter memory only with a cited measurement. After each round a research agent writes papers: one typed change to the team plus a predicted gain. The verifier reruns dev with and without the change. A change is kept only for a measured gain or a 15% cost cut. Held-out tasks score the final version.

## 3. Why it matters

Most agent teams score the same on run 100 as on run 1, and their shared notes fill with stale or wrong claims. If an LLM judge decides what the team learns, the team learns what sounds right, and the errors compound. A verifier that recomputes the answer gives a measurement no agent can argue past. Many valuable problems already have one: layouts, schedules, solvers, kernels. For Crusoe, the AI data-center build-out is full of them, and a gate that also keeps cheaper versions makes each verified point cost fewer tokens.

## 4. What is general, and the evidence

| Piece | Lines | Changes per domain? |
|---|---|---|
| Shared Brain: evidence-gated memory, dedup, dispute tests, pruning (`team/brain.py`) | 390 | No |
| Version N+1: papers, claimed vs measured, score and cost gate, held-out report (`team/evolve.py`) | 547 | No |
| Crusoe client: retries, rate limits, cost per role (`team/crusoe.py`) | 482 | No |
| Data-hall adapter (`team/datahall.py`) | 484 | Yes |
| QAPLIB adapter (`benchmarks/qaplib/team_runner.py`) | 204 | Yes |

Each adapter is one function with the same signature: `run_team(version, brain, task, llm) -> RunResult`.

## 5. Track fit

### Shared Brain

| Track asks | Mechanism | File |
|---|---|---|
| Every agent reads before it works | `read()` ranks entries for the role by net helps, task relevance and recency within a character budget | `team/brain.py` |
| Every agent updates afterward | Agents never write entries. `propose()` is the only write path, and the curator decides what lands | `team/brain.py`, `team/evolve.py` (`proposals_from_run`) |
| Keep it current | An entry earns helps or hurts when a later run that read it beats or trails the last score on that task | `team/brain.py` (`credit`) |
| Keep it lean | 280-character entries, a 40-entry active cap, retire after 3 uses when hurts exceed helps + 1, and proven entries leave memory once promoted | `team/brain.py` (`prune`), `team/evolve.py` |
| Conflict-aware | Same wording with different numbers is never a duplicate. A contradiction keeps the better-supported entry or marks both disputed. Each generation, up to 2 disputed pairs are settled by running dev once with each side hidden, and the loser is retired with the measurement on record. Writes are file-locked, and every decision goes to `ledger.jsonl` | `team/brain.py`, `team/evolve.py` |
| Measurable gains | Held-out ablation: v0 with empty memory vs v0 with learned memory | `team/evolve.py` (`final_report`) |
| Cheap on Crusoe | Cost, tokens and calls are tracked per role and model. Promoted entries shrink what later calls read. Crusoe reported 0 cached tokens in our 2026-10-17 probe, so we claim no cache savings | `team/crusoe.py`, `team/evolve.py` |

### Version N+1

| Track asks | Mechanism | File |
|---|---|---|
| Review what worked and what didn't | The Reviewer reads mean score, pass rate, cost, the top 15 failures by count, and the most helpful memory entries | `team/evolve.py` (`Reviewer`) |
| Rewrite prompts, playbooks, tools, team structure | 8 typed edits: prompt edit, playbook add or remove, config set, tool on/off from an allowlist, add or remove a role, promote memory into a playbook. Each must carry a numeric claim (dev gain, cost change) or it is dropped | `team/evolve.py` (`MUTATIONS`, `claim_error`) |
| Keep a change only if it helps | Candidate and parent run on the same dev tasks. A change is kept for a mean gain of 0.05 or more with no lost passes, or a cost cut of 15% or more at equal score | `team/evolve.py` (`Gate`) |
| Before/after on unseen tasks | Train, dev and held-out splits must be disjoint. Touching a held-out task before the final report raises an error | `team/evolve.py` (`Evolver`) |

## 6. Case studies: optimizing every layer of an AI data center

Each layer is framed as an optimization problem that code can check.

| Layer | Case study | Verifier (no LLM) | Baseline: v0 cold (no-LLM reference) | Status |
|---|---|---|---|---|
| 1. Hall layout (**hero**) | GPU data-hall layout, DoneTwin | Rikin's scorer, 0 to 100 | v0 cold 71.0 on 15 held-out halls (greedy 71.0) | Run 1: final = v0 cold, gate rejected 9 of 9 changes. Settings test: only Nano with reasoning on beat the cold start (+2.66 points on 4 train halls). Run 2 live: [RUN2_SCORE] |
| 2. Equipment placement | QAPLIB facility layout (quadratic assignment) | Exact assignment cost, as a gap to the best-known answer | v0 cold [QAP_V0] (2-swap 0.995) | QAPLIB final 0.966 on held-out. On the data-center QAP instances (rack placement by traffic and cable length): 0.477 → 0.802, with 2-swap at 0.967. More runs live |
| 3. GPU kernels | KernelBench mini, 12 Level-1 CUDA problems | Correctness and speed vs PyTorch, with reward-hack guards | v0 cold (PyTorch eager 1.0x) | Not run: our Crusoe project has zero GPU quota |
| 4. Inference | Data-center operations questions from DoneTwin halls (row power budget, rack counts, CDU headroom, hose reach), with code-checked exact answers | Exact answer, plus measured Crusoe cost | v0 cold: Super, reasoning on, one call per question | Live run 1 (4/3/4 halls, 12 questions each): same accuracy at [INF_COST_PCT]% of the cost |

Every headline gain is v0 cold to final on held-out tasks. Greedy and 2-swap are shown only as no-LLM references.

Fits the same interface, not run: AlgoTune, ALE-Bench.

Details are in `DEMO.md`, and the video plan is in `VIDEO.md`.

**Closest prior work:**
- ACE (Agentic Context Engineering, 2025) already has a curator with helpful/harmful counters.
- GEPA, DGM and ADAS optimize prompts or code against a score.

**What we add:**
- Memory admits a claim only when a verifier's measurement reproduces it.
- Disputes are settled by a test run, not a judge.
- Proven lessons move into role playbooks, then leave memory.
- The same gate keeps cheaper versions.
- All of it is reused unchanged across domains.
