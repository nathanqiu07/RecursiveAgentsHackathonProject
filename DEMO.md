# Demo: improvement, what is distinctive, benchmark

**Core idea (SA's):** A research agent writes papers into shared memory. A solver agent runs a benchmark. A verifier agent checks each paper before anyone trusts it.

**How it maps onto our code:**

| SA's role | Our component | What it does |
|---|---|---|
| Research agent | Reviewer (`team/evolve.py`) | Writes a **paper**: a hypothesis, one typed change, a predicted dev gain and cost change, and the run ids it is based on |
| Solver | Data-hall team (`team/datahall.py`) | Planner, Layout and Checker roles on Nemotron Nano via Crusoe |
| Verifier | Gate plus Rikin's scorer (`donetwin/donetwin/score.py`) | Reruns the paired dev tasks with and without the change. A non-LLM scorer measures both |
| Peer review | LLM reviewer | May only narrow a claim's scope ("3 halls" does not support "all halls"). It never accepts a claim |

A change is one of: a prompt edit, a playbook rule, a config change (model per role, max tokens, retries, thinking on/off), a tool on/off, a role added or removed, or promoting memory into a playbook.

**Acceptance rule:** dev score rises by 0.05 or more with no lost passes, or cost falls by 15% or more at equal score. Memory disputes use the same rule: one dev run with each side hidden.

**What happens to accepted papers:** they enter the Shared Brain with their measurement attached (Shared Brain track). Proven ones graduate into role playbooks as a new team version and leave memory (Version N+1 track).

**Why a verifier and not an LLM judge decides:** a judge lets the team learn what sounds right. That is the documented false-success failure mode.

## Benchmark

**Primary: DoneTwin GPU data-hall layout**
- **Score:** 0 to 100 = 70 for density (IT kW vs a computed upper bound) + 20 for resilience to a single CDU failure + 10 for hose length (`score.py`, line 527).
- **Hard rules, which score 0 if broken:** 2N power on A/B busways, 1.2 m hot/cold aisles, keep-outs, hose reach, CDU capacity, room air budget.
- **No LLM in scoring.** Open-ended, with no proven ceiling.
- **Baseline:** greedy averages about 68 and usually earns 0 of the 20 resilience points. The team starts from greedy's layout and climbs.
- **Splits:** fixed by seed in `team/datahall.py`. Train is 12 halls and dev is 8, drawn from disjoint seeds in 0..9999. Held-out is 15 halls from seeds 10000..10049, never seen in train or dev.

**Second axis: inference cost.** The same gate optimizes the team's own Crusoe bill: cost, tokens and calls per verified point.

**Measured live today:**
- Nano: 0.55 s per small call, about 11x cheaper per call than Super.
- Turning Nano's reasoning off: one call went from 180 tokens and 2.0 s to 7 tokens and 0.2 s.
- Crusoe reported 0 cached tokens, so we claim no cache savings.
- Our opening live task scored 0 with reasoning on: reasoning hit the 8192-token cap and the JSON was cut off. A fix is in progress.

**Next, not in today's results:**

| Benchmark | Status |
|---|---|
| QAPLIB | Built: 45 instances, 16 with best-known answers that are not proven optimal |
| AlgoTune | Validated: 77 tasks |
| ALE-Bench | No Docker here |
| KernelBench | Needs an NVIDIA GPU |

## 1. How we show improvement

**Held-out table:** 15 halls, never seen in train or dev.

| Config | Score /100 | Pass rate | Calls/task | Tokens/task | $/task | $ per verified point |
|---|---|---|---|---|---|---|
| Greedy (no LLM) | [~68] | [ ] | 0 | 0 | 0 | n/a |
| v0, empty memory | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| v0, learned memory (Shared Brain effect) | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| vN, learned memory (Version N+1 effect) | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

Each row is reported as the mean plus the per-hall spread (min, max).

**Version history:** v0 to vN, each step naming the accepted paper and its measured dev gain or cost cut.

**Papers ledger** (`reports/papers.md`):

| Paper | Change | Claimed | Measured | Verdict |
|---|---|---|---|---|
| [id] | [typed change] | [+x score, -y% cost] | [+x', -y'%] | reproduced / smaller than claimed / not reproduced |

## 2. How we show what is distinctive

**Prior art:**
- ACE (Agentic Context Engineering, 2025) already has a curator with helpful/harmful counters.
- GEPA, DGM and ADAS optimize prompts or code against a score.

**What we add:**
- **(a) Reproduction before admission.** A claim enters memory only after an independent non-LLM measurement reproduces it. The ledger shows claimed vs measured for every paper.
- **(b) Disputes settled by a test run, not a judge.** Each side is hidden in turn on the same dev tasks.
- **(c) Graduation.** Proven lessons move from memory into role playbooks and then leave memory, so memory stays small.
- **(d) One gate for quality and cost.** The same rule that keeps a better layout keeps a cheaper team: model per role, thinking on/off, max tokens.

**Strongest single piece of evidence:** the share of papers that did not reproduce. Each is a claim an LLM judge would likely have admitted.

**Stretch, if time allows:** three memory arms on held-out (ungated: admit every claim; gated; none). A 2026 paper (arXiv 2605.12978) found ungated memory can do worse than no memory.

**Outside evidence that refinement helps Nemotron:** on ALE-Bench's public leaderboard, Nemotron 3 Super goes from 214 to 1017 average performance between 1 and 16 refinement rounds.

## 3. Honesty

- **Small sample:** 15 held-out halls and one seed. We report the mean with the per-hall spread. We make no significance claim.
- **Limited model:** the data-hall model is 2D, built from published specs. No airflow, no structural design, and it is not an engineering tool. A high score here does not show real-world capability.
- **Not ours:** the scorer and the hall generator are a teammate's code (Rikin's DoneTwin).
- **No cache savings claimed:** Crusoe reported 0 cached tokens.
- **Every number above is a placeholder** until the live run finishes.

## 4. 60-second demo script

| Time | On screen | What we say |
|---|---|---|
| 0:00 to 0:10 | One hall: greedy's layout, score about 68, resilience 0/20 | "Greedy packs racks but loses 20 points when one CDU fails. Our team starts here and climbs, scored by a verifier with no LLM in it." |
| 0:10 to 0:25 | The Reviewer's paper: hypothesis, change, predicted gain | "A research agent writes a paper: a change to the team plus a predicted gain. It is a claim, not a fact." |
| 0:25 to 0:40 | Papers ledger: claimed vs measured, with some rows not reproduced | "The verifier reruns the dev halls with and without the change. [N of M] papers did not reproduce. An LLM judge would have admitted them." |
| 0:40 to 0:52 | Version history, then the held-out table | "Accepted papers enter shared memory. Proven ones graduate into playbooks. On 15 unseen halls: [before] to [after], at [$ per verified point]." |
| 0:52 to 1:00 | Cost row: thinking off, 180 to 7 tokens | "The same gate cuts our Crusoe bill. A change is kept only if it scores higher or costs less." |
