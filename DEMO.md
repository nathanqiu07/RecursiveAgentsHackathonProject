# Demo: one harness, several case studies

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

## Case study 1: GPU data-hall layout (live)

- **Verifier:** Rikin's DoneTwin scorer (`donetwin/donetwin/score.py`), no LLM. Score 0 to 100 = 70 for density + 20 for resilience to a single CDU failure + 10 for hose length.
- **Hard rules, which score 0 if broken:** 2N power on A/B busways, 1.2 m hot/cold aisles, keep-outs, hose reach, CDU capacity, room air budget.
- **Baseline:** greedy, about 68 to 70, usually 0 of the 20 resilience points. Every run starts from greedy's layout, so a run never scores below greedy. Any gain is the team's.
- **Splits:** train 12 and dev 8 halls from disjoint seeds in 0..9999. Held-out is 15 halls from seeds 10000..10049.
- **Live run 1:** every role on Nemotron Nano with reasoning off. The team matched greedy (dev 0.703, i.e. 70.3/100), and no candidate change gained anything, so the gate kept v0. An experiment is now testing which model and reasoning settings beat greedy.

Both outcomes are reportable:
- **If a setting beats greedy:** the table below shows the gain on unseen halls.
- **If none does:** the ledger shows every change the gate refused, with claimed vs measured. "The gate refused every change that did not help" is Version N+1 working as designed. A team that edits itself without a gate would have shipped those changes.

| Config (15 held-out halls) | Score /100 | Pass rate | Calls/task | Tokens/task | $/task | $ per verified point |
|---|---|---|---|---|---|---|
| Greedy (no LLM) | [~68 to 70] | [ ] | 0 | 0 | 0 | n/a |
| v0, empty memory | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| v0, learned memory (Shared Brain effect) | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| vN, learned memory (Version N+1 effect) | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

## Case study 2: QAPLIB facility layout (being wired in)

- **Task:** 45 instances, split 18 / 9 / 18.
- **Verifier:** computes the exact cost of the returned assignment.
- **Score:** reported as the gap to the best-known answer. 16 of the best-known answers are not proven optimal, so a score above 1.0 on one of those would be a new record.
- **How the team works:** the solver role writes a heuristic program. It runs in a subprocess with a hard timeout (10 s in the baseline), and each attempt's verified gap is fed back.
- **Baselines** (`reports/qaplib_baseline.json`):
  - 2-swap: 0.989 mean score on held-out, 1.2% mean gap, at the best-known answer on 5 of 18.
  - Random: 0.707.
- **Live numbers are pending.** Same table as case study 1, with the 2-swap baseline in place of greedy.

## Case study 3: inference cost (an axis on every case study)

This is not a separate benchmark. On every case study the harness measures Crusoe cost, tokens and calls per verified point, and the gate keeps a change that cuts cost by 15% or more at equal score.

**Measured live on 2026-10-17:**
- Nano is about 11x cheaper per call than Super, at 0.55 s per small call.
- Turning Nano's reasoning off took one call from 180 tokens and 2.0 s to 7 tokens and 0.2 s.
- An early data-hall task with reasoning on scored 0: reasoning hit the 8192-token cap and the JSON was cut off.
- Crusoe reported 0 cached tokens, so we claim no cache savings.

## Next case studies (fit the interface, not run)

| Benchmark | Why it fits | Blocker |
|---|---|---|
| AlgoTune | 77 numpy/scipy tasks with timing verifiers, validated on this Mac | Adapter not written |
| ALE-Bench | Scored heuristic contests. Nemotron 3 Super's published average goes from 214 to 1017 across 1 vs 16 refinement rounds | No Docker here |
| KernelBench | GPU kernels scored on correctness and speed | Needs an NVIDIA GPU, e.g. a Crusoe VM |

## What we show

1. **Per case study:** the held-out table above. Each row is reported as the mean plus the per-task spread (min, max).
2. **Version history:** v0 to vN, each step naming the accepted paper and its measured dev gain or cost cut.
3. **Papers ledger** (`reports/papers.md`):

| Paper | Change | Claimed | Measured | Verdict |
|---|---|---|---|---|
| [id] | [typed change] | [+x score, -y% cost] | [+x', -y'%] | reproduced / smaller than claimed / not reproduced |

4. **Generality evidence:** the same 3 core files run both case studies. Only the adapter differs.

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

- **Generality is limited:** it is claimed only for problems with a programmatic verifier, and is shown on 2 domains.
- **Small sample:** 15 held-out halls and one seed. We report the mean with the per-task spread and make no significance claim.
- **Greedy is the floor by construction:** the data-hall team starts from greedy's layout and keeps its best layout, so it cannot score below greedy. The claim is the gain over greedy, not the absolute score.
- **Limited model:** the data-hall model is 2D, built from published specs. No airflow, no structural design, and it is not an engineering tool. A high score here does not show real-world capability.
- **Not ours:** the data-hall scorer and hall generator are a teammate's code (Rikin's DoneTwin).
- **No cache savings claimed:** Crusoe reported 0 cached tokens.
- **Every number in brackets is a placeholder** until the live runs finish.

## 60-second demo script

| Time | On screen | What we say |
|---|---|---|
| 0:00 to 0:10 | Three boxes: tasks, verifier, prompts, feeding one harness | "Bring a problem with a verifier: tasks, a scorer, role prompts. The harness turns it into a team that improves itself." |
| 0:10 to 0:22 | Data hall: greedy's layout, about 70, resilience 0/20 | "Case study one: GPU data halls. Greedy packs racks but loses 20 points when one CDU fails. The team starts here, scored by code with no LLM in it." |
| 0:22 to 0:38 | Papers ledger: claimed vs measured, some rows not reproduced | "A research agent writes papers: a change plus a predicted gain. The verifier reruns dev with and without it. [N of M] did not reproduce. An LLM judge would have admitted them." |
| 0:38 to 0:50 | Held-out table, then QAPLIB on the same 3 core files | "On 15 unseen halls: [greedy] to [vN]. Same harness, second case study: QAPLIB, against 16 best-known answers that are not proven optimal." |
| 0:50 to 1:00 | Cost row: thinking off, 180 to 7 tokens | "The same gate cuts our Crusoe bill. A change ships only if it scores higher or costs less." |
