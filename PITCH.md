# Pitch

## 1. One line

An agent team on Crusoe and Nemotron that stores only verifier-measured lessons in shared memory and ships a new version of itself only when it scores higher.

**Tagline:** Learn only what a verifier can measure.

## 2. What we do

A task arrives. Each role, a Nemotron model on Crusoe, reads its slice of the Shared Brain, then works. A non-LLM verifier scores the result. Failures become proposed memory entries that cite the run and its score. A curator admits only cited entries, merges duplicates (never when the numbers differ), and settles contradictions by weight of evidence or marks both disputed. Later scores credit entries with helps or hurts; losers are pruned. After each round a Reviewer proposes up to 3 self-edits: a prompt, a playbook rule, a config value, a role, or promoting proven memory into a playbook. A gate keeps an edit only if dev scores rise and no pass is lost. Held-out tasks score the final before/after.

## 3. Why it matters

Most agent teams score the same on run 100 as on run 1, and their shared notes fill with stale or wrong claims. If an LLM judge decides what the team learns, the team learns what sounds right, and the errors compound across runs. A verifier that recomputes the answer gives a measurement no agent can argue past. For Crusoe, the AI data-center build-out is full of checkable layout and capacity problems. Each lesson that moves from memory into a playbook trims tokens from every later call, so verified work per token gets cheaper.

## 4. Track fit

### Shared Brain

| Track asks | Mechanism | File |
|---|---|---|
| Every agent reads before it works | `read()` ranks entries for the role by net helps, task relevance and recency within a character budget | `team/brain.py` |
| Every agent updates afterward | Agents never write entries. `propose()` is the only write path, and the curator decides what lands | `team/brain.py`, `team/evolve.py` (`proposals_from_run`) |
| Keep it current | An entry earns helps or hurts when a later run that read it beats or trails the last score on that task | `team/brain.py` (`credit`) |
| Keep it lean | 280-character entries, a 40-entry active cap, retire after 3 uses when hurts exceed helps + 1, and proven entries leave memory once promoted | `team/brain.py` (`prune`), `team/evolve.py` |
| Conflict-aware | Same wording with different numbers is never a duplicate. A contradiction keeps the better-supported entry or marks both disputed. Writes are file-locked, and every decision goes to `ledger.jsonl` | `team/brain.py` (`Curator`, `SharedBrain`) |
| Measurable gains | Held-out ablation: v0 with empty memory vs v0 with learned memory | `team/evolve.py` (`final_report`) |
| Cheap on Crusoe | `stable_prefix()` renders byte-identical text until the brain changes, so it is prefix-cacheable. Cost is tracked per role, cached tokens included | `team/brain.py`, `team/crusoe.py` |

### Version N+1

| Track asks | Mechanism | File |
|---|---|---|
| Review what worked and what didn't | The Reviewer reads mean score, pass rate, cost, the top 15 failures by count, and the most helpful memory entries | `team/evolve.py` (`Reviewer`) |
| Rewrite prompts, playbooks, team structure | 7 typed edits: prompt edit, playbook add or remove, config set, add or remove a role, promote memory into a playbook. Each is validated before it runs | `team/evolve.py` (`MUTATIONS`) |
| Keep a change only if it helps | The gate runs candidate and parent on the same dev tasks and needs a mean gain of at least 0.05 with no lost passes. Promotion is accepted at equal score, with the promoted entries hidden, because memory gets smaller | `team/evolve.py` (`Gate`) |
| Before/after on unseen tasks | Train, dev and held-out splits must be disjoint, and touching a held-out task before the final report raises an error | `team/evolve.py` (`Evolver`) |

## 5. What we will show

| Config (held-out tasks) | Mean score | Pass rate | Tokens | Cost (USD) |
|---|---|---|---|---|
| v0 + empty memory | [before] | [before] | [tokens] | [cost] |
| v0 + learned memory | [after] | [after] | [tokens] | [cost] |
| vN + learned memory | [after] | [after] | [tokens] | [cost] |

All numbers are pending. No live Crusoe run exists yet, and the offline demo (`scripts/demo_offline.py`, toy domain in `team/bench.py`) uses a scripted LLM.

**Benchmarks.** Each is an open-ended hill climb scored by a non-LLM verifier against a best-known score.

1. **GPU data-hall layout** (`optimizer/`). Generated halls with GB200 NVL72 racks at 132 kW, busway power with derate, CDU liquid cooling with N+1, hose limits, columns and keep-outs. `verify.py` shares no code with the solver. An OR-Tools CP-SAT model gives the optimum over a discrete slot grid. This is a stated 2D model built from published specs, not an engineering design tool: no airflow, A/B power or structural design.
2. **QAPLIB facility layout** (`benchmarks/qaplib/`, in progress). Best-known solutions, some not proven optimal.
3. **Inference or code optimization** (AlgoTune). Still being checked.

**Closest prior work.** ACE (Agentic Context Engineering, 2025) already has helpful/harmful counters and a curator. GEPA, DGM and ADAS optimize prompts or code against a score. What we add: memory is admitted only when it cites a verifier's measurement, and proven lessons move into role playbooks through a gated A/B on dev tasks, then leave memory.
