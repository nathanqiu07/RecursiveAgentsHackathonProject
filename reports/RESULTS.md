# Benchmark results (live Crusoe runs only)

Every number here comes from a live run on Crusoe Managed Inference (Nemotron-3 Nano 30B and Super 120B). Offline and scripted runs are tests and are excluded. Scores: higher is better; gain = best team version minus baseline on the held-out set.

## Summary

| Case study | Baseline | Best team version | Gain | Status | Cost |
|---|---|---|---|---|---|
| Data hall layout, live run 1 | greedy baseline (no LLM) 0.710 | v0 + empty brain (cold) 0.710 | +0.000 (+0.0%) | done | $0.1730 (475 calls) |
| Data hall layout, live run 2 | greedy layout (no LLM) (pending) | pending | pending | not started | n/a |
| QAPLIB (quadratic assignment) | 2-swap local search, 10 s restarts (no LLM) 0.9954 | n/a | n/a | blocked: crashed: RateLimitError, HTTP 429, max_requests_limit; 1 candidate scored before the crash; last write 15:09 PDT | n/a (crashed before usage.json) |
| ALE-Bench AHC039 (Purse Seine Fishing) | reference greedy (no LLM) 1.000 | n/a | n/a | blocked: crashed: RateLimitError, HTTP 429, max_requests_limit; 1 candidate scored before the crash; last write 15:09 PDT | n/a (crashed before usage.json) |
| KernelBench mini (CUDA kernels) | torch eager (speedup 1.0x) 1.00 | n/a | n/a | blocked: needs a Crusoe GPU VM (no runs/kernelbench-* yet) | n/a |
| Model settings: Nano vs Super, reasoning on/off (4 train halls) | greedy layout (no LLM) 0.6949 | B (Nano 30B, reasoning on, max 16k tokens) 0.7084 | +0.0135 | running: 20 hall runs in 6 configs; last write 15:09 PDT | $0.1350 (93 calls) |

## Data hall layout, live run 1

- Verifier: DoneTwin checker: hard feasibility rules (collision, 2N power, CDU capacity, hose reach, zones); score = DoneTwin score / 100, pass = feasible.
- Baseline: greedy baseline (no LLM), held-out mean 0.710 (source: final held-out table).
- Status: done; final report and live usage written.
- Live evidence: usage.json: 475 paid calls, $0.1730.
- Totals: 475 calls, 1,974,292 tokens (1,523,865 prompt + 450,427 completion, 0 cached), $0.1730, 0 errors, wall 1353.7 s.
- By model: NVIDIA-Nemotron-3-Nano-30B-A3B 472 calls $0.1656; NVIDIA-Nemotron-3-Super-120B-A12B 3 calls $0.0075.

### Held-out

| Config | n | Mean score | Pass rate | Tokens | Cost (USD) | vs baseline |
|---|---|---|---|---|---|---|
| v0 + empty brain (cold) | 15 | 0.710 | 100% | 223,921 | $0.0197 | +0.000 (+0.0%) |
| v0 + final brain | 15 | 0.710 | 100% | 257,007 | $0.0227 | +0.000 (+0.0%) |
| greedy baseline (no LLM) | 15 | 0.710 | 100% | 0 | $0.0000 | baseline |

Note: 'v0 + final brain' appears twice in the final report: no candidate passed the gate, so the final version is v0 and its arm repeats 'v0 + final brain'; shown once.

Dev mean of the parent by generation: gen 0 0.703, gen 1 0.703, gen 2 0.703.

### Generations (each candidate is gated on the dev set)

9 candidates, 0 accepted.

| Gen | Mutation | Change | Claim | Measured dev gain | Decision | Why |
|---|---|---|---|---|---|---|
| 0 | playbook_add | Layout step: Run capacity_calc tool to verify that the proposed layout respects... | none recorded (ledger predates claims) | n/a | rejected | invalid: step longer than 280 chars |
| 0 | tool_set | tool capacity_calc enabled=True | none recorded (ledger predates claims) | -0.0006 | rejected | gain -0.001 < 0.05 |
| 0 | prompt_edit | Planner prompt: You are the power and cooling Planner for a 2N AI data hall. Before... | none recorded (ledger predates claims) | -0.0006 | rejected | gain -0.001 < 0.05 |
| 1 | config_set | tools.capacity_calc.enabled = true | none recorded (ledger predates claims) | +0.0000 | rejected | gain +0.000 < 0.05 |
| 1 | playbook_add | Layout step: Validate CDU capacity and hose reach: for each CDU, count assigned... | none recorded (ledger predates claims) | +0.0000 | rejected | gain +0.000 < 0.05 |
| 1 | prompt_edit | Planner prompt: When budgeting racks per busway row, enforce 2N dual‑feed power red... | none recorded (ledger predates claims) | +0.0000 | rejected | gain +0.000 < 0.05 |
| 2 | playbook_add | Layout step: For each liquid-cooled rack, query the nearest_examples tool for ca... | no numeric claim; rationale: Help the Layout engineer place liquid racks within hose r... | n/a | rejected | invalid: step longer than 280 chars |
| 2 | config_set | tools.capacity_calc.enabled = true | no numeric claim; rationale: Enable the capacity_calc tool so the Planner can compute... | +0.0000 | rejected | gain +0.000 < 0.05 |
| 2 | playbook_add | Planner step: Use the capacity_calc tool to determine the maximum number of racks... | no numeric claim; rationale: Give the Planner a concrete step to calculate budgets usi... | +0.0000 | rejected | gain +0.000 < 0.05 |

Sources: `reports/datahall_live_1/before_after.json`, `reports/datahall_live_1/vs_greedy.md`, `reports/datahall_live_1/usage.json`, `reports/datahall_live_1/version_ledger.jsonl`, `reports/datahall_live_1/run.log`.

## Data hall layout, live run 2

- Verifier: DoneTwin checker: hard feasibility rules (collision, 2N power, CDU capacity, hose reach, zones); score = DoneTwin score / 100, pass = feasible.
- Baseline: greedy layout (no LLM), held-out mean n/a (source: not available yet).
- Status: not started; nothing on disk yet.
- Live evidence: named live run.

## QAPLIB (quadratic assignment)

- Verifier: exact QAP cost of the printed permutation; score = BKS / cost (1.0 = best known), pass = gap to BKS <= 1%.
- Baseline: 2-swap local search, 10 s restarts (no LLM), held-out mean 0.9954 (source: reports/qaplib_baseline.json on this run's 6 held-out instances).
- Status: blocked; crashed: RateLimitError, HTTP 429, max_requests_limit; 1 candidate scored before the crash; last write 15:09 PDT.
- Live evidence: log shows the Crusoe RPM cap.
- Totals: n/a; the run crashed before writing usage.json, so its bill is not on disk.

Dev mean of the parent by generation: gen 0 0.9373.

### Generations (each candidate is gated on the dev set)

1 candidates, 0 accepted; claim vs measured verdicts: 1 not reproduced.

| Gen | Mutation | Change | Claim | Measured dev gain | Decision | Why |
|---|---|---|---|---|---|---|
| 0 | prompt_edit | solver prompt: You write heuristic optimization programs for the Quadratic Assignm... | dev gain +0.070, cost +1.0%: Adds error handling to avoid runtime errors and improves solution q... | -0.1367 | rejected | gain -0.137 < 0.003 |

Sources: `runs/qaplib-evolve-1/versions/ledger.jsonl`, `runs/qaplib-evolve-1.log`.

## ALE-Bench AHC039 (Purse Seine Fishing)

- Verifier: ALE-Bench AHC039 scorer on private Lite seeds; score = mean case score / greedy case score (1.0 = greedy); rank vs 583 contestants on all 150 private seeds.
- Baseline: reference greedy (no LLM), held-out mean 1.000 (source: 1.0 by definition of the score).
- Status: blocked; crashed: RateLimitError, HTTP 429, max_requests_limit; 1 candidate scored before the crash; last write 15:09 PDT.
- Live evidence: log shows the Crusoe RPM cap.
- Totals: n/a; the run crashed before writing usage.json, so its bill is not on disk.

### Generations (each candidate is gated on the dev set)

1 candidates, 0 accepted.

| Gen | Mutation | Change | Claim | Measured dev gain | Decision | Why |
|---|---|---|---|---|---|---|
| 0 | playbook_add | solver step: First, apply multi‑scale grid with iterative refinement and edge ad... | dev gain +0.015, cost +0.0%: Adds proven multi‑scale grid with iterative refinement, fallback gr... | n/a | rejected | invalid: step longer than 280 chars |

Sources: `runs/alebench-live-1/versions/ledger.jsonl`, `runs/alebench-live-1.log`.

## KernelBench mini (CUDA kernels)

- Verifier: compile, correctness on 5 signed input sets plus a held-back seed, CUDA-event timing; score = speedup over torch eager if correct (cap 10), else 0.
- Baseline: torch eager (speedup 1.0x), held-out mean 1.00 (source: 1.0 by definition of the score).
- Status: blocked; needs a Crusoe GPU VM (no runs/kernelbench-* yet).

## Model settings: Nano vs Super, reasoning on/off

- Verifier: DoneTwin checker on 4 train halls (Layout role varies; Planner and Checker stay on Nano); score = DoneTwin score / 100.
- Baseline: greedy layout (no LLM), mean over the halls 0.6949.
- Status: running; 20 hall runs in 6 configs; last write 15:09 PDT.
- Cost so far: $0.1350 over 93 calls.

| Config | Layout model and settings | Run tag | Halls | Mean score | Greedy | Mean gain (pts of 100) | Improved | Feasible attempts | Calls/hall | Tokens/hall | Cost (USD) | Wall s/hall | Errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | Nano 30B, reasoning off, max 6k tokens | main | 4 | 0.6949 | 0.6949 | +0.00 | 0/4 | 0/12 | 6.0 | 18,662 | $0.0056 | 138 | 0 |
| B | Nano 30B, reasoning on, max 16k tokens | main | 4 | 0.7084 | 0.6949 | +1.34 | 1/4 | 1/12 | 4.0 | 56,528 | $0.0394 | 362 | 0 |
| C | Super 120B, reasoning off, max 6k tokens | main | 4 | 0.6949 | 0.6949 | +0.00 | 0/4 | 0/12 | 6.0 | 22,340 | $0.0459 | 131 | 0 |
| D | Super 120B, reasoning on, max 16k tokens | main | 1 | 0.6981 | 0.6981 | +0.00 | 0/1 | 0/1 | 2.0 | 19,726 | $0.0391 | 356 | 1 |
| F | Super 120B, reasoning off, max 6k tokens, small-edit playbook | pb | 3 | 0.6652 | 0.6651 | +0.00 | 0/3 | 0/0 | 1.0 | 1,420 | $0.0003 | 112 | 3 |
| G | Nano 30B, reasoning off, max 6k tokens, small-edit playbook | pb | 4 | 0.6949 | 0.6949 | +0.00 | 0/4 | 0/12 | 6.0 | 16,794 | $0.0048 | 99 | 0 |

Sources: `/private/tmp/claude-501/-Users-iamdaone-Documents-RecursiveAgentsHackathonProject/068a120f-637f-4002-856f-3eb6683d41a7/scratchpad/headroom/results_main.jsonl`, `/private/tmp/claude-501/-Users-iamdaone-Documents-RecursiveAgentsHackathonProject/068a120f-637f-4002-856f-3eb6683d41a7/scratchpad/headroom/results_pb.jsonl`.

## Live smoke tests

**Crusoe connection probe**, copied from the console output of scripts/crusoe_smoke.py; no file on disk:

| Model | Latency per call (s) | Cost per call (USD) | Cached tokens |
|---|---|---|---|
| Nemotron-3 Nano 30B | 0.55 | 0.000013 | 0 |
| Nemotron-3 Super 120B | 2.00 | 0.000147 | 0 |

Reasoning off cut one call from 180 tokens and 2.0 s to 7 tokens and 0.2 s.

**qaplib-one**: one train task with v0; bill printed to the console only, not saved.

- qaplib-els19: best score 0.8776, best gap to BKS 13.94%; 3/3 programs ran (ok, ok, ok).

Sources: `runs/qaplib-one/logs/qaplib-els19-v0.json`.

**alebench-one**: one train task with v0; 2 calls, 6,738 prompt + 2,545 completion tokens (3,456 cached), $0.0076, 59.1 s of call latency; model NVIDIA-Nemotron-3-Super-120B-A12B.

- ahc039-train0: best score 0.3475; 1/2 programs ran (timeout, ok).

Sources: `runs/alebench-one/logs/ahc039-train0-v0.json`, `runs/alebench-one/usage.json`.

## Not counted

These run directories are not results: `runs/alebench-offline` (no paid usage.json: offline, scripted or a service log); `runs/dashboard` (no paid usage.json: offline, scripted or a service log); `runs/datahall-20261017-143109` (no paid usage.json: offline, scripted or a service log); `runs/localhost` (no paid usage.json: offline, scripted or a service log).

Generated 2026-10-17 15:11:04 PDT. Regenerate with `.venv/bin/python scripts/collect_results.py` (rewrites reports/RESULTS.md and reports/results.json).
