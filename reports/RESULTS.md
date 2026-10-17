# Benchmark results (live Crusoe runs only)

Every number here comes from a live run on Crusoe Managed Inference (Nemotron-3 Nano 30B and Super 120B). Offline and scripted runs are tests and are excluded. Scores: higher is better. The baseline is v0 cold: the starting team version with an empty shared brain. Gain = final team version minus v0 cold on the held-out set.

## Summary

| Case study | Baseline: v0 cold | Final team version | Gain (final - v0 cold) | Status | Cost |
|---|---|---|---|---|---|
| Data hall layout, live run 1 | 0.710 | v0 + final brain 0.710 | +0.000 (+0.0%) | done | $0.1730 (475 calls) |
| Data hall layout, live run 2 | 0.688 | v0 + final brain 0.688 | +0.000 (+0.0%) | done | $0.0433 (128 calls) |
| Data hall layout, live run 2d | 0.688 | v0 + final brain 0.688 | +0.000 (+0.0%) | done | $0.0794 (109 calls) |
| Data hall layout, live run 3 | 0.688 | v0 + final brain 0.688 | +0.000 (+0.0%) | done | $0.0385 (129 calls) |
| Data hall layout, live run 5 | 0.688 | v1 + final brain 0.728 | +0.040 (+5.8%) | done | $0.0468 (152 calls) |
| QAPLIB (quadratic assignment) | n/a | n/a | n/a | blocked: crashed: RateLimitError, HTTP 429, max_requests_limit; 1 candidate scored before the crash; last write 15:09 PDT | n/a (crashed before usage.json) |
| QAPLIB, live run with research | 0.9593 | v2 + final brain 0.9662 | +0.0069 (+0.7%) | done | $0.0163 (89 calls) |
| GPU rack placement (data-center QAP), live run 1 | 0.5042 | v1 + final brain 0.6394 | +0.1352 (+26.8%) | done | $0.0168 (96 calls) |
| GPU rack placement (data-center QAP), live run with research | 0.4773 | v2 + final brain 0.8020 | +0.3247 (+68.0%) | done | $0.0208 (89 calls) |
| QAPLIB, live run 3 (Shared Brain only) | 0.8529 | v0 + final brain 0.9300 | +0.0771 (+9.0%) | done | $0.0111 (54 calls) |
| QAPLIB, live run with research on the fixed runner | 0.6741 | v1 + final brain 0.7537 | +0.0796 (+11.8%) | done | $0.0208 (102 calls) |
| GPU rack placement (data-center QAP), research on the fixed runner | 0.5973 | v3 + final brain 0.8452 | +0.2479 (+41.5%) | done | $0.0477 (112 calls) |
| GPU rack placement (data-center QAP), research, 4 generations | 0.4295 | v3 + final brain 0.9213 | +0.4918 (+114.5%) | done | $0.0737 (159 calls) |
| ALE-Bench AHC039 (Purse Seine Fishing) | n/a | n/a | n/a | blocked: crashed: RateLimitError, HTTP 429, max_requests_limit; 1 candidate scored before the crash; last write 15:09 PDT | n/a (crashed before usage.json) |
| KernelBench mini (CUDA kernels) | n/a | n/a | n/a | blocked: needs a Crusoe GPU VM (no runs/kernelbench-* yet) | n/a |
| KernelBench mini on Apple GPU (MPS) | 0.195 | v0 + final brain 0.439 | +0.244 (+125.6%) | done | $0.1908 (98 calls) |
| Inference serving cost (exact-answer data-center ops Q&A) | $0.001092 per correct answer | v1 + final brain (final) $0.000753 per correct answer | 31% less per correct answer (48 held-out questions) | done | $0.2650 (208 calls) |
| Model settings: Nano vs Super, reasoning on/off (4 train halls) | n/a (settings sweep, no v0 arm) | B32 (Nano 30B, reasoning on, max 32k tokens) 0.7216 | n/a | done | $0.2473 (127 calls) |

QAPLIB and rack placement: a 10 s 2-swap search scores 0.995 / 0.967 on the same tasks.
Note, Data hall layout, live run 5: The Reviewer switched on a verifier-guided search tool (polish); the gate kept it at +3.30 dev. Held-out 0.688 → 0.728 on 6 unseen halls. Greedy + polish with no LLM scores the same, so the gain is the tool's; the team chose it.
Note, GPU rack placement (data-center QAP), live run with research: v0 cold saved a solution for 5 of the 6 held-out halls. The missing hall is scored as invalid (score 0), so 0.4773 is the mean over all 6 with that hall at 0 (the 5 saved halls average 0.5728).
Note, QAPLIB, live run with research on the fixed runner: v0 cold found no valid permutation on tai15a in either attempt, so it scores 0 there. Without tai15a, v0 cold averages 0.8089 on the other 5 and v1 0.7410 (-0.068): no gain on QAPLIB.
Note, GPU rack placement (data-center QAP), research, 4 generations: v0 cold found no valid placement on dc-large-10035 in either attempt (score 0). Without that hall, v0 cold averages 0.5154 on the other 5 and v3 0.9138 (+0.398); v3 is higher on all 6 halls.
Note, KernelBench mini on Apple GPU (MPS): correct kernels 20% -> 60% (v0 cold -> final). No kernel beat PyTorch eager (fast_1 0%); MPS timings are not comparable to CUDA. Full numbers: reports/kernelbench_mps/SUMMARY.md.
Note, Inference serving cost (exact-answer data-center ops Q&A): accuracy 47/48 -> 48/48 is one question, so no accuracy gain is claimed; the headline is cost per correct answer. Median call latency 9.7 s -> 19.0 s; time per hall 163 s -> 77 s.

## Data hall layout, live run 1

- Verifier: DoneTwin checker: hard feasibility rules (collision, 2N power, CDU capacity, hose reach, zones); score = DoneTwin score / 100, pass = feasible.
- Baseline: v0 cold (v0 + empty brain (cold)), held-out mean 0.710 (source: held-out table).
- Status: done; final report and live usage written.
- Live evidence: usage.json: 475 paid calls, $0.1730.
- Totals: 475 calls, 1,974,292 tokens (1,523,865 prompt + 450,427 completion, 0 cached), $0.1730, 0 errors, wall 1353.7 s.
- By model: NVIDIA-Nemotron-3-Nano-30B-A3B 472 calls $0.1656; NVIDIA-Nemotron-3-Super-120B-A12B 3 calls $0.0075.

### Held-out

| Config | n | Mean score | Pass rate | Tokens | Cost (USD) | vs v0 cold |
|---|---|---|---|---|---|---|
| v0 + empty brain (cold) | 15 | 0.710 | 100% | 223,921 | $0.0197 | baseline |
| v0 + final brain | 15 | 0.710 | 100% | 257,007 | $0.0227 | +0.000 (+0.0%) |

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
- Baseline: v0 cold (v0 + empty brain (cold)), held-out mean 0.688 (source: held-out table).
- Status: done; final report and live usage written.
- Live evidence: usage.json: 128 paid calls, $0.0433.
- Totals: 128 calls, 466,832 tokens (357,091 prompt + 109,741 completion, 0 cached), $0.0433, 0 errors, wall 512.1 s.
- By model: NVIDIA-Nemotron-3-Nano-30B-A3B 127 calls $0.0394; NVIDIA-Nemotron-3-Super-120B-A12B 1 calls $0.0039.

### Held-out

| Config | n | Mean score | Pass rate | Tokens | Cost (USD) | vs v0 cold |
|---|---|---|---|---|---|---|
| v0 + empty brain (cold) | 6 | 0.688 | 100% | 97,150 | $0.0087 | baseline |
| v0 + final brain | 6 | 0.688 | 100% | 111,536 | $0.0114 | +0.000 (+0.0%) |

Note: 'v0 + final brain' appears twice in the final report: no candidate passed the gate, so the final version is v0 and its arm repeats 'v0 + final brain'; shown once.

Dev mean of the parent by generation: gen 0 0.692.

### Generations (each candidate is gated on the dev set)

3 candidates, 0 accepted; claim vs measured verdicts: 3 not reproduced.

| Gen | Mutation | Change | Claim | Measured dev gain | Decision | Why |
|---|---|---|---|---|---|---|
| 0 | tool_set | tool capacity_calc enabled=True | dev gain +0.040, cost -20.0%: Provides quick capacity checks, reducing Layout reasoning tokens an... | +0.0000 | rejected | gain +0.000 < 0.02; cost -2% misses the -15% cut |
| 0 | tool_set | tool nearest_examples enabled=True | dev gain +0.030, cost -10.0%: Gives Layout concrete patterns to avoid collisions and hose issues,... | +0.0009 | rejected | gain +0.001 < 0.02; cost +2% misses the -15% cut |
| 0 | playbook_add | Layout step: After proposing a rack placement, invoke capacity_calc to verify bu... | dev gain +0.050, cost -5.0%: Early capacity verification reduces rework and iterations, improvin... | +0.0000 | rejected | gain +0.000 < 0.02; cost +11% misses the -15% cut |

Sources: `runs/datahall-live-2/reports/before_after.json`, `runs/datahall-live-2/reports/vs_greedy.md`, `runs/datahall-live-2/reports/usage.json`, `runs/datahall-live-2/versions/ledger.jsonl`, `runs/datahall-live-2.log`.

## Data hall layout, live run 2d

- Verifier: DoneTwin checker: hard feasibility rules (collision, 2N power, CDU capacity, hose reach, zones); score = DoneTwin score / 100, pass = feasible.
- Baseline: v0 cold (v0 + empty brain (cold)), held-out mean 0.688 (source: held-out table).
- Status: done; final report and live usage written.
- Live evidence: usage.json: 109 paid calls, $0.0794.
- Totals: 109 calls, 621,901 tokens (300,028 prompt + 321,873 completion, 0 cached), $0.0794, 0 errors, wall 1105.9 s.
- By model: NVIDIA-Nemotron-3-Nano-30B-A3B 109 calls $0.0794.

### Held-out

| Config | n | Mean score | Pass rate | Tokens | Cost (USD) | vs v0 cold |
|---|---|---|---|---|---|---|
| v0 + empty brain (cold) | 6 | 0.688 | 100% | 84,732 | $0.0073 | baseline |
| v0 + final brain | 6 | 0.688 | 100% | 113,543 | $0.0114 | +0.000 (+0.0%) |

Note: 'v0 + final brain' appears twice in the final report: no candidate passed the gate, so the final version is v0 and its arm repeats 'v0 + final brain'; shown once.

Dev mean of the parent by generation: gen 0 0.692.

### Generations (each candidate is gated on the dev set)

3 candidates, 0 accepted; claim vs measured verdicts: 3 not reproduced.

| Gen | Mutation | Change | Claim | Measured dev gain | Decision | Why |
|---|---|---|---|---|---|---|
| 0 | playbook_add | Layout step: Validate each rack's assigned CDU has a hose length within Manhatta... | dev gain +0.020, cost -0.1%: Improves feasibility by reducing hose violations without increasing... | +0.0000 | rejected | gain +0.000 < 0.02; cost -1% misses the -15% cut |
| 0 | config_set | thinking = {"Layout": true} | dev gain +0.030, cost -0.1%: Enables reasoning improvements; cost impact minimal due to unchange... | +0.0047 | rejected | gain +0.005 < 0.02; cost +1190% misses the -15% cut |
| 0 | config_set | max_tokens = 32000 | dev gain +0.015, cost +0.0%: Prevents truncated responses that cause 'no_improvement' failures;... | +0.0000 | rejected | gain +0.000 < 0.02; cost +0% misses the -15% cut |

Sources: `runs/datahall-live-2d/reports/before_after.json`, `runs/datahall-live-2d/reports/vs_greedy.md`, `runs/datahall-live-2d/reports/usage.json`, `runs/datahall-live-2d/versions/ledger.jsonl`, `runs/datahall-live-2d.log`.

## Data hall layout, live run 3

- Verifier: DoneTwin checker: hard feasibility rules (collision, 2N power, CDU capacity, hose reach, zones); score = DoneTwin score / 100, pass = feasible.
- Baseline: v0 cold (v0 + empty brain (cold)), held-out mean 0.688 (source: held-out table).
- Status: done; final report and live usage written.
- Live evidence: usage.json: 129 paid calls, $0.0385.
- Totals: 129 calls, 454,799 tokens (349,625 prompt + 105,174 completion, 0 cached), $0.0385, 0 errors, wall 585.6 s.
- By model: NVIDIA-Nemotron-3-Nano-30B-A3B 129 calls $0.0385.

### Held-out

| Config | n | Mean score | Pass rate | Tokens | Cost (USD) | vs v0 cold |
|---|---|---|---|---|---|---|
| v0 + empty brain (cold) | 6 | 0.688 | 100% | 86,955 | $0.0075 | baseline |
| v0 + final brain | 6 | 0.688 | 100% | 80,679 | $0.0058 | +0.000 (+0.0%) |

Note: 'v0 + final brain' appears twice in the final report: no candidate passed the gate, so the final version is v0 and its arm repeats 'v0 + final brain'; shown once.

Dev mean of the parent by generation: gen 0 0.692.

### Generations (each candidate is gated on the dev set)

4 candidates, 0 accepted; claim vs measured verdicts: 3 not reproduced.

| Gen | Mutation | Change | Claim | Measured dev gain | Decision | Why |
|---|---|---|---|---|---|---|
| 0 | None | {} | none recorded (ledger predates claims) | n/a | rejected | None |
| 0 | playbook_add | Layout step: Reassign each rack to the nearest CDU with remaining capacity, usin... | dev gain +0.020, cost -0.2%: Expected improvement in mean dev score by fixing the most frequent... | +0.0000 | rejected | gain +0.000 < 0.02; cost +4% misses the -15% cut |
| 0 | config_set | models.Layout = {"nvidia/NVIDIA-Nemotron-3-Super-120B-A12B": [0.3, 2.4]} | dev gain +0.015, cost -0.1%: Higher-capacity model may yield better layouts with fewer retries;... | +0.0000 | rejected | gain +0.000 < 0.02; cost -3% misses the -15% cut |
| 0 | config_set | capacity_calc = {"enabled": true} | dev gain +0.025, cost -0.2%: Accurate capacity calculations will eliminate cdu_capacity violatio... | +0.0009 | rejected | gain +0.001 < 0.02; cost +167% misses the -15% cut |

Sources: `runs/datahall-live-3/reports/before_after.json`, `runs/datahall-live-3/reports/vs_greedy.md`, `runs/datahall-live-3/reports/usage.json`, `runs/datahall-live-3/versions/ledger.jsonl`, `runs/datahall-live-3.log`.

## Data hall layout, live run 5

- Verifier: DoneTwin checker: hard feasibility rules (collision, 2N power, CDU capacity, hose reach, zones); score = DoneTwin score / 100, pass = feasible.
- Baseline: v0 cold (v0 + empty brain (cold)), held-out mean 0.688 (source: held-out table).
- Status: done; final report and live usage written.
- Live evidence: usage.json: 152 paid calls, $0.0468.
- Totals: 152 calls, 558,505 tokens (445,706 prompt + 112,799 completion, 0 cached), $0.0468, 0 errors, wall 589.7 s.
- By model: NVIDIA-Nemotron-3-Nano-30B-A3B 151 calls $0.0446; NVIDIA-Nemotron-3-Super-120B-A12B 1 calls $0.0022.

### Held-out

| Config | n | Mean score | Pass rate | Tokens | Cost (USD) | vs v0 cold |
|---|---|---|---|---|---|---|
| v0 + empty brain (cold) | 6 | 0.688 | 100% | 97,457 | $0.0087 | baseline |
| v0 + final brain | 6 | 0.688 | 100% | 99,439 | $0.0085 | +0.000 (+0.0%) |
| v1 + final brain | 6 | 0.728 | 100% | 95,055 | $0.0075 | +0.040 (+5.8%) |

Note: The Reviewer switched on a verifier-guided search tool (polish); the gate kept it at +3.30 dev. Held-out 0.688 → 0.728 on 6 unseen halls. Greedy + polish with no LLM scores the same, so the gain is the tool's; the team chose it.

Dev mean of the parent by generation: gen 0 0.692.

### Generations (each candidate is gated on the dev set)

4 candidates, 1 accepted; claim vs measured verdicts: 2 not reproduced, 1 reproduced.

| Gen | Mutation | Change | Claim | Measured dev gain | Decision | Why |
|---|---|---|---|---|---|---|
| 0 | None | {} | none recorded (ledger predates claims) | n/a | rejected | None |
| 0 | config_set | polish = true | dev gain +0.050, cost -10.0%: Polish improves feasibility and score via cheap verifier-guided mov... | +0.0330 | accepted (score) | accepted |
| 0 | config_set | polish_checks = 800 | dev gain +0.020, cost -15.0%: Fewer polish checks reduce verifier tool usage and cost, with minim... | +0.0009 | rejected | gain +0.001 < 0.02; cost -14% misses the -15% cut |
| 0 | config_set | temperature = 0.0 | dev gain +0.030, cost -5.0%: Deterministic outputs reduce wasted exploration in layout generatio... | +0.0000 | rejected | gain +0.000 < 0.02; cost -3% misses the -15% cut |

Sources: `runs/datahall-live-5/reports/before_after.json`, `runs/datahall-live-5/reports/vs_greedy.md`, `runs/datahall-live-5/reports/usage.json`, `runs/datahall-live-5/versions/ledger.jsonl`, `runs/datahall-live-5.log`.

## QAPLIB (quadratic assignment)

- Verifier: exact QAP cost of the printed permutation; score = BKS / cost (1.0 = best known), pass = gap to BKS <= 1%.
- Baseline: v0 cold (v0 cold), held-out mean n/a (source: pending: the run's held-out table).
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

## QAPLIB, live run with research

- Verifier: exact QAP cost of the printed permutation; score = BKS / cost (1.0 = best known), pass = gap to BKS <= 1%.
- Baseline: v0 cold (v0 + empty brain (cold)), held-out mean 0.9593 (source: held-out table).
- Status: done; final report and live usage written.
- Live evidence: usage.json: 89 paid calls, $0.0163.
- Totals: 89 calls, 154,477 tokens (97,346 prompt + 57,131 completion, 0 cached), $0.0163, 0 errors, wall 507.3 s.
- By model: NVIDIA-Nemotron-3-Nano-30B-A3B 89 calls $0.0163.

### Held-out

| Config | n | Mean score | Pass rate | Gap to BKS (%) | Tokens | Cost (USD) | vs v0 cold |
|---|---|---|---|---|---|---|---|
| v0 + empty brain (cold) | 6 | 0.9593 | 33% | 4.33 | 17,037 | $0.0021 | baseline |
| v0 + final brain | 6 | 0.8738 | 0% | 15.10 | 20,147 | $0.0021 | -0.0855 (-8.9%) |
| v2 + final brain | 6 | 0.9662 | 33% | 3.63 | 30,028 | $0.0033 | +0.0069 (+0.7%) |

Dev mean of the parent by generation: gen 0 0.8868, gen 1 0.9122.

### Generations (each candidate is gated on the dev set)

8 candidates, 2 accepted; claim vs measured verdicts: 2 not reproduced, 1 reproduced, 1 accepted, effect smaller than claimed.

| Gen | Mutation | Change | Claim | Measured dev gain | Decision | Why |
|---|---|---|---|---|---|---|
| 0 | None | {} | none recorded (ledger predates claims) | n/a | rejected | None |
| 0 | playbook_add | None step: Validate output permutation format and checksum before returning. | dev gain +0.050, cost -0.1%: Verifying permutations prevents syntax/runtime errors observed in f... | n/a | rejected | invalid: playbook_add missing ['role'] |
| 0 | config_set | models.solver = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B" | dev gain +0.070, cost -0.1%: Using the larger 120B model (nvidia/NVIDIA-Nemotron-3-Super-120B-A1... | -0.0820 | rejected | gain -0.082 < 0.003 |
| 0 | config_set | thinking.solver = true | dev gain +0.040, cost -0.1%: Activating thinking (reasoning) for the solver role is expected to... | +0.0728 | accepted (score) | accepted |
| 1 | None | {} | none recorded (ledger predates claims) | n/a | rejected | None |
| 1 | playbook_remove | {"index": 0} | dev gain +0.015, cost -0.2%: Eliminating simulated_annealing reduces runtime errors and large ga... | n/a | rejected | invalid: playbook_remove missing ['role'] |
| 1 | config_set | max_tokens = 8000 | dev gain +0.020, cost -0.1%: Increasing max_tokens to 8000 allows longer chain-of-thought withou... | -0.0112 | rejected | gain -0.011 < 0.003 |
| 1 | config_set | temperature = 0.2 | dev gain +0.010, cost -0.1%: Lowering temperature to 0.2 sharpens output consistency, reducing g... | +0.0365 | accepted (score) | accepted |

Sources: `reports/qaplib_live/research1_nano/before_after.json`, `reports/qaplib_live/research1_nano/vs_2swap.md`, `reports/qaplib_live/research1_nano/qaplib_rows.json`, `reports/qaplib_live/research1_nano/usage.json`, `reports/qaplib_live/research1_nano/version_ledger.jsonl`, `reports/qaplib_live/research1_nano/run.log`.

## GPU rack placement (data-center QAP), live run 1

- Verifier: exact traffic x cable-length cost of the rack placement on generated GPU halls; score = reference cost / cost.
- Baseline: v0 cold (v0 + empty brain (cold)), held-out mean 0.5042 (source: held-out table).
- Status: done; final report and live usage written.
- Live evidence: usage.json: 96 paid calls, $0.0168.
- Totals: 96 calls, 186,817 tokens (136,994 prompt + 49,823 completion, 0 cached), $0.0168, 0 errors, wall 574.4 s.
- By model: NVIDIA-Nemotron-3-Nano-30B-A3B 96 calls $0.0168.

### Held-out

| Config | n | Mean score | Pass rate | Gap to BKS (%) | Tokens | Cost (USD) | vs v0 cold |
|---|---|---|---|---|---|---|---|
| v0 + empty brain (cold) | 6 | 0.5042 | 0% | 146.21 | 15,899 | $0.0018 | baseline |
| v0 + final brain | 6 | 0.5865 | 0% | 87.54 | 27,586 | $0.0022 | +0.0823 (+16.3%) |
| v1 + final brain | 6 | 0.6394 | 0% | 86.48 | 27,256 | $0.0022 | +0.1352 (+26.8%) |

Dev mean of the parent by generation: gen 0 0.8041, gen 1 0.7151.

### Generations (each candidate is gated on the dev set)

6 candidates, 1 accepted; claim vs measured verdicts: 4 not reproduced, 1 reproduced.

| Gen | Mutation | Change | Claim | Measured dev gain | Decision | Why |
|---|---|---|---|---|---|---|
| 0 | playbook_add | solver step: Validate permutation length matches problem size before execution | dev gain +0.020, cost -0.1%: Prevents runtime errors and reduces wasted compute on invalid solut... | -0.2424 | rejected | gain -0.242 < 0.003 |
| 0 | config_set | max_tokens = 4000 | dev gain +0.010, cost -0.1%: Reduces token usage while keeping solution length sufficient, cutti... | -0.2168 | rejected | gain -0.217 < 0.003 |
| 0 | config_set | temperature = 0.1 | dev gain +0.015, cost -0.1%: Lower temperature yields more deterministic outputs, improving veri... | -0.3957 | rejected | gain -0.396 < 0.003 |
| 1 | playbook_remove | {"role": "solver", "index": 0} | dev gain +0.005, cost -0.1%: Removes a generic heuristic that consistently yields high gaps, for... | n/a | rejected | invalid: playbook index 0 out of range |
| 1 | config_set | max_tokens = 4000 | dev gain +0.003, cost -0.1%: Reduces token consumption per call, lowering cost per task by rough... | -0.2409 | rejected | gain -0.241 < 0.003 |
| 1 | config_set | temperature = 0.1 | dev gain +0.004, cost -0.1%: Lower temperature yields more deterministic outputs, improving solu... | +0.0933 | accepted (score) | accepted |

Sources: `reports/qaplib_live/dc1_nano/before_after.json`, `reports/qaplib_live/dc1_nano/vs_2swap.md`, `reports/qaplib_live/dc1_nano/qaplib_rows.json`, `reports/qaplib_live/dc1_nano/usage.json`, `reports/qaplib_live/dc1_nano/version_ledger.jsonl`, `reports/qaplib_live/dc1_nano/run.log`.

## GPU rack placement (data-center QAP), live run with research

- Verifier: exact traffic x cable-length cost of the rack placement on generated GPU halls; score = reference cost / cost.
- Baseline: v0 cold (v0 + empty brain (cold)), held-out mean 0.4773 (source: held-out table).
- Status: done; final report and live usage written.
- Live evidence: usage.json: 89 paid calls, $0.0208.
- Totals: 89 calls, 197,961 tokens (125,049 prompt + 72,912 completion, 0 cached), $0.0208, 0 errors, wall 513.7 s.
- By model: NVIDIA-Nemotron-3-Nano-30B-A3B 89 calls $0.0208.

### Held-out

| Config | n | Mean score | Pass rate | Gap to BKS (%) | Tokens | Cost (USD) | vs v0 cold |
|---|---|---|---|---|---|---|---|
| v0 + empty brain (cold) | 6 | 0.4773 | 17% | 117.05 | 15,206 | $0.0018 | baseline |
| v0 + final brain | 6 | 0.7132 | 0% | 50.77 | 27,526 | $0.0023 | +0.2359 (+49.4%) |
| v2 + final brain | 6 | 0.8020 | 0% | 28.57 | 46,241 | $0.0062 | +0.3247 (+68.0%) |

Note: v0 cold saved a solution for 5 of the 6 held-out halls. The missing hall is scored as invalid (score 0), so 0.4773 is the mean over all 6 with that hall at 0 (the 5 saved halls average 0.5728).

Dev mean of the parent by generation: gen 0 0.5707, gen 1 0.6759.

### Generations (each candidate is gated on the dev set)

8 candidates, 3 accepted; claim vs measured verdicts: 2 reproduced, 1 not reproduced, 1 accepted, effect smaller than claimed.

| Gen | Mutation | Change | Claim | Measured dev gain | Decision | Why |
|---|---|---|---|---|---|---|
| 0 | None | {} | none recorded (ledger predates claims) | n/a | rejected | None |
| 0 | playbook_add | None step: Run tabu search with aspiration threshold and 2-opt neighborhood fo... | dev gain +0.050, cost -0.1%: Based on [1] which shows tabu search reduces gap by ~30% and improv... | n/a | rejected | invalid: playbook_add missing ['role'] |
| 0 | config_set | max_tokens = 8000 | dev gain +0.020, cost -0.1%: Increasing max_tokens to 8000 allows longer reasoning without trunc... | +0.2740 | accepted (score) | accepted |
| 0 | config_set | temperature = 0.1 | dev gain +0.010, cost -0.1%: Lower temperature reduces stochasticity, leading to more determinis... | +0.2413 | accepted (score) | accepted |
| 1 | None | {} | none recorded (ledger predates claims) | n/a | rejected | None |
| 1 | playbook_add | None step: Run tabu search with adaptive neighborhood and restart after 5 iter... | dev gain +0.050, cost -0.1%: Improves solution quality by reducing gap; modest cost increase fro... | n/a | rejected | invalid: playbook_add missing ['role'] |
| 1 | config_set | max_tokens = 12000 | dev gain +0.020, cost -0.1%: Higher token budget improves solution correctness; slight cost redu... | -0.1178 | rejected | gain -0.118 < 0.003 |
| 1 | config_set | thinking = {"solver": true} | dev gain +0.030, cost -0.1%: Thinking improves heuristic design accuracy; reduces need for multi... | +0.1463 | accepted (score) | accepted |

Sources: `reports/qaplib_live/dc_research1_nano/before_after.json`, `reports/qaplib_live/dc_research1_nano/vs_2swap.md`, `reports/qaplib_live/dc_research1_nano/qaplib_rows.json`, `reports/qaplib_live/dc_research1_nano/usage.json`, `reports/qaplib_live/dc_research1_nano/version_ledger.jsonl`, `reports/qaplib_live/dc_research1_nano/run.log`.

## QAPLIB, live run 3 (Shared Brain only)

- Verifier: exact QAP cost of the printed permutation; score = BKS / cost (1.0 = best known), pass = gap to BKS <= 1%.
- Baseline: v0 cold (v0 + empty brain (cold)), held-out mean 0.8529 (source: held-out table).
- Status: done; final report and live usage written.
- Live evidence: usage.json: 54 paid calls, $0.0111.
- Totals: 54 calls, 105,820 tokens (67,356 prompt + 38,464 completion, 0 cached), $0.0111, 0 errors, wall 368.5 s.
- By model: NVIDIA-Nemotron-3-Nano-30B-A3B 54 calls $0.0111.

### Held-out

| Config | n | Mean score | Pass rate | Gap to BKS (%) | Tokens | Cost (USD) | vs v0 cold |
|---|---|---|---|---|---|---|---|
| v0 + empty brain (cold) | 6 | 0.8529 | 0% | 17.93 | 15,959 | $0.0019 | baseline |
| v0 + final brain | 6 | 0.9300 | 33% | 8.38 | 24,888 | $0.0021 | +0.0771 (+9.0%) |

Note: 'v0 + final brain' appears twice in the final report: no candidate passed the gate, so the final version is v0 and its arm repeats 'v0 + final brain'; shown once.

Dev mean of the parent by generation: gen 0 0.9725, gen 1 0.9206.

Sources: `reports/qaplib_live/run3_nano/before_after.json`, `reports/qaplib_live/run3_nano/vs_2swap.md`, `reports/qaplib_live/run3_nano/qaplib_rows.json`, `reports/qaplib_live/run3_nano/usage.json`, `reports/qaplib_live/run3_nano/run.log`.

## QAPLIB, live run with research on the fixed runner

- Verifier: exact QAP cost of the printed permutation; score = BKS / cost (1.0 = best known), pass = gap to BKS <= 1%.
- Baseline: v0 cold (v0 + empty brain (cold)), held-out mean 0.6741 (source: held-out table).
- Status: done; final report and live usage written.
- Live evidence: usage.json: 102 paid calls, $0.0208.
- Totals: 102 calls, 236,679 tokens (177,067 prompt + 59,612 completion, 0 cached), $0.0208, 0 errors, wall 924.4 s.
- By model: NVIDIA-Nemotron-3-Nano-30B-A3B 102 calls $0.0208.

### Held-out

| Config | n | Mean score | Pass rate | Gap to BKS (%) | Tokens | Cost (USD) | vs v0 cold |
|---|---|---|---|---|---|---|---|
| v0 + empty brain (cold) | 6 | 0.6741 | 17% | 28.11 | 17,350 | $0.0020 | baseline |
| v0 + final brain | 6 | 0.6825 | 0% | 26.72 | 32,324 | $0.0026 | +0.0084 (+1.2%) |
| v1 + final brain | 6 | 0.7537 | 0% | 43.58 | 33,830 | $0.0028 | +0.0796 (+11.8%) |

Note: v0 cold found no valid permutation on tai15a in either attempt, so it scores 0 there. Without tai15a, v0 cold averages 0.8089 on the other 5 and v1 0.7410 (-0.068): no gain on QAPLIB.

Dev mean of the parent by generation: gen 0 0.9211, gen 1 0.8992.

### Generations (each candidate is gated on the dev set)

8 candidates, 1 accepted; claim vs measured verdicts: 5 not reproduced, 1 accepted, effect smaller than claimed.

| Gen | Mutation | Change | Claim | Measured dev gain | Decision | Why |
|---|---|---|---|---|---|---|
| 0 | None | {} | none recorded (ledger predates claims) | n/a | rejected | None |
| 0 | playbook_add | solver step: Add a verification step that checks the permutation satisfies the Q... | dev gain +0.020, cost -0.1%: Implementing a constraint check (as described in [2]) reduces faile... | +0.0282 | accepted (score) | accepted |
| 0 | config_set | max_tokens = 8000 | dev gain +0.015, cost -0.1%: Increasing max_tokens to 8000 (within the 256‑32768 range) gives th... | -0.0977 | rejected | gain -0.098 < 0.003 |
| 0 | config_set | temperature = 0.1 | dev gain +0.018, cost -0.1%: Lowering temperature to 0.1 makes outputs more deterministic, impro... | -0.4195 | rejected | gain -0.420 < 0.003 |
| 1 | None | {} | none recorded (ledger predates claims) | n/a | rejected | None |
| 1 | playbook_add | solver step: Add a verification step that checks the permutation satisfies the Q... | dev gain +0.028, cost -0.1%: Based on v1 recent change [1] which reported a dev gain of +0.028 a... | -0.0046 | rejected | gain -0.005 < 0.003 |
| 1 | config_set | max_tokens = 4000 | dev gain +0.015, cost -0.1%: Reducing max_tokens from 6000 to 4000 cuts token cost while preserv... | -0.4680 | rejected | gain -0.468 < 0.003 |
| 1 | config_set | temperature = 0.2 | dev gain +0.012, cost -0.1%: Lowering temperature to 0.2 reduces stochastic variation, improving... | -0.0348 | rejected | gain -0.035 < 0.003 |

Sources: `reports/qaplib_live/research2_nano/before_after.json`, `reports/qaplib_live/research2_nano/vs_2swap.md`, `reports/qaplib_live/research2_nano/qaplib_rows.json`, `reports/qaplib_live/research2_nano/usage.json`, `reports/qaplib_live/research2_nano/version_ledger.jsonl`, `reports/qaplib_live/research2_nano/run.log`.

## GPU rack placement (data-center QAP), research on the fixed runner

- Verifier: exact traffic x cable-length cost of the rack placement on generated GPU halls; score = reference cost / cost.
- Baseline: v0 cold (v0 + empty brain (cold)), held-out mean 0.5973 (source: held-out table).
- Status: done; final report and live usage written.
- Live evidence: usage.json: 112 paid calls, $0.0477.
- Totals: 112 calls, 394,859 tokens (208,553 prompt + 186,306 completion, 0 cached), $0.0477, 1 errors, wall 1260.5 s.
- By model: NVIDIA-Nemotron-3-Nano-30B-A3B 112 calls $0.0477.

### Held-out

| Config | n | Mean score | Pass rate | Gap to BKS (%) | Tokens | Cost (USD) | vs v0 cold |
|---|---|---|---|---|---|---|---|
| v0 + empty brain (cold) | 6 | 0.5973 | 0% | 92.63 | 18,625 | $0.0021 | baseline |
| v0 + final brain | 6 | 0.6454 | 0% | 71.01 | 34,558 | $0.0027 | +0.0481 (+8.1%) |
| v3 + final brain | 6 | 0.8452 | 0% | 18.88 | 52,959 | $0.0064 | +0.2479 (+41.5%) |

Dev mean of the parent by generation: gen 0 0.7129, gen 1 0.8561, gen 2 0.8561.

### Generations (each candidate is gated on the dev set)

12 candidates, 4 accepted; claim vs measured verdicts: 3 accepted, effect smaller than claimed, 2 not reproduced, 1 reproduced.

| Gen | Mutation | Change | Claim | Measured dev gain | Decision | Why |
|---|---|---|---|---|---|---|
| 0 | None | {} | none recorded (ledger predates claims) | n/a | rejected | None |
| 0 | playbook_remove | {"role": "solver", "index": 0} | dev gain +0.000, cost -15.0%: Removing the first playbook step eliminates the costly runtime_erro... | n/a | rejected | invalid: playbook index 0 out of range |
| 0 | config_set | max_tokens = {"solver": 4000} | dev gain +0.020, cost -12.0%: Reducing max_tokens to 4000 prevents truncation of longer heuristic... | n/a | rejected | invalid: max_tokens must be an int in [256, 32768], got {'solver': 4000} |
| 0 | config_set | thinking = {"solver": true} | dev gain +0.015, cost -18.0%: Enabling thinking for solver adds reasoning that can catch budget m... | +0.0918 | accepted (score) | accepted |
| 1 | None | {} | none recorded (ledger predates claims) | n/a | rejected | None |
| 1 | config_set | max_tokens = 4000 | dev gain +0.015, cost -0.2%: Reducing max_tokens from 6000 to 4000 cuts token cost and may impro... | +0.0327 | accepted (score) | accepted |
| 1 | config_set | temperature = 0.1 | dev gain +0.012, cost -0.1%: Lower temperature reduces stochastic variation and may improve cons... | -0.3716 | rejected | gain -0.372 < 0.003 |
| 1 | config_set | attempts = 1 | dev gain +0.008, cost -0.3%: Running only one attempt cuts cost per task by ~33% while keeping d... | -0.0150 | rejected | gain -0.015 < 0.003 |
| 2 | None | {} | none recorded (ledger predates claims) | n/a | rejected | None |
| 2 | playbook_remove | {"role": "solver", "index": 0} | dev gain +0.012, cost -0.2%: Removes the only playbook step that repeatedly triggers runtime err... | n/a | rejected | invalid: playbook index 0 out of range |
| 2 | config_set | max_tokens = 8000 | dev gain +0.025, cost -0.1%: Doubling max_tokens to 8000 allows full reasoning traces without tr... | +0.0748 | accepted (score) | accepted |
| 2 | playbook_add | solver step: if runtime_error: abort after 0.9s and fallback to 2‑swap | dev gain +0.018, cost -0.1%: Adds early abort logic for runtime errors to prevent wasting the fu... | +0.0443 | accepted (score) | accepted |

Sources: `reports/qaplib_live/dc_research2_nano/before_after.json`, `reports/qaplib_live/dc_research2_nano/vs_2swap.md`, `reports/qaplib_live/dc_research2_nano/qaplib_rows.json`, `reports/qaplib_live/dc_research2_nano/usage.json`, `reports/qaplib_live/dc_research2_nano/version_ledger.jsonl`, `reports/qaplib_live/dc_research2_nano/run.log`.

## GPU rack placement (data-center QAP), research, 4 generations

- Verifier: exact traffic x cable-length cost of the rack placement on generated GPU halls; score = reference cost / cost.
- Baseline: v0 cold (v0 + empty brain (cold)), held-out mean 0.4295 (source: held-out table).
- Status: done; final report and live usage written.
- Live evidence: usage.json: 159 paid calls, $0.0737.
- Totals: 159 calls, 617,668 tokens (332,512 prompt + 285,156 completion, 0 cached), $0.0737, 0 errors, wall 1419.5 s.
- By model: NVIDIA-Nemotron-3-Nano-30B-A3B 159 calls $0.0737.

### Held-out

| Config | n | Mean score | Pass rate | Gap to BKS (%) | Tokens | Cost (USD) | vs v0 cold |
|---|---|---|---|---|---|---|---|
| v0 + empty brain (cold) | 6 | 0.4295 | 0% | 138.88 | 17,200 | $0.0019 | baseline |
| v0 + final brain | 6 | 0.6260 | 0% | 87.12 | 34,695 | $0.0026 | +0.1965 (+45.8%) |
| v3 + final brain | 6 | 0.9213 | 17% | 8.95 | 54,897 | $0.0067 | +0.4918 (+114.5%) |

Note: v0 cold found no valid placement on dc-large-10035 in either attempt (score 0). Without that hall, v0 cold averages 0.5154 on the other 5 and v3 0.9138 (+0.398); v3 is higher on all 6 halls.

Dev mean of the parent by generation: gen 0 0.8039, gen 1 0.5668, gen 2 0.5993, gen 3 0.9356.

### Generations (each candidate is gated on the dev set)

16 candidates, 6 accepted; claim vs measured verdicts: 5 not reproduced, 4 reproduced, 2 accepted, effect smaller than claimed.

| Gen | Mutation | Change | Claim | Measured dev gain | Decision | Why |
|---|---|---|---|---|---|---|
| 0 | None | {} | none recorded (ledger predates claims) | n/a | rejected | None |
| 0 | playbook_add | solver step: Add a pre‑run sanity check that validates the permutation syntax an... | dev gain +0.020, cost -0.1%: Fixes the dc-medium-1381 UnboundLocalError failures, reducing waste... | -0.1092 | rejected | gain -0.109 < 0.003 |
| 0 | playbook_add | solver step: Replace Simulated annealing with a hybrid Tabu‑Search + 2‑opt local... | dev gain +0.040, cost -0.1%: Hybrid approach improves gap percentages (targeting ~12% gap vs 2‑s... | -0.4410 | rejected | gain -0.441 < 0.003 |
| 0 | config_set | thinking = {"solver": true} | dev gain +0.015, cost -0.1%: Enabling reasoning for the solver role adds systematic exploration,... | +0.0414 | accepted (score) | accepted |
| 1 | None | {} | none recorded (ledger predates claims) | n/a | rejected | None |
| 1 | prompt_edit | solver prompt: You write heuristic optimization programs for the Quadratic Assignm... | dev gain +0.020, cost -0.1%: Adds explicit error handling to prevent uncaught exceptions (e.g.,... | +0.3449 | accepted (score) | accepted |
| 1 | playbook_add | solver step: Validate input dimensions and raise a clear error if they are not s... | dev gain +0.015, cost -0.1%: Prevents invalid QAP instances from causing runtime errors; improve... | +0.1509 | accepted (score) | accepted |
| 1 | config_set | max_tokens = 8000 | dev gain +0.010, cost -0.2%: Increasing max_tokens to 8000 allows full reasoning traces without... | +0.2239 | accepted (score) | accepted |
| 2 | None | {} | none recorded (ledger predates claims) | n/a | rejected | None |
| 2 | playbook_remove | {"role": "solver", "index": 0} | dev gain +0.000, cost -0.1%: Removing the placeholder playbook eliminates unused steps, reducing... | n/a | rejected | invalid: playbook index 0 out of range |
| 2 | config_set | max_tokens = 4000 | dev gain +0.020, cost -0.2%: Reducing max_tokens from 6000 to 4000 cuts token consumption per ca... | +0.2358 | accepted (score) | accepted |
| 2 | config_set | temperature = 0.2 | dev gain +0.010, cost -0.1%: Lowering temperature to 0.2 makes outputs more deterministic, reduc... | +0.2048 | accepted (score) | accepted |
| 3 | None | {} | none recorded (ledger predates claims) | n/a | rejected | None |
| 3 | prompt_edit | solver prompt: You write heuristic optimization programs for the Quadratic Assignm... | dev gain +0.050, cost -0.1%: Enforcing valid permutation first and immediate error abort should... | +0.0000 | rejected | gain +0.000 < 0.003; cost +0% misses the -15% cut |
| 3 | config_set | max_tokens = 2500 | dev gain +0.020, cost -0.0%: Lower token ceiling reduces truncation failures observed in brain e... | -0.1313 | rejected | gain -0.131 < 0.003 |
| 3 | config_set | attempts = 3 | dev gain +0.040, cost -0.0%: More attempts allow retry after transient errors; cost impact is mi... | -0.0557 | rejected | gain -0.056 < 0.003 |

Sources: `reports/qaplib_live/dc_research3_nano/before_after.json`, `reports/qaplib_live/dc_research3_nano/vs_2swap.md`, `reports/qaplib_live/dc_research3_nano/qaplib_rows.json`, `reports/qaplib_live/dc_research3_nano/usage.json`, `reports/qaplib_live/dc_research3_nano/version_ledger.jsonl`, `reports/qaplib_live/dc_research3_nano/run.log`.

## ALE-Bench AHC039 (Purse Seine Fishing)

- Verifier: ALE-Bench AHC039 scorer on private Lite seeds; score = mean case score / greedy case score (1.0 = greedy); rank vs 583 contestants on all 150 private seeds.
- Baseline: v0 cold (v0 cold), held-out mean n/a (source: pending: the run's held-out table).
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
- Baseline: v0 cold (v0 cold), held-out mean n/a (source: pending: no live run).
- Status: blocked; needs a Crusoe GPU VM (no runs/kernelbench-* yet).

## KernelBench mini on Apple GPU (MPS)

- Verifier: correct vs the PyTorch reference on fresh inputs, then timed against PyTorch eager on the Mac's Apple GPU (MPS); hack guards reject copies, caching and input mutation; score = speedup if correct, else 0.
- Baseline: v0 cold (v0 + empty brain (cold)), held-out mean 0.195 (source: held-out table).
- Status: done; final report and live usage written.
- Live evidence: usage.json: 98 paid calls, $0.1908.
- Totals: 98 calls, 286,841 tokens (225,623 prompt + 61,218 completion, 91,008 cached), $0.1908, 0 errors, wall 461.9 s.
- By model: NVIDIA-Nemotron-3-Nano-30B-A3B 2 calls $0.0010; NVIDIA-Nemotron-3-Super-120B-A12B 96 calls $0.1898.

### Held-out

| Config | n | Mean score | Pass rate | Tokens | Cost (USD) | vs v0 cold |
|---|---|---|---|---|---|---|
| v0 + empty brain (cold) | 5 | 0.195 | n/a | 59,042 | $0.0496 | baseline |
| v0 + final brain | 5 | 0.439 | n/a | 68,781 | $0.0461 | +0.244 (+125.6%) |

Note: 'v0 + final brain' appears twice in the final report: no candidate passed the gate, so the final version is v0 and its arm repeats 'v0 + final brain'; shown once.

Note: correct kernels 20% -> 60% (v0 cold -> final). No kernel beat PyTorch eager (fast_1 0%); MPS timings are not comparable to CUDA. Full numbers: reports/kernelbench_mps/SUMMARY.md.

Dev mean of the parent by generation: gen 0 0.000, gen 1 0.000.

### Generations (each candidate is gated on the dev set)

2 candidates, 0 accepted.

| Gen | Mutation | Change | Claim | Measured dev gain | Decision | Why |
|---|---|---|---|---|---|---|
| 0 | None | {} | none recorded (ledger predates claims) | n/a | rejected | None |
| 1 | None | {} | none recorded (ledger predates claims) | n/a | rejected | None |

Sources: `runs/kernelbench-mps-2/reports/before_after.json`, `runs/kernelbench-mps-2/reports/usage.json`, `runs/kernelbench-mps-2/versions/ledger.jsonl`, `runs/kernelbench-mps-2.log`.

## Inference serving cost (exact-answer data-center ops Q&A)

- Verifier: exact numeric answers checked in code within a stated tolerance; score = accuracy, pass = accuracy >= 0.9; cost = Crusoe billed tokens x team.crusoe.PRICES (the run's goal is the same accuracy at lower cost; see reports/inference_opt_2/heldout_cost_latency.md).
- Baseline: v0 cold (v0 + empty brain (cold)), held-out mean 0.979 (source: held-out table).
- Status: done; final report and live usage written.
- Headline: $ per correct answer 0.001092 -> 0.000753, 31% less per correct answer (48 held-out questions; source: heldout_cost_latency.json).
- Live evidence: usage.json: 208 paid calls, $0.2650.
- Totals: 208 calls, 200,035 tokens (102,440 prompt + 97,595 completion, 0 cached), $0.2650, 1 errors, wall 927.9 s.
- By model: NVIDIA-Nemotron-3-Super-120B-A12B 208 calls $0.2650.

### Held-out

| Config | n | Mean score | Pass rate | Tokens | Cost (USD) | vs v0 cold |
|---|---|---|---|---|---|---|
| v0 + empty brain (cold) | 4 | 0.979 | 100% | 39,211 | $0.0513 | baseline |
| v0 + final brain | 4 | 0.979 | 100% | 42,903 | $0.0552 | +0.000 (+0.0%) |
| v1 + final brain | 4 | 1.000 | 100% | 24,971 | $0.0361 | +0.021 (+2.1%) |

Note: accuracy 47/48 -> 48/48 is one question, so no accuracy gain is claimed; the headline is cost per correct answer. Median call latency 9.7 s -> 19.0 s; time per hall 163 s -> 77 s.

Dev mean of the parent by generation: gen 0 1.000.

### Generations (each candidate is gated on the dev set)

2 candidates, 1 accepted; claim vs measured verdicts: 1 reproduced.

| Gen | Mutation | Change | Claim | Measured dev gain | Decision | Why |
|---|---|---|---|---|---|---|
| 0 | None | {} | none recorded (ledger predates claims) | n/a | rejected | None |
| 0 | config_set | batch_size = 3 | dev gain +0.000, cost -55.0%: Batching 3 questions per call reduces calls_per_task from ~12 to ~4... | +0.0000 | accepted (cost) | cheaper at equal score: cost -32% |

Sources: `runs/inference-opt-2/reports/before_after.json`, `runs/inference-opt-2/reports/usage.json`, `runs/inference-opt-2/versions/ledger.jsonl`, `runs/inference-opt-2.log`.

## Model settings: Nano vs Super, reasoning on/off

- Verifier: DoneTwin checker on 4 train halls (Layout role varies; Planner and Checker stay on Nano); score = DoneTwin score / 100.
- Status: done; 28 hall runs in 8 configs; last write 15:19 PDT.
- Cost so far: $0.2473 over 127 calls.

| Config | Layout model and settings | Run tag | Halls | Mean score | Feasible attempts | Calls/hall | Tokens/hall | Cost (USD) | Wall s/hall | Errors |
|---|---|---|---|---|---|---|---|---|---|---|
| A | Nano 30B, reasoning off, max 6k tokens | main | 4 | 0.6949 | 0/12 | 6.0 | 18,662 | $0.0056 | 138 | 0 |
| B | Nano 30B, reasoning on, max 16k tokens | main | 4 | 0.7084 | 1/12 | 4.0 | 56,528 | $0.0394 | 362 | 0 |
| B | Nano 30B, reasoning on, max 16k tokens | rep | 4 | 0.6949 | 0/12 | 4.0 | 57,760 | $0.0404 | 437 | 0 |
| B32 | Nano 30B, reasoning on, max 32k tokens | rep | 4 | 0.7216 | 3/12 | 4.5 | 97,594 | $0.0719 | 739 | 0 |
| C | Super 120B, reasoning off, max 6k tokens | main | 4 | 0.6949 | 0/12 | 6.0 | 22,340 | $0.0459 | 131 | 0 |
| D | Super 120B, reasoning on, max 16k tokens | main | 1 | 0.6981 | 0/1 | 2.0 | 19,726 | $0.0391 | 356 | 1 |
| F | Super 120B, reasoning off, max 6k tokens, small-edit playbook | pb | 3 | 0.6652 | 0/0 | 1.0 | 1,420 | $0.0003 | 112 | 3 |
| G | Nano 30B, reasoning off, max 6k tokens, small-edit playbook | pb | 4 | 0.6949 | 0/12 | 6.0 | 16,794 | $0.0048 | 99 | 0 |

Sources: `/private/tmp/claude-501/-Users-iamdaone-Documents-RecursiveAgentsHackathonProject/068a120f-637f-4002-856f-3eb6683d41a7/scratchpad/headroom/results_main.jsonl`, `/private/tmp/claude-501/-Users-iamdaone-Documents-RecursiveAgentsHackathonProject/068a120f-637f-4002-856f-3eb6683d41a7/scratchpad/headroom/results_pb.jsonl`, `/private/tmp/claude-501/-Users-iamdaone-Documents-RecursiveAgentsHackathonProject/068a120f-637f-4002-856f-3eb6683d41a7/scratchpad/headroom/results_rep.jsonl`.

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

These run directories are not results: `runs/alebench-live-2` (no paid usage.json: offline, scripted or a service log); `runs/alebench-offline` (no paid usage.json: offline, scripted or a service log); `runs/dashboard` (no paid usage.json: offline, scripted or a service log); `runs/datahall-20261017-143109` (no paid usage.json: offline, scripted or a service log); `runs/datahall-live-4` (no paid usage.json: offline, scripted or a service log); `runs/datahall-live-6` (no paid usage.json: offline, scripted or a service log); `runs/inference-opt-1` (no paid usage.json: offline, scripted or a service log); `runs/inference-opt-1-attempt0` (no paid usage.json: offline, scripted or a service log); `runs/kernelbench-mps-1` (not a named live run); `runs/localhost` (no paid usage.json: offline, scripted or a service log); `runs/qaplib-evolve-2` (no paid usage.json: offline, scripted or a service log); `runs/qaplib-evolve-4` (no paid usage.json: offline, scripted or a service log); `runs/qaplib-evolve-5` (no paid usage.json: offline, scripted or a service log).

Generated 2026-10-17 16:53:06 PDT. Regenerate with `.venv/bin/python scripts/collect_results.py` (rewrites reports/RESULTS.md and reports/results.json).
