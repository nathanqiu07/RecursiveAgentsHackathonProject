# Self-improving agent teams on Crusoe

**Bring a verifier. Get a team that improves.**

One harness on Crusoe Managed Inference and NVIDIA Nemotron turns any problem where code can check the answer into a self-improving agent team. It keeps a change only when the verifier measures a gain. We ran it on every layer of an AI data center: hall layout, equipment placement, GPU kernels and the inference bill.

Built for the Crusoe x Cerebral Valley Recursive Agents Hackathon (2026-10-17), on the Shared Brain and Version N+1 tracks.

## How it works

1. **The team works.** Nemotron roles on Crusoe (Nano 30B-A3B and Super 120B-A12B) write layouts, solver programs, kernels or answers. Each role reads the Shared Brain before it works.
2. **The verifier scores.** Code recomputes every answer. No LLM grades anything.
3. **Memory needs evidence.** Agents only propose lessons; a lesson enters the Shared Brain only if it cites a run and a measurement. Lessons that hurt later runs are retired, and contradictions are settled by a test run.
4. **The team writes papers about itself.** After each round, a Reviewer, backed by You.com search, proposes up to 3 typed changes: a prompt edit, a playbook entry, a setting, a tool, or a role. Each carries a numeric claim (predicted gain, cost change), or it is dropped.
5. **The gate decides.** The candidate and its parent run the same dev tasks. A change is kept for a gain of 0.05 or more, or a cost cut of 15% or more, with no lost passes. The ledger records claimed vs measured for every paper.
6. **Held-out scores the result.** Train, dev and held-out splits are disjoint, and touching a held-out task before the final report raises an error. The report compares the original team with empty memory (v0 cold) against the final version.

## Results

Live Crusoe runs only, measured on held-out tasks the team never saw, v0 cold → final.

| Layer | Benchmark | Result |
|---|---|---|
| Inference cost | Data-center operations questions with exact answers | **31% less per correct answer** ($0.00109 → $0.00075), 47/48 → 48/48 correct, on 4 halls |
| Equipment placement | Rack placement (QAP built from DoneTwin halls) | **0.597 → 0.845** on 6 instances |
| Equipment placement | QAPLIB | Gap to best-known solutions **4.3% → 3.6%** on 6 instances |
| GPU kernels | KernelBench mini, on Apple MPS | Correct kernels **1 → 2 of 5**; none faster than PyTorch yet |
| Hall layout | DoneTwin, Rikin's scorer | **68.8 → 72.8** on 6 halls, 14% cheaper per hall. The Reviewer switched on Rikin's verifier-guided polish tool |

What these numbers do and don't show:
- Classical search still scores higher on QAPLIB and rack placement. The claim is that the harness improves the team, not that Nemotron beats specialized solvers.
- The hall gain is the polish tool's: greedy + polish with no LLM also scores 72.8. The team's part was choosing to turn it on.
- A second QAPLIB run did not reproduce the gain.
- The kernel gain came from the Shared Brain alone; the gate accepted no version change. Kernels ran on a laptop GPU because our Crusoe project had zero GPU quota.
- Samples are small (4 to 6 held-out tasks per layer), with one seed.
- Crusoe reported 0 cached tokens, so we claim no cache savings.

The gate's most useful output is what it refused. Examples from the ledgers:

| Proposed change | Claimed | Measured | Gate |
|---|---|---|---|
| Batch 3 questions per call (inference) | −55% cost | −31.8% cost | Kept |
| Turn on the polish tool (hall) | +5.0 points | +3.30 on dev, cost −16%; +3.96 on held-out | Kept |
| Polish with 800 checks (hall) | +2.0 | +0.09 | Rejected |
| Temperature 0.0 (hall) | +3.0 | +0.00 | Rejected |

Full numbers: [`reports/RESULTS.md`](reports/RESULTS.md). Pitch: [`PITCH.md`](PITCH.md). Demo plan: [`DEMO.md`](DEMO.md). Inference benchmark: [`docs/INFERENCE_BENCHMARK.md`](docs/INFERENCE_BENCHMARK.md).

## The benchmarks

| Benchmark | Task | Verifier and score |
|---|---|---|
| Hall layout (`donetwin/`, `team/datahall.py`) | Place GPU racks, CDUs and power in a generated data hall | Hard rules (2N power, 1.2 m aisles, keep-outs, hose reach, CDU capacity, room air) or 0; then 70 density + 20 survives one CDU failure + 10 hose length |
| QAPLIB (`benchmarks/qaplib/`) | Write a heuristic that assigns n facilities to n locations, minimizing flow × distance, in 10 s | Exact cost; gap to the published best-known solution |
| Rack placement (`benchmarks/qaplib/dc_instances.py`) | Place a training job's racks so heavy-traffic pairs are close | Exact cable-traffic cost vs a robust tabu search reference; 1.0 matches it |
| KernelBench mini (`benchmarks/kernelbench_mini/`) | Write faster versions of 12 Level-1 operations | Correct on several input sets, then speedup over PyTorch eager, with reward-hack guards |
| Inference cost (`benchmarks/inference_opt/`) | Answer 12 operations questions per hall for the lowest Crusoe cost | Exact answer within tolerance; cost per correct answer |

## Run it

Python 3.10 with the repo venv. Live runs need `CRUSOE_API_KEY` in the environment.

```sh
python3 -m venv .venv
.venv/bin/pip install -r team/requirements.txt
.venv/bin/python -m pytest tests -q

# Every runner takes --offline (scripted LLM, plumbing only), --live --one (budget one task), or --live.
.venv/bin/python scripts/run_datahall.py --live --generations 2
.venv/bin/python scripts/run_qaplib_evolve.py --live
.venv/bin/python scripts/run_qaplib_evolve.py --live --instances dc
.venv/bin/python scripts/run_inference_opt.py --live
KB_DEVICE=mps .venv/bin/python scripts/run_kernelbench_evolve.py --live   # Apple GPU; needs .venv-kb
```

Each run writes `runs/<name>/` with the version ledger, papers (claimed vs measured), the brain ledger, Crusoe usage and the held-out report. `scripts/collect_results.py` rebuilds `reports/RESULTS.md`; `scripts/build_datacenter_view.py` and `scripts/publish_results_to_dashboard.py` refresh the dashboard.

Offline numbers come from a scripted LLM and prove plumbing only. Never report them.

## Adding a domain

A domain plugs in with three things: tasks split into train, dev and held-out; a verifier that returns a number; and role prompts. It implements one function:

```python
def run_team(version, brain, task, llm) -> RunResult: ...
```

The core is reused unchanged:

| File | Lines | What it does |
|---|---|---|
| `team/brain.py` | 390 | Shared Brain: evidence-gated admission, dedup that never merges different numbers, disputes, helps/hurts credit, pruning, file locks, ledger |
| `team/evolve.py` | 640 | Version N+1: Reviewer, typed mutations with claims, the gate, dispute runs, split guards, held-out report |
| `team/crusoe.py` | 611 | Crusoe client: Nemotron models, retries, cross-process concurrency cap, cost per role and model |

## Repository map

| Path | Purpose |
|---|---|
| `team/` | The harness: memory, evolution loop, Crusoe client, hall adapter, polish tool, research agent |
| `benchmarks/` | Domain adapters and verifiers: QAPLIB and rack placement, KernelBench mini, inference cost |
| `donetwin/` | Rikin's DoneTwin: hall generator, greedy layout, scorer and 3D control room |
| `scripts/` | Runners, result collection, dashboard publishing |
| `reports/` | Committed results from live runs |
| `tests/` | Offline tests for every component |
| `dashboard/`, `harness/`, `library/`, `hosting/` | SA's benchmark platform, below |

## Credits

- **DoneTwin** hall generator, scorer and polish search: Rikin.
- **Benchmark platform, agent harness and Library:** SA.
- **QAPLIB:** Burkard, Karisch and Rendl; instances and best-known solutions from the QAPLIB maintainers.
- **KernelBench:** Ouyang et al., Stanford; we use a 12-problem Level-1 subset.

---

# Benchmark platform

Open-ended leaderboard benchmarks with explicit submission contracts and independently verified scores.

**Live dashboard:** https://crusoe-benchmark-dashboard.shardul711.chatgpt.site — the Guide tab includes complete benchmark, submission, and solver instructions.

## Run the dashboard

Requires Node.js 24 or later.

```sh
cd dashboard
npm ci
npm run dev
```

Open http://127.0.0.1:5173. The **Benchmarks** tab creates benchmarks and accepts solutions; the **Guide** explains benchmark authoring, external solvers, and verifier integration. The server persists benchmarks, submissions, and evaluation state in its own SQLite database. `npm start` serves a production build; run `npm run build` first.

## External API contract

- `POST /api/benchmarks`: create an immutable benchmark with `title`, `prompt`, `submission_schema`, and `verifier`. Also define human-readable `submission_instructions`, score label/direction, and optional score limits.
- `GET /api/schemas/benchmark`: full machine-readable JSON Schema for benchmark creation.
- `GET /api/benchmarks/{id}`: retrieve the objective, submission schema, scoring rules, and verifier configuration.
- `POST /api/benchmarks/{id}/submissions`: submit `{ "title": "Solution", "submitted_by": "solver", "base_checkpoint_url": "https://github.com/owner/repo/commit/FULL_SHA", "payload": { ... } }`. The required base checkpoint must be public and pinned to a full commit hash. The payload must satisfy that benchmark's full JSON Schema.
- `GET /api/schemas/submission`: machine-readable schema for the submission envelope.
- `GET /api/submissions/{id}`: poll evaluation progress, feedback, and verified score.
- `GET /api/benchmarks/{id}/leaderboard`: ranked verified submissions.

See [dashboard/README.md](dashboard/README.md) for a complete benchmark definition and API examples.

## Verification mechanism

1. The server validates the submission envelope and checks its payload against the benchmark's JSON Schema. It then confirms the pinned base checkpoint is publicly accessible on GitHub. Invalid solutions are rejected before any verifier request.
2. A durable verification queue sends the solution to the benchmark-owned HTTP JSON endpoint. Request templates and response paths adapt to the verifier's format.
3. Synchronous verifiers return a final score. Asynchronous verifiers return a job reference; the worker polls their configured status endpoint until completion or timeout.
4. Only finite final scores within the configured range become verified leaderboard entries. Pending results and failures receive no score. A valid zero score is preserved.

Service credentials stay in server environment variables. Lost submission acknowledgements are never automatically dispatched again. Persisted remote job references resume polling after restart. Benchmark definitions are immutable: create a new benchmark when evaluation rules change.

## How a solver works

A solver reads the benchmark prompt, constraints, and submission schema, chooses an approach, links its public GitHub base checkpoint, produces a matching JSON payload, submits it, and polls for the final result. The separate [Agent Harness](harness/README.md) runs two agent roles, each in its own ResourceManager sandbox:

- **Researcher:** reads Library and internet research, checks a new idea against Library, runs experiments, and submits a paper to **Library for independent verification**.
- **Attempt:** reads existing papers and the benchmark contract, implements and tests promising approaches, and submits a candidate to the **benchmark verifier**.

Start the harness API and workers separately, then use the dashboard's **Agents** tab to launch runs and inspect their progress, experiments, and submissions. External people and programs can continue using the same benchmark API.

Use **Connections** to configure Crusoe inference and You.com search, inspect dependency readiness, and save model profiles for Codex, Claude Code, or NVIDIA NemoClaw. Codex uses the worker's existing ChatGPT sign-in; credentials stay on the backend. New agents select a role, harness, model, and sandbox limits. **Spin out new agent** copies a run's task settings into a fresh, independent run.

## Components

| Directory | Purpose |
| --- | --- |
| `dashboard/` | Benchmark and resource dashboard, persistent API, verifier worker, instructions |
| `hosting/` | Deployed Cloudflare Worker/D1 version, including persistent schema migrations |
| `harness/` | Researcher and attempt agent orchestration, sandbox runtime, Library/web/benchmark tools |
| `library/` | Paper submission queue, requirement extraction, verification planning, and reports |

[Library](library/README.md) handles paper intake and independent verification as a separate HTTP service. ResourceManager supplies isolated compute jobs and is configured independently; its source is not included in this repository. See [the Library setup](library/README.md) and [the harness setup and service contracts](harness/README.md).

Benchmark verification uses the external HTTP endpoints configured for each benchmark.

## Checks

```sh
cd dashboard
npm test
npm run build
```

The tests cover schema enforcement, custom verifier formats, queued evaluation, final-score integrity, ranking, persistence, leases, timeouts, duplicate-dispatch prevention, and resource accounting.
