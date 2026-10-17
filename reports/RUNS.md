# Runs by layer

Live Crusoe runs only, built by `.venv/bin/python scripts/build_datacenter_view.py` at 2026-10-17 16:30:45 PDT. Offline runs and runs with a $0 usage.json are left out. Status: done = final usage line written; running = process alive or log written in the last 3 min; crashed = Traceback without usage; stopped = none of these.

## Hall layout

| Run | Benchmark | Status | Start | End | Solver | Reviewer | Research | Gens | Scored | Accepted | v0 cold | Final | Δ vs v0 | Cost (USD) | Calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| datahall-live-1 | DoneTwin data hall | done | 2026-10-17 14:30 | 2026-10-17 14:53 | Nano | Super | off | 3 | 7 | 0 | 0.710 | 0.710 | +0.000 | 0.1730 | 475 |
| datahall-live-2 | DoneTwin data hall | done | 2026-10-17 15:23 | 2026-10-17 15:32 | Nano | Super | off | 1 | 3 | 0 | 0.688 | 0.688 | +0.000 | 0.0433 | 128 |
| datahall-live-2d | DoneTwin data hall | done | 2026-10-17 15:30 | 2026-10-17 15:49 | Nano | Nano | off | 1 | 3 | 0 | 0.688 | 0.688 | +0.000 | 0.0794 | 109 |
| datahall-live-3 | DoneTwin data hall | done | 2026-10-17 15:54 | 2026-10-17 16:04 | Nano | Nano | on | 1 | 3 | 0 | 0.688 | 0.688 | +0.000 | 0.0385 | 129 |
| datahall-live-4 | DoneTwin data hall | running | 2026-10-17 16:25 | n/a | Nano | Nano | on | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a |

## Equipment placement

| Run | Benchmark | Status | Start | End | Solver | Reviewer | Research | Gens | Scored | Accepted | v0 cold | Final | Δ vs v0 | Cost (USD) | Calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qaplib-evolve-1 | QAPLIB | crashed | 2026-10-17 14:56 | 2026-10-17 15:04 | Super | Super | off | 1 | 1 | 0 | n/a | n/a | n/a | n/a | n/a |
| qaplib-evolve-2 | QAPLIB | crashed | 2026-10-17 15:16 | 2026-10-17 15:21 | Super | Super | off | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a |
| qaplib-evolve-3 | QAPLIB | done | 2026-10-17 15:25 | 2026-10-17 15:31 | Nano | Nano | off | 0 | 0 | 0 | 0.853 | 0.930 | +0.077 | 0.0111 | 54 |
| qaplib-evolve-4 | QAPLIB | stopped | 2026-10-17 15:31 | 2026-10-17 15:38 | Super | Super | off | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a |
| qaplib-evolve-5 | QAPLIB | stopped | 2026-10-17 15:38 | 2026-10-17 16:04 | Super | Super | off | 2 | 3 | 1 | n/a | n/a | n/a | n/a | n/a |
| qaplib-dc-1 | GPU rack placement | done | 2026-10-17 15:41 | 2026-10-17 15:51 | Nano | Nano | off | 2 | 5 | 1 | 0.504 | 0.639 | +0.135 | 0.0168 | 96 |
| qaplib-research-1 | QAPLIB | done | 2026-10-17 15:51 | 2026-10-17 16:00 | Nano | Nano | on | 2 | 4 | 2 | 0.959 | 0.966 | +0.007 | 0.0163 | 89 |
| qaplib-dc-research-1 | GPU rack placement | done | 2026-10-17 15:51 | 2026-10-17 16:00 | Nano | Nano | on | 2 | 4 | 3 | 0.477 | 0.802 | +0.325 | 0.0208 | 89 |
| qaplib-dc-research-2 | GPU rack placement | done | 2026-10-17 16:04 | 2026-10-17 16:25 | Nano | Nano | on | 3 | 6 | 4 | 0.597 | 0.845 | +0.248 | 0.0477 | 112 |
| qaplib-research-2 | QAPLIB | done | 2026-10-17 16:04 | 2026-10-17 16:19 | Nano | Nano | on | 2 | 6 | 1 | 0.674 | 0.754 | +0.080 | 0.0208 | 102 |
| qaplib-dc-research-3 | GPU rack placement | running | 2026-10-17 16:26 | n/a | Nano | Nano | on | 1 | 3 | 1 | n/a | n/a | n/a | n/a | n/a |

## GPU kernels

No live run.

## Inference cost

| Run | Benchmark | Status | Start | End | Solver | Reviewer | Research | Gens | Scored | Accepted | v0 cold | Final | Δ vs v0 | Cost (USD) | Calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| inference-opt-1-attempt0 | Crusoe ops Q&A | stopped | 2026-10-17 15:31 | 2026-10-17 15:48 | Super | Super | off | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a |
| inference-opt-1 | Crusoe ops Q&A | stopped | 2026-10-17 15:48 | 2026-10-17 15:59 | Super | Super | off | 1 | 1 | 1 | n/a | n/a | n/a | n/a | n/a |
| inference-opt-2 | Crusoe ops Q&A | done | 2026-10-17 15:59 | 2026-10-17 16:14 | Super | Super | on | 1 | 1 | 1 | 0.001092 | 0.000753 | -0.000339 | 0.2650 | 208 |
