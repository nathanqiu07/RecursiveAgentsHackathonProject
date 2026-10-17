# Runs by layer

Live Crusoe runs only, built by `.venv/bin/python scripts/build_datacenter_view.py` at 2026-10-17 16:05:39 PDT. Offline runs and runs with a $0 usage.json are left out. Status: done = final usage line written; running = process alive or log written in the last 3 min; crashed = Traceback without usage; stopped = none of these.

## Hall layout

| Run | Benchmark | Status | Start | End | Solver | Reviewer | Research | Gens | Scored | Accepted | Held-out best | Baseline | Delta | Cost (USD) | Calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| datahall-live-1 | DoneTwin data hall | done | 2026-10-17 14:30 | 2026-10-17 14:53 | Nemotron-3-Nano-30B-A3B | Nemotron-3-Super-120B-A12B | off | 3 | 7 | 0 | 0.710 | 0.710 | +0.000 | 0.1730 | 475 |
| datahall-live-2 | DoneTwin data hall | done | 2026-10-17 15:23 | 2026-10-17 15:32 | Nemotron-3-Nano-30B-A3B | Nemotron-3-Super-120B-A12B | off | 1 | 3 | 0 | 0.688 | 0.688 | -0.000 | 0.0433 | 128 |
| datahall-live-2d | DoneTwin data hall | done | 2026-10-17 15:30 | 2026-10-17 15:49 | Nemotron-3-Nano-30B-A3B | Nemotron-3-Nano-30B-A3B | off | 1 | 3 | 0 | 0.688 | 0.688 | -0.000 | 0.0794 | 109 |
| datahall-live-3 | DoneTwin data hall | done | 2026-10-17 15:54 | 2026-10-17 16:04 | Nemotron-3-Nano-30B-A3B | Nemotron-3-Nano-30B-A3B | on | 1 | 3 | 0 | 0.688 | 0.688 | -0.000 | 0.0385 | 129 |

## Equipment placement

| Run | Benchmark | Status | Start | End | Solver | Reviewer | Research | Gens | Scored | Accepted | Held-out best | Baseline | Delta | Cost (USD) | Calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qaplib-evolve-1 | QAPLIB | crashed | 2026-10-17 14:56 | 2026-10-17 15:04 | Nemotron-3-Super-120B-A12B | Nemotron-3-Super-120B-A12B | off | 1 | 1 | 0 | n/a | n/a | n/a | n/a | n/a |
| qaplib-evolve-2 | QAPLIB | crashed | 2026-10-17 15:16 | 2026-10-17 15:21 | Nemotron-3-Super-120B-A12B | Nemotron-3-Super-120B-A12B | off | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a |
| qaplib-evolve-3 | QAPLIB | done | 2026-10-17 15:25 | 2026-10-17 15:31 | Nemotron-3-Nano-30B-A3B | Nemotron-3-Nano-30B-A3B | off | 0 | 0 | 0 | 0.930 | 0.995 | -0.065 | 0.0111 | 54 |
| qaplib-evolve-4 | QAPLIB | stopped | 2026-10-17 15:31 | 2026-10-17 15:38 | Nemotron-3-Super-120B-A12B | Nemotron-3-Super-120B-A12B | off | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a |
| qaplib-evolve-5 | QAPLIB | running | 2026-10-17 15:38 | n/a | Nemotron-3-Super-120B-A12B | Nemotron-3-Super-120B-A12B | off | 2 | 3 | 1 | n/a | n/a | n/a | n/a | n/a |
| qaplib-dc-1 | GPU rack placement | done | 2026-10-17 15:41 | 2026-10-17 15:51 | Nemotron-3-Nano-30B-A3B | Nemotron-3-Nano-30B-A3B | off | 2 | 5 | 1 | 0.639 | 0.967 | -0.327 | 0.0168 | 96 |
| qaplib-research-1 | QAPLIB | done | 2026-10-17 15:51 | 2026-10-17 16:00 | Nemotron-3-Nano-30B-A3B | Nemotron-3-Nano-30B-A3B | on | 2 | 4 | 2 | 0.966 | 0.995 | -0.029 | 0.0163 | 89 |
| qaplib-dc-research-1 | GPU rack placement | done | 2026-10-17 15:51 | 2026-10-17 16:00 | Nemotron-3-Nano-30B-A3B | Nemotron-3-Nano-30B-A3B | on | 2 | 4 | 3 | 0.802 | 0.967 | -0.165 | 0.0208 | 89 |
| qaplib-dc-research-2 | GPU rack placement | running | 2026-10-17 16:04 | n/a | Nemotron-3-Nano-30B-A3B | Nemotron-3-Nano-30B-A3B | on | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a |
| qaplib-research-2 | QAPLIB | running | 2026-10-17 16:04 | n/a | Nemotron-3-Nano-30B-A3B | Nemotron-3-Nano-30B-A3B | on | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a |

## GPU kernels

No live run.

## Inference cost

| Run | Benchmark | Status | Start | End | Solver | Reviewer | Research | Gens | Scored | Accepted | Held-out best | Baseline | Delta | Cost (USD) | Calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| inference-opt-1-attempt0 | Crusoe ops Q&A | stopped | 2026-10-17 15:31 | 2026-10-17 15:48 | super (per question type) | Nemotron-3-Super-120B-A12B | off | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a |
| inference-opt-1 | Crusoe ops Q&A | stopped | 2026-10-17 15:48 | 2026-10-17 15:59 | super (per question type) | Nemotron-3-Super-120B-A12B | off | 1 | 1 | 1 | n/a | n/a | n/a | n/a | n/a |
| inference-opt-2 | Crusoe ops Q&A | running | 2026-10-17 15:59 | n/a | super (per question type) | Nemotron-3-Super-120B-A12B | on | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a |
