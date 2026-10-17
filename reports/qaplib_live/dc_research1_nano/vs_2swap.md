| Config | Mean score | Mean gap to BKS (%) | At/above BKS | Pass (gap <= 1%) | Calls/task | Tokens/task | Cost/task (USD) |
|---|---|---|---|---|---|---|---|
| v0 + empty brain (cold) | 0.4773 | 117.05 (1 invalid) | 1/6 | 17% | 1.83 | 2534 | 0.000301 |
| v0 + final brain | 0.7132 | 50.77 | 0/6 | 0% | 2.00 | 4588 | 0.000381 |
| v2 + final brain | 0.8020 | 28.57 | 0/6 | 0% | 2.00 | 7707 | 0.001025 |
| 2-swap baseline (no LLM, 10 s restarts) | 0.9666 | 3.52 | 1/6 | 17% | 0.00 | 0 | 0.000000 |

- v0 + empty brain (cold): -48.93 points vs 2-swap
- v0 + final brain: -25.34 points vs 2-swap
- v2 + final brain: -16.46 points vs 2-swap
