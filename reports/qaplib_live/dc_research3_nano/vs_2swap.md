| Config | Mean score | Mean gap to BKS (%) | At/above BKS | Pass (gap <= 1%) | Calls/task | Tokens/task | Cost/task (USD) |
|---|---|---|---|---|---|---|---|
| v0 + empty brain (cold) | 0.4295 | 138.88 (1 invalid) | 0/6 | 0% | 2.00 | 2867 | 0.000309 |
| v0 + final brain | 0.6260 | 87.12 | 0/6 | 0% | 2.00 | 5782 | 0.000437 |
| v3 + final brain | 0.9213 | 8.95 | 1/6 | 17% | 2.00 | 9150 | 0.001123 |
| 2-swap baseline (no LLM, 10 s restarts) | 0.9666 | 3.52 | 1/6 | 17% | 0.00 | 0 | 0.000000 |

- v0 + empty brain (cold): -53.71 points vs 2-swap
- v0 + final brain: -34.06 points vs 2-swap
- v3 + final brain: -4.53 points vs 2-swap
