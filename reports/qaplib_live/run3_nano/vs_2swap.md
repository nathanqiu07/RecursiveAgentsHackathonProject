| Config | Mean score | Mean gap to BKS (%) | At/above BKS | Pass (gap <= 1%) | Calls/task | Tokens/task | Cost/task (USD) |
|---|---|---|---|---|---|---|---|
| v0 + empty brain (cold) | 0.8529 | 17.93 | 0/6 | 0% | 2.00 | 2660 | 0.000321 |
| v0 + final brain | 0.9300 | 8.38 | 0/6 | 33% | 2.00 | 4148 | 0.000357 |
| v0 + final brain | 0.9300 | 8.38 | 0/6 | 33% | 2.00 | 4148 | 0.000357 |
| 2-swap baseline (no LLM, 10 s restarts) | 0.9954 | 0.46 | 2/6 | 83% | 0.00 | 0 | 0.000000 |

- v0 + empty brain (cold): -14.25 points vs 2-swap
- v0 + final brain: -6.54 points vs 2-swap
- v0 + final brain: -6.54 points vs 2-swap
