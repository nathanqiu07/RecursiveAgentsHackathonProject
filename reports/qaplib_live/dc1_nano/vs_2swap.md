| Config | Mean score | Mean gap to BKS (%) | At/above BKS | Pass (gap <= 1%) | Calls/task | Tokens/task | Cost/task (USD) |
|---|---|---|---|---|---|---|---|
| v0 + empty brain (cold) | 0.5042 | 146.21 | 0/6 | 0% | 2.00 | 2650 | 0.000299 |
| v0 + final brain | 0.5865 | 87.54 | 0/6 | 0% | 2.00 | 4598 | 0.000374 |
| v1 + final brain | 0.6394 | 86.48 | 0/6 | 0% | 2.00 | 4543 | 0.000373 |
| 2-swap baseline (no LLM, 10 s restarts) | 0.9666 | 3.52 | 1/6 | 17% | 0.00 | 0 | 0.000000 |

- v0 + empty brain (cold): -46.24 points vs 2-swap
- v0 + final brain: -38.01 points vs 2-swap
- v1 + final brain: -32.72 points vs 2-swap
