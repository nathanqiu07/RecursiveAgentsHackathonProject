| Config | Mean score | Mean gap to BKS (%) | At/above BKS | Pass (gap <= 1%) | Calls/task | Tokens/task | Cost/task (USD) |
|---|---|---|---|---|---|---|---|
| v0 + empty brain (cold) | 0.6741 | 28.11 (1 invalid) | 0/6 | 17% | 2.00 | 2892 | 0.000327 |
| v0 + final brain | 0.6825 | 26.72 (1 invalid) | 0/6 | 0% | 2.00 | 5387 | 0.000428 |
| v1 + final brain | 0.7537 | 43.58 | 0/6 | 0% | 2.00 | 5638 | 0.000468 |
| 2-swap baseline (no LLM, 10 s restarts) | 0.9954 | 0.46 | 2/6 | 83% | 0.00 | 0 | 0.000000 |

- v0 + empty brain (cold): -32.13 points vs 2-swap
- v0 + final brain: -31.29 points vs 2-swap
- v1 + final brain: -24.17 points vs 2-swap
