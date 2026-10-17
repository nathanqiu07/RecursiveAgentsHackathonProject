| Config | Mean score | Mean gap to BKS (%) | At/above BKS | Pass (gap <= 1%) | Calls/task | Tokens/task | Cost/task (USD) |
|---|---|---|---|---|---|---|---|
| v0 + empty brain (cold) | 0.9593 | 4.33 | 0/6 | 33% | 2.00 | 2840 | 0.000351 |
| v0 + final brain | 0.8738 | 15.10 | 0/6 | 0% | 2.00 | 3358 | 0.000347 |
| v2 + final brain | 0.9662 | 3.63 | 0/6 | 33% | 2.00 | 5005 | 0.000555 |
| 2-swap baseline (no LLM, 10 s restarts) | 0.9954 | 0.46 | 2/6 | 83% | 0.00 | 0 | 0.000000 |

- v0 + empty brain (cold): -3.61 points vs 2-swap
- v0 + final brain: -12.16 points vs 2-swap
- v2 + final brain: -2.92 points vs 2-swap
