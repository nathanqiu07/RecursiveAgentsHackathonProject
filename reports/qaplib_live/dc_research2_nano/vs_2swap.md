| Config | Mean score | Mean gap to BKS (%) | At/above BKS | Pass (gap <= 1%) | Calls/task | Tokens/task | Cost/task (USD) |
|---|---|---|---|---|---|---|---|
| v0 + empty brain (cold) | 0.5973 | 92.63 | 0/6 | 0% | 2.00 | 3104 | 0.000342 |
| v0 + final brain | 0.6454 | 71.01 | 0/6 | 0% | 2.00 | 5760 | 0.000451 |
| v3 + final brain | 0.8452 | 18.88 | 0/6 | 0% | 2.00 | 8826 | 0.001069 |
| 2-swap baseline (no LLM, 10 s restarts) | 0.9666 | 3.52 | 1/6 | 17% | 0.00 | 0 | 0.000000 |

- v0 + empty brain (cold): -36.93 points vs 2-swap
- v0 + final brain: -32.12 points vs 2-swap
- v3 + final brain: -12.14 points vs 2-swap
