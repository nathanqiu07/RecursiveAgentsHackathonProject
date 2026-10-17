| Config | Mean score | Pass rate | Mean iters | Calls/task | Tokens/task | Cost/task (USD) | Cost per verified point (USD) | Tokens | Cost (USD) |
|---|---|---|---|---|---|---|---|---|---|
| v0 + empty brain (cold) | 0.688 | 100% | 2.00 | 4.00 | 16243 | 0.001443 | 0.002098 | 97457 | 0.0087 |
| v0 + final brain | 0.688 | 100% | 2.00 | 4.00 | 16573 | 0.001412 | 0.002053 | 99439 | 0.0085 |
| v1 + final brain | 0.728 | 100% | 2.00 | 4.00 | 15842 | 0.001245 | 0.001712 | 95055 | 0.0075 |
| greedy + polish (no LLM) | 0.728 | 100% | 0.00 | - | - | - | - | 0 | 0.0000 |
| greedy baseline (no LLM) | 0.688 | 100% | 0.00 | - | - | - | - | 0 | 0.0000 |

- v0 + empty brain (cold): +0.00 points vs greedy, -3.96 vs greedy + polish
- v0 + final brain: +0.00 points vs greedy, -3.96 vs greedy + polish
- v1 + final brain: +3.96 points vs greedy, +0.00 vs greedy + polish
- greedy + polish (no LLM): +3.96 points vs greedy
