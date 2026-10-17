| Generation | Mutation type | Claim | Measured | Verdict |
|---|---|---|---|---|
| 0 | config_set | gain +0.050, cost -10.0% (Polish improves feasibility and score via cheap verifier-guided moves, reducing costly Layout iterations and LLM token usage while fixing top failure modes.) | gain +0.033, cost -16.2% | reproduced |
| 0 | config_set | gain +0.020, cost -15.0% (Fewer polish checks reduce verifier tool usage and cost, with minimal risk to solution quality given the hill-climbing nature of layout improvements.) | gain +0.001, cost -14.1% | not reproduced |
| 0 | config_set | gain +0.030, cost -5.0% (Deterministic outputs reduce wasted exploration in layout generation, lowering token use and improving consistency in meeting hard rules.) | gain +0.000, cost -2.5% | not reproduced |
