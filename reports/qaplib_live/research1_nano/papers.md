| Generation | Mutation type | Claim | Measured | Verdict |
|---|---|---|---|---|
| 0 | config_set | gain +0.070, cost -0.1% (Using the larger 120B model (nvidia/NVIDIA-Nemotron-3-Super-120B-A12B) should raise mean dev score by 7% while cost per task drops 10% due to fewer attempts nee) | gain -0.082, cost -17.9% | not reproduced |
| 0 | config_set | gain +0.040, cost -0.1% (Activating thinking (reasoning) for the solver role is expected to boost dev score by 4% and reduce cost per task by 5% by producing better heuristics, similar ) | gain +0.073, cost -9.8% | reproduced |
| 1 | config_set | gain +0.020, cost -0.1% (Increasing max_tokens to 8000 allows longer chain-of-thought without truncation, improving solution quality; higher token ceiling raises cost slightly but still) | gain -0.011, cost +34.9% | not reproduced |
| 1 | config_set | gain +0.010, cost -0.1% (Lowering temperature to 0.2 sharpens output consistency, reducing gaps; the modest cost reduction comes from fewer tokens used per call.) | gain +0.036, cost +14.0% | accepted, effect smaller than claimed |
