| Generation | Mutation type | Claim | Measured | Verdict |
|---|---|---|---|---|
| 0 | config_set | gain +0.020, cost -0.1% (Increasing max_tokens to 8000 allows longer reasoning without truncation; [3] indicates truncation causes failures, and larger token budget improves quality whi) | gain +0.274, cost -7.2% | reproduced |
| 0 | config_set | gain +0.010, cost -0.1% (Lower temperature reduces stochasticity, leading to more deterministic outputs; small cost reduction from fewer retries, with minor quality gain.) | gain +0.241, cost -6.5% | reproduced |
| 1 | config_set | gain +0.020, cost -0.1% (Higher token budget improves solution correctness; slight cost reduction from fewer retries) | gain -0.118, cost +0.2% | not reproduced |
| 1 | config_set | gain +0.030, cost -0.1% (Thinking improves heuristic design accuracy; reduces need for multiple attempts, lowering cost) | gain +0.146, cost +93.2% | accepted, effect smaller than claimed |
