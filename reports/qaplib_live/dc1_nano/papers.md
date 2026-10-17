| Generation | Mutation type | Claim | Measured | Verdict |
|---|---|---|---|---|
| 0 | playbook_add | gain +0.020, cost -0.1% (Prevents runtime errors and reduces wasted compute on invalid solutions, improving mean score and lowering cost per task.) | gain -0.242, cost +5.7% | not reproduced |
| 0 | config_set | gain +0.010, cost -0.1% (Reduces token usage while keeping solution length sufficient, cutting cost per task and modestly improving score.) | gain -0.217, cost +7.8% | not reproduced |
| 0 | config_set | gain +0.015, cost -0.1% (Lower temperature yields more deterministic outputs, improving verification pass rate and reducing retries.) | gain -0.396, cost +9.1% | not reproduced |
| 1 | config_set | gain +0.003, cost -0.1% (Reduces token consumption per call, lowering cost per task by roughly 12% without dropping the mean score, as shorter responses are sufficient for verification.) | gain -0.241, cost -5.3% | not reproduced |
| 1 | config_set | gain +0.004, cost -0.1% (Lower temperature yields more deterministic outputs, improving solution correctness and slightly raising the mean score while decreasing cost per task by about ) | gain +0.093, cost -9.2% | reproduced |
