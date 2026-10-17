| Config | Accuracy | Pass rate | Cost/task (USD) | Cost/correct (USD) | p50 call latency (s) | p50 task wall (s) | Calls/task | Tokens/task |
|---|---|---|---|---|---|---|---|---|
| v0 (cold brain) | 0.979 (47/48) | 100% | 0.012831 | 0.0010920 | 9.67 | 163.13 | 12.0 | 9803 |
| v1 + final brain (final) | 1.000 (48/48) | 100% | 0.009033 | 0.0007528 | 19.01 | 77.18 | 4.0 | 6243 |

Held-out: accuracy 0.979 -> 1.000 at 70.4% of the cost per task; p50 call latency 9.672 s -> 19.013 s.
