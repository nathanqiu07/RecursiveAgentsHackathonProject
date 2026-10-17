# KernelBench mini on Apple GPU (MPS), not CUDA; Crusoe GPU quota was zero

**Device:** Apple M3 Pro GPU via PyTorch MPS (torch 2.14.1, `.venv-kb`). Baseline: PyTorch eager on MPS = 1.0x. Same 12 Level-1 problems and frozen split. Team: Nemotron Super (it answered the probe) writes PyTorch ops or `torch.mps.compile_shader` Metal kernels. Research on (You.com).

**Noise band:** we timed the reference against an identical copy of itself: 100 interleaved trials, 12 problems x 3 repeats. The ratio ranged 0.90 to 1.09 (SD 0.036), with 30 of 36 inside 0.95 to 1.05. The machine was busy with other runs. A per-problem speedup under 1.09x, or a 5-problem mean under about 1.03x, is noise.

**Validation:** 19 of 19 hack samples were rejected: 16 static, 2 reference_copy, 1 input_mutated, and skip-work cases scored incorrect. Two hand-written legit rewrites were correct (fused mse_loss 0.65x, Metal ReLU 0.94 to 0.99x). Neither was given to the team.

Held-out set: argmax, conv2d, cumsum, matmul_square, relu. All five are single-op references, so no fusion gain is available.

| Run | Arm | Correct | fast_1 | Mean speedup (correct) | Mean score |
|---|---|---|---|---|---|
| mps-1, 2 attempts | v0 cold | 20% | 0% | 0.46x | 0.091 |
| mps-1 | v0 + final brain | 40% | 0% | 0.71x | 0.284 |
| mps-2, 4 attempts | v0 cold | 20% | 0% | 0.97x | 0.195 |
| mps-2 | v0 + final brain | 60% | 0% | 0.73x | 0.439 |

**Gate:** no mutation reached the Gate, so the final team is v0 and the arms differ only by the shared brain. In mps-1 the Super Reviewer made 3 proposals: prompt_edit, temperature 0.1 (claimed +0.10 dev), and attempts 3 (claimed +0.20 dev at +25% cost). All three were dropped as invalid because the claim fields sat outside `claim`, so no measured value exists for them. In mps-2 the Nano Reviewer made 0 proposals in 2 generations.

**Cost:** mps-1 $0.104 (50 calls), mps-2 $0.191 (98 calls), smoke $0.004. Total $0.30.

**Verdict:** no held-out mean speedup is above 1.0, let alone beyond the noise band. The brain raised correctness (20% to 60%), not speed, so the team does not beat PyTorch eager on this layer.

Files: `noise_band.json`, `validation.jsonl`, `runs/kernelbench-mps-{1,2}/`.
