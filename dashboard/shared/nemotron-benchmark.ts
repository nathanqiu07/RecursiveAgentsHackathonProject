import type { BenchmarkInput } from './benchmarks';

export const nemotronModel = 'nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4';
export const nemotronRevision = 'bee7596271d1495f6992ae224aefde4410e816b8';
export const nemotronReferenceUrl = 'https://docs.nvidia.com/nemotron/0.2.0/nemotron/lightning35/evaluate.html';
export const nemotronMemoryReferenceUrl = 'https://docs.nvidia.com/nim/large-language-models/latest/supported-models.html';

// External published references are context, never our measured baselines or leaderboard entries.
export const nemotronBaselines = [
  { key: 'mmlu', name: 'MMLU', setting: 'Mini · 57 questions · 57 subjects', gate: 'At most one additional incorrect answer' },
  { key: 'ppl', name: 'PPL', setting: 'Mini · 4,096 WikiText-2 tokens', gate: 'At most 1% above baseline' },
  { key: 'gsm8k', name: 'GSM8K', setting: 'Mini · 16 problems · exact match', gate: 'At most one additional incorrect answer' },
] as const;

export const nemotronBenchmark: Omit<BenchmarkInput, 'verifier'> = {
  title: 'Optimize Nemotron Lightning · RTX A5000 · NVFP4',
  prompt: `Optimize inference throughput for NVIDIA Nemotron 3.5 Lightning 30B-A3B on one NVIDIA RTX A5000 24 GB, while preserving model quality on MMLU, WikiText-2 perplexity (PPL), and GSM8K. Choose your own kernels, runtime, scheduling, and memory optimizations.

MODEL AND HARDWARE
Use ${nemotronModel} at revision ${nemotronRevision}. Use exactly one RTX A5000 with 24 GB VRAM on the existing resource server. RTX A5000 has no native FP4 tensor cores: preserve the NVFP4 weights and use the W4A16 compute path. This measures NVFP4 storage and runtime optimization, not native Blackwell FP4 arithmetic. Preserve the released NVFP4 weight format and mixed-precision exceptions in its quantization configuration; NVFP4 here does not mean every activation is FP4. No retraining, alternate main model, extra GPUs, or CPU/disk weight offloading. Lossless speculative decoding is allowed only with its complete memory and runtime costs included on the same GPU.

THROUGHPUT SCORE
Target workload: C20 / 2K input / 1K output — 20 concurrent client requests, exactly 2,048 input tokens and 1,024 generated tokens per request, EOS stopping disabled. After one warmup request, measure 20 requests in a closed loop with at most 20 requests in flight. Score is aggregate output tokens divided by wall time from the first timed request until the final timed request completes, including prefill, queueing, and API overhead. Higher output tokens/second is better. Model loading and warmups are excluded. Prefix/output caching must be disabled. The engine may limit active sequences and queue requests to fit the 24 GB A5000; it may not lower the offered client concurrency or token budgets. The verifier uses fixed prompts, tokenizer, seed 42, GPU identity, and settings for reference and candidate.
The released checkpoint's stored tensors occupy about 21.56 GB before runtime allocations. One A5000 is a tight memory target; successful loading and C20 completion are requirements to be measured, not promised. NVIDIA's stock NIM W4A16 profile has a 32 GB floor; a custom vLLM/SGLang path must establish its own fit. Smaller concurrency runs are diagnostics, not interchangeable leaderboard scores. Candidate deployments are registered by the operator with their exact source repository and commit. The local verifier never executes submitted source code or accepts submitter-reported scores.
Memory reference: ${nemotronMemoryReferenceUrl}

FAST BASELINE QUALITY CHECK
Use one frozen mini suite prepared with seed 42. MMLU: 57 test questions, one per each of the 57 subjects, 0-shot multiple choice, greedy generated answer-letter extraction, at most 8 generated tokens, thinking disabled. GSM8K: 16 test problems, 0-shot chain-of-thought, greedy decoding, at most 512 generated tokens, final numeric exact match using the #### answer marker. Invalid answers and truncations count as incorrect. PPL: the first 4,096 tokens of WikiText-2 wikitext-2-raw-v1 test text, joined with two newlines and tokenized without chat formatting or added special tokens; use 2,048-token windows, stride 1,024, mask overlapping targets, and compute exp(total negative log likelihood / scored tokens). Teacher-forced prompt-token log probabilities are required; generated-token confidence is not PPL.
Quality requests run with up to four clients. The mini suite and short timing sample target 10–20 minutes of execution on a prepared VM, with actual duration reported. This is a runtime target, not a hard cutoff: valid slow jobs remain eligible. Queue wait, VM boot, and model loading are reported separately. The same three resource nodes are shared with Experiments, with one active VM per node. Each node has its own immutable reference, and candidates compare against the reference from that same physical GPU.
The verifier first measures and freezes the unmodified checkpoint's mini MMLU correct count, mini GSM8K correct count, and mini PPL, with the model/dataset revisions and file checksums. Each candidate may have at most one additional incorrect answer on MMLU and at most one on GSM8K, and its PPL must be no more than 1.01 times the measured reference. A failed quality check rejects the submission before the longer C20 throughput run and receives no leaderboard score. These small deterministic checks are regression screening; their coarse accuracy percentages are not full-suite MMLU or GSM8K results. Harnesses and verifiers run locally; only inference/job execution runs on resource-server VMs.

PUBLISHED REFERENCE SCORES
NVIDIA reports MMLU 78.63% and GSM8K 91.74% for the Lightning Base BF16 checkpoint; PPL is not reported. These are context from a different checkpoint and recipe, not measured RTX A5000 NVFP4 baselines or acceptance thresholds. All three A5000 baseline values remain pending until the independent verifier completes the mini suite. Do not substitute MMLU-Pro scores for MMLU.
Source: ${nemotronReferenceUrl}
Target checkpoint: https://huggingface.co/${nemotronModel}/tree/${nemotronRevision}

VERIFICATION OUTPUT
Return a numeric score only after hardware, precision, reproducibility, and all quality checks pass. Include measured reference and candidate mini MMLU (% and correct/57), WikiText-2 mini PPL, mini GSM8K (% and correct/16), end-to-end output throughput, runtime identity, GPU identity, and baseline-manifest identifier in feedback. Reference/candidate results must use the same manifest. No measured RTX A5000 baseline or throughput score is claimed by this prepared benchmark.`,
  submission_instructions: 'Submit a public GitHub repository and full immutable commit hash for the optimization. Include README setup instructions and pinned dependencies. Deploy the pinned source on the RTX A5000 and have the operator register its deployment alias with the verifier before submitting. The verifier evaluates only registered inference endpoints and does not run submitted code. The top-level base_checkpoint_url must point to the public GitHub base commit from which your implementation started. The payload commit identifies the final implementation. Include the target model, GPU, and precision exactly as specified. Do not submit claimed quality or throughput values as the official score; the independent verifier measures them.',
  submission_schema: {
    type: 'object',
    properties: {
      repository: { type: 'string', format: 'uri', pattern: '^https://github\\.com/[A-Za-z0-9-]+/[A-Za-z0-9_.-]+/?$', maxLength: 2000 },
      commit: { type: 'string', pattern: '^(?:[a-fA-F0-9]{40}|[a-fA-F0-9]{64})$' },
      deployment: { type: 'string', minLength: 1, maxLength: 100, pattern: '^[A-Za-z0-9_-]+$' },
      model: { const: nemotronModel },
      gpu: { const: 'NVIDIA RTX A5000' },
      precision: { const: 'NVFP4' },
      notes: { type: 'string', maxLength: 6000 },
    },
    required: ['repository', 'commit', 'deployment', 'model', 'gpu', 'precision'],
    additionalProperties: false,
  },
  score_name: 'Output throughput (tokens/s)',
  score_direction: 'higher',
  score_min: 0,
  score_max: null,
};

export const isNemotronBenchmark = (benchmark: BenchmarkInput) => benchmark.prompt === nemotronBenchmark.prompt;
