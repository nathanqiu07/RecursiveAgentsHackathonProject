# Local RTX A5000 mini verifier

The API, durable evaluation queue, MMLU/GSM8K harness, PPL harness, and load generator run on your local computer. Only inference runs in the GPU VM on your resource server. This service never clones or executes a submitted repository. The operator deploys reviewed source and registers its exact public GitHub repository, full commit, model revision, execution format, and engine settings.

The target is one NVIDIA RTX A5000 with 24 GB VRAM and the pinned `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` checkpoint, revision `bee7596271d1495f6992ae224aefde4410e816b8`. `NVFP4` names the weight checkpoint. Execution on Ampere must use a supported **W4A16** path; native FP4 arithmetic is not available on this GPU. The registry describes intended engine settings; installing an engine does not establish that the checkpoint fits or that its quantization backend supports this model. The real baseline must complete first.

## Start locally

From `verifier/`:

```sh
uv sync --extra evaluation --group dev
cp config.example.json config.json
cp .env.example .env
uv run python -m nemotron_verifier.prepare --output data/mini
```

`data/mini` must be empty when preparing. Preparation downloads public datasets and the tokenizer, freezes their revisions and sample hashes, and runs without a GPU. An existing prepared suite should be retained. `data/` is ignored by Git. Configuration paths are relative to the configuration file.

Use the resource-node enrollment flow in `../nodes/` to create an authenticated local SSH forward, then start the operator-approved inference job in its GPU VM. The verifier's `hardware.base_url` points at that node's loopback `/v1` service. Its authenticated `/v1/hardware` must return a **fresh** probe of the active VM, including the actual GPU UUID, model, memory, power limit, and driver. An enrollment smoke test or an idle VM is insufficient. The local CPU never runs `nvidia-smi`.

Set `NEMOTRON_RESOURCE_NODE_KEY` to the scoped node key in the private local environment. Edit deployment URLs to use the node inference proxy or separately configured local SSH forwards. URLs must have the form `http://127.0.0.1:PORT/v1`. Set an independent random `NEMOTRON_VERIFIER_API_KEY`, with the dashboard's `BENCHMARK_VERIFIER_NEMOTRON_KEY` set to the same value. Keep `.env` private. Registry JSON contains only the **names** of key environment variables. Keys are never part of submissions or evaluation feedback.

Start the API and the single worker with that environment loaded:

```sh
uv run uvicorn nemotron_verifier.main:app --host 127.0.0.1 --port 9000
uv run python -m nemotron_verifier.worker
```

The development launcher at `../dev/local_stack.py` may manage these processes alongside the local dashboard; see the root instructions. Bind the verifier to loopback. No cloud hosting, paid cloud VM, or external verifier service is required.

`GET /health` is unauthenticated and reports `ready`, `baseline_ready`, and `baseline_status`. It reports setup required until the pinned suite and live RTX A5000 metadata are available. Every other route requires the verifier bearer key. `POST /baseline` queues measurement; `GET /baseline` returns `{metrics, evaluation}`. Repeating the POST coalesces a queued/running baseline or returns the existing completed baseline. A failed or interrupted baseline can be retried by an explicit POST. Restarting the worker never silently repeats interrupted evaluation.

## Fixed mini protocol

These are **mini regression checks**, not full MMLU or GSM8K scores:

| Check | Frozen definition |
| --- | --- |
| Mini MMLU | Seed 42; 57 test questions, one per each of 57 subjects; 0-shot; greedy answer-letter generation; maximum eight output tokens |
| Mini GSM8K | Seed 42; 16 test problems; 0-shot step-by-step prompt; maximum 512 output tokens; final `#### number` exact match |
| Mini Wikitext PPL | First 4,096 tokens of Wikitext-2 raw test, pinned model tokenizer; teacher-forced probabilities; 2,048-token windows, stride 1,024; overlap masked, 4,095 next-token targets |

PPL requires prompt-token log probabilities from `/v1/completions` with `echo=true`, `logprobs=1`, and `max_tokens=0`. Engines lacking that capability fail the evaluation. Generated-token likelihoods are not a substitute. Truncated answer generation counts as incorrect. Quality generation uses four concurrent local clients. The measured end-to-end runtime is recorded against an **advisory 10–20 minute target**; a slow valid run is never rejected or killed for exceeding that target. HTTP transport failures remain evaluation failures. No evaluation wall-clock deadline is imposed.

Measure the unmodified reference first. Its identity binds the exact model revision, dataset revisions, frozen sample manifest, protocol, actual GPU UUID and driver/power configuration, and reference source/settings. Completed baselines are immutable. A candidate may add at most **one incorrect MMLU answer**, at most **one incorrect GSM8K answer**, and at most **1% PPL** relative to that reference. The candidate stops before all throughput warmups if any quality gate fails; it receives no leaderboard score.

A passing candidate is timed at **C20 / 2,048 input / 1,024 output**: one warmup request and 20 measured requests, with up to 20 client requests in flight. Twenty local client workers issue one measured request each, including engine queueing in total wall time. Throughput is total measured output tokens divided by total elapsed wall time, including dispatch, engine queueing, prefill, generation, and API overhead. Every completed request must report exactly the required token counts. Engines may use a smaller active sequence limit, such as four, to fit VRAM; queued requests still count toward the C20 client workload. Frozen distinct input windows reduce warmup reuse; record and control prefix caching in engine settings.

One A5000 hosts one active inference VM/engine. Measure reference, stop it, then deploy the candidate on the **same GPU UUID** before candidate evaluation. If both aliases use the same node proxy, they may share a URL, but registration must describe the engine actually running. Different GPUs or changed drivers/settings require a new baseline and new benchmark; the API permanently binds each benchmark ID to its first accepted reference identity. Candidate source and settings are also frozen when queued.

The deployment registry is an operator trust boundary. The service checks the reported model name and live hardware but does not cryptographically attest the engine's loaded weights or deployed source. Register only deployments you verified. Never point an alias at unreviewed or concurrently changing jobs.

## Dashboard evaluation contract

`POST /evaluate` accepts the existing dashboard envelope `{submission_id, benchmark_id, base_checkpoint_url, submission}`. The submission has only `repository`, `commit`, `deployment`, `model`, `gpu`, `precision`, and optional `notes`. Use GPU `NVIDIA RTX A5000`, precision `NVFP4`, the pinned model ID, and an operator-registered deployment alias. No endpoint or key is accepted in a submission. `Idempotency-Key` must equal `submission_id`.

The 202 response contains `{id, status}`. Poll authenticated `GET /evaluations/{id}`. States are `queued`, `running`, `completed`, `rejected`, or `failed`. Only a quality-passing completed candidate has a numeric `score`, measured in aggregate output tokens/second. Metrics include `mmlu`, `mmlu_correct`, `mmlu_count=57`, `gsm8k`, `gsm8k_correct`, `gsm8k_count=16`, `ppl`, and the immutable `identity`; percentage fields use 0–100. A reference baseline never receives a leaderboard score.

Run checks with `uv run pytest -q`. Tests exercise exact gates, teacher-forced overlap masking, remote hardware authentication, durable dispatch/recovery, baseline retry and immutability, benchmark identity binding, settings/source pinning, C20 client concurrency, and rejection before throughput. They do not claim a measured model score or a verified memory fit without an actual A5000 run.

## Local coordinator for three GPU nodes

The API runs on this computer at `http://127.0.0.1:9000`. Each physical A5000
has one exclusive VM slot shared by verification and Experiments. The coordinator
starts up to three independent evaluations in parallel, never splits a single
benchmark across GPUs, and runs all measurement code locally.

```sh
# Enroll the actual hosts first; see nodes/README.md for SSH/GPU/image prerequisites.
./nodes/enroll.sh --agents --inventory "$HOME/.crusoe-nodes/inventory.json" \
  --parallel 3 --state "$HOME/.crusoe-nodes" --resources-dir "$PWD/Resources"

# Private operator config; edit node IDs/receipt paths and approved deployments.
cp verifier/pool.example.json verifier/config.json
chmod 600 verifier/config.json
python3 dev/local_stack.py up --only nemotron-api,nemotron-worker
```

Every node's approved deployment map must contain the reference and candidate
aliases. Set its exact command and metadata in the enrollment inventory. The
coordinator selects only an operator-approved alias with `X-Deployment-Alias`;
it does not execute commands from submissions. The model weights, engine, and
GPU guest image must already be installed and cached. Example settings are a
starting point, and do not establish that this 24 GB GPU can load the model.
Live hardware metadata must match the pinned source, model revision, execution
mode, and settings before measurement begins.

`POST /baseline` queues one immutable reference per physical GPU. `GET /baseline`
returns every node's baseline and an actual first-node summary only when the
entire pool is measured. Candidates compare to their assigned node's reference.
The pool identity freezes the model, mini-suite, reference settings and enrolled
GPU/image identities for the benchmark. An operator change requires a new
benchmark identity; queued candidates also reject changed runtime settings.

Authenticated routes (Bearer key from the private `dev/data/local-stack.env`):

| Route | Purpose |
| --- | --- |
| `GET /nodes` | Real available/busy/offline node states, no node keys |
| `POST /baseline` | Measure each node's unmodified reference |
| `GET /baseline` | Per-node frozen measurements |
| `POST /evaluate` | Queue a pinned candidate, idempotent by submission ID |
| `GET /evaluations` | Recent persistent queue history |
| `GET /evaluations/{id}` | Assignment, result and measured duration |
| `POST /evaluations/{id}/cancel` | Explicitly stop a queued/running evaluation |

The 10–20 minute goal is advisory, measured on a prepared GPU VM. Queue wait,
VM boot, model loading and cleanup are reported separately from the measurement.
Slow valid runs remain eligible. The API does not impose a 10- or 20-minute
cutoff. Operator VM safety timeouts remain configurable (six hours requested by
default, capped by the enrolled node's configured limit). No measurements or
runtime guarantees are claimed before a real A5000 completes the workload.

An experiment that already holds a node makes verification wait. A racing busy
response requeues only an unaccepted VM request. A lost VM acknowledgement is
never blindly replayed: the coordinator cancels the same persisted UUID. On
restart interrupted evaluations are failed and their recorded VMs are cancelled;
they are never silently re-run or scored from partial measurements. Node cleanup
must be confirmed before a score is saved. API and worker use the same SQLite DB.

The same registered resources appear under Agents → Experiment resources. Their
availability is read from the node service and Resource Manager reservations,
so an active verifier VM is busy for Experiments too. Private connection receipts,
keys, configuration, model weights and data caches stay outside Git.
