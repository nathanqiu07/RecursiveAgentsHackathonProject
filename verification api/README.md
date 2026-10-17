# Verification API

A server and mechanical execution nodes for jobs stored in Git.

**Submit a checkpoint → queue → node checks it out → run the job's runner → upload its result.**

The node follows the committed runner script. The runner owns the job's work and returns JSON. The server handles registration, queueing, leases, cancellation, and results.

## Start the server

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --python 3.12
uv run uvicorn verification_api.main:app --host 127.0.0.1 --port 8000
```

API documentation: <http://127.0.0.1:8000/docs>. For remote nodes, bind to a reachable interface behind your private network or HTTPS endpoint. Set `VERIFIER_API_KEY` on the server and nodes to enable bearer authentication.

## Turn a machine into a node

Clone this repository on the machine, enter its directory, and run:

```sh
./scripts/setup-node.sh --server https://YOUR_SERVER --node-id node-01
```

The script prepares an isolated Python environment using an existing `uv`, or Python 3.10+ with pip as a fallback. With uv it installs the locked dependencies. It then checks:

- Linux/macOS support, the runner's Python version, and process-group support.
- Writable cache storage, available disk, CPU count, and system RAM.
- Actual Git commit/worktree checkout and SHA-256 object-format support using a disposable local repository.
- Server health, authenticated access, and API compatibility.
- Node ID conflicts, registration, heartbeat, and a config readback from the server.

It saves a stable node ID, detected machine config, and a readiness report under `data/node`, then starts waiting for jobs. A local lock prevents two setup/node processes using that state directory. Re-running the same command reuses the saved identity. The process stays in the foreground; reboot startup and process supervision are not installed automatically.

Useful options:

```sh
# Check readiness and server access without registering or claiming work.
./scripts/setup-node.sh --server https://YOUR_SERVER --check-only

# Register and synchronize, then exit without claiming work.
./scripts/setup-node.sh --server https://YOUR_SERVER --node-id node-01 --enroll-only

# Require working NVIDIA GPU compute on Linux.
./scripts/setup-node.sh --server https://YOUR_SERVER --node-id gpu-01 --require-gpu

# Use the Python environment required by your job scripts.
./scripts/setup-node.sh --server https://YOUR_SERVER \
  --runner-python /opt/jobs/bin/python --label pool=verification
```

`--require-gpu` checks NVIDIA inventory and launches a tiny CUDA kernel through the driver API on each visible CUDA device, verifies its output, and cleans up the allocations/contexts. It uses the standard library and does not need PyTorch or nvcc. The driver must already be installed. See [NVIDIA's driver API guide](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-programming-guide/03-advanced/driver-api.html) and [NVIDIA SMI documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html).

Use `--min-free-disk-gb` and `--min-ram-gb` to change the baseline 1 GiB checks. These are enrollment thresholds, not a guarantee that a machine meets every job's resource requirements. Dependencies, model/dataset access, and tools specific to a job are handled by its runner. GPU checks are opt-in; normal nodes do not need a GPU.

Credentials come from `VERIFIER_API_KEY` in the environment, not a command-line option or saved config. `data/node/node.json` contains the detected configuration and `data/node/preflight.json` records readiness/sync checks. Changing an existing state's ID requires a new `--state-dir`. The setup rejects an ID already associated with another detected machine, and an ID with a running job.

To run an already configured worker directly:

```sh
uv run python -m verification_api.worker \
  --api-url http://127.0.0.1:8000 --node-config examples/node.json
```

The manual node example is declared metadata; setup detects the actual machine. `--node-id` can replace the config file, and `--worker-id` remains an alias.

## Submit a job

```http
POST /jobs
Content-Type: application/json
Authorization: Bearer YOUR_KEY
```

```json
{
  "repository": "https://github.com/your-org/your-project.git",
  "commit": "FULL_40_OR_64_CHARACTER_COMMIT_HASH",
  "manifest_path": "verification.json"
}
```

The response is HTTP 202 with the job `id` and `status: queued`. Poll `GET /jobs/{id}` for status and the report. `manifest_path` defaults to `verification.json`. Git URLs use HTTP(S), and private repositories use host-managed Git credentials. Embedded URL credentials, branch names, abbreviated commit hashes, and paths outside the checkout are rejected.

Each checkpoint must include its manifest and Python runner script:

```json
{
  "schema_version": 1,
  "runner_script": "scripts/run_job.py",
  "parameters": {"file": "input/data.json"},
  "timeout_seconds": 300
}
```

Parameters are passed through unchanged. The runner is executed from the pinned checkout using the node's selected Python interpreter, without a shell. The node supplies:

| Environment variable | Value |
| --- | --- |
| `VERIFIER_JOB_FILE` | Absolute path to the validated manifest |
| `VERIFIER_CHECKOUT_DIR` | Absolute source checkout path |
| `VERIFIER_CACHE_DIR` | Persistent artifact cache |
| `VERIFIER_NODE_ID` | Execution node ID |
| `VERIFIER_NODE_CONFIG_FILE` | JSON file containing the node ID and job config snapshot |

The script writes **one JSON object to stdout**, logs to stderr, and exits nonzero on execution failure:

```json
{
  "schema_version": 1,
  "result": {"passed": true, "details": {"checked": 10}},
  "metadata": {"runner_version": "your revision"}
}
```

The server preserves the runner's result. `completed` means the runner exited successfully and produced a valid report; inspect `result` for the job's own outcome. The node adds its ID/config, resolved commit, runner path, manifest checksum, and checkout setup duration under `report.worker`.

`examples/verification.json` and `examples/runners/check_file.py` form a working file-verification example. `examples/job.json` points to that manifest; replace the placeholder commit with a real repository commit before submitting it.

### Inference benchmarks

Benchmark runners can additionally use the optional `engine` and `benchmarks` manifest fields in `examples/benchmark.json`. The caller specifies all workloads, concurrency, input/output budgets, warmup counts, and other parameters in Git. No test suite or concurrency sweep is added automatically.

A benchmark runner returns `benchmarks` instead of, or alongside, `result`. Each entry includes `id`, `status` (`completed` or `failed`), optional finite nonnegative `metrics`, metadata, and an error for failed entries. When the manifest lists benchmarks, every requested benchmark ID must appear exactly once in the report. Workload implementations and accurate measurement belong to the runner. No inference engine, dataset, or performance result is fabricated by this API.

## API

| Endpoint | Request | Returns |
| --- | --- | --- |
| `POST /jobs` | Git URL, commit, optional manifest path | 202: queued job |
| `GET /jobs/{id}` | Job ID | Status, source, node/config snapshot, report/error |
| `POST /jobs/{id}/cancel` | Job ID | Cancelled job |
| `POST /nodes/register` | `id`, `config` | Registered node/config/status |
| `GET /nodes` | — | Nodes and their status/config |
| `GET /nodes/{id}` | Node ID | Node config, status, active job, timestamps |
| `POST /nodes/{id}/heartbeat` | Node ID | Updated node |
| `POST /worker/claim` | `node_id` | Claimed job and lease token, or null |
| `POST /worker/jobs/{id}/heartbeat` | `lease_token` | Renewed lease duration |
| `POST /worker/jobs/{id}/finish` | Lease token, status, report/error | Updated job |
| `GET /health` | — | `{"status":"ok"}` |

Schemas are in `schemas/`: submission, manifest, report, node registration, and node response. OpenAPI is available at `/openapi.json`.

## Queue and node behavior

SQLite persists jobs, node configs, and results in `data/jobs.sqlite3`; change it using `VERIFIER_DB_PATH`. Use persistent local storage. Existing databases gain the node configuration column automatically. Claims are atomic across server processes.

Nodes must register before claiming jobs and hold at most one running job. Node statuses are `idle`, `busy`, or `offline` after 120 seconds without contact. Configuration is copied into a job at claim time. Re-registering an idle node updates future jobs without changing historical snapshots. A busy node cannot change config. IDs are 1–100 letters/digits/dots/underscores/hyphens and begin with a letter or digit. `worker_id` remains a compatibility alias for `node_id` in claim requests and job responses.

Each job has a 120-second lease, renewed every 20 seconds while commands run. Cancellation revokes the lease; the node stops the command's process group at its next heartbeat. Failed renewal also stops local execution. Expired jobs become failed on subsequent queue access and are not automatically retried. Checkout has a separate 120-second timeout; the manifest sets the runner timeout, capped at one day.

Git objects and artifacts are cached; disposable worktrees are removed after execution. Manifests are limited to 1 MiB, runner stdout to 2 MiB, and other command output streams to 16 MiB. Logs stay in the node cache's `jobs/JOB_ID` directory. Cache, job, and log retention are manual. Host capability routing and GPU allocation across nodes are not implemented, so use compatible nodes and unique IDs for each machine/allocation.

## Deployment scope

This is a trusted-operator prototype. Runners execute repository code with node permissions. Untrusted submissions require isolated disposable machines and resource/network controls. The shared service key has no separate user/worker roles or per-user job ownership. The node removes that key from runner environments, but other inherited variables remain available; use a dedicated account with only needed credentials. Health and API docs are public.

The setup script prepares the application environment and checks readiness; it does not install system packages/drivers or configure reboot startup. Public deployment also needs quotas, role separation, isolation, retention, and scheduling.

## Validate

```sh
uv run pytest
uv run ruff check .
sh -n scripts/setup-node.sh
```

Tests cover actual local Git/worktree execution, generic and benchmark runner reports, queue exclusivity/durability, leases, cancellation, node registration/config snapshots, database migration, enrollment checks, auth failures, and sync. GPU test responses are simulated in the automated suite; the CUDA kernel smoke test must run on the target GPU machine.
