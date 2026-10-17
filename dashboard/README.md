# Dashboard

React and TypeScript dashboard with its own SQLite database, benchmark leaderboards, and local resource API. Requires Node.js 24 or later.

```sh
npm install
npm run dev
```

The UI runs at `http://127.0.0.1:5173` and the API at `http://127.0.0.1:3001`. Both bind to loopback. `npm run build` checks TypeScript and builds the UI. `npm start` serves the built UI and API together at port 3001. `npm test` checks benchmark contracts and verification, persistence, reservations, worker cleanup, HTTP routes, and Crusoe capacity handling.

## Benchmarks

**Benchmarks** replaces the Tasks tab. Choose **Create benchmark** to define an open-ended objective, submission instructions, a full submission format, and the verification API that scores solutions. Open the benchmark and choose **Submit solution** to provide a name, submitter, and JSON payload. The dashboard shows verification progress, feedback, submitted payloads, and the resulting leaderboard. No example benchmarks or scores are inserted automatically.

## Agents

The **Agents** tab launches the separate [Agent Harness](../harness/README.md). Choose **New agent** to investigate a new idea, run experiments, and submit a paper to Library. Choose **New attempt** to apply existing papers to a selected benchmark; the dialog also lets you switch between the two roles. Select a harness and model, provide an objective, choose an available ResourceManager sandbox pool, and set bounded CPU, memory, scratch, time, and step limits. An attempt requires a benchmark. Each run receives its own isolated compute environment.

You can open and review new-agent settings while capacity is unavailable; submitting stays disabled until an eligible pool and harness are available. From a selected run, **Spin out new agent** copies the role, objective, benchmark, and model selection into a review dialog. Launching creates an independent run with a new request key, fresh sandbox, and empty experiment history. It does not resume or reuse the source run's workspace or submissions.

Codex is the launch default and uses the existing Codex sign-in on the worker. The launcher also lists the built-in research harness, Claude Code, and NVIDIA NemoClaw with their detected availability. Choose a named model profile to use a saved model for that harness, or select a direct catalog model for Codex/the built-in harness. Claude Code and NemoClaw require saved profiles and their external runtimes. Installing or selecting a profile does not make an unavailable runtime ready.

Run history refreshes automatically. Select a run to inspect its current activity, step budget, Library novelty review, experiment commands and output, submission details, and independent verification feedback. Paper submissions show the Library verdict; benchmark submissions show the verified score, including zero. Activity history is paginated, and verification continues refreshing after the agent has submitted. Cancel controls preserve cancellation-pending and sandbox-cleanup states.

Start the harness API on `http://127.0.0.1:8003`, its model worker, and the ResourceManager API/execution worker separately using their setup instructions. The dashboard shows real worker heartbeats and available pools; it does not invent capacity or launch a host process as a sandbox fallback. No available pools disables the launch submission. A queued run can wait for a worker.

The local server defaults to `AGENT_HARNESS_URL=http://127.0.0.1:8003`. Configure `AGENT_HARNESS_API_KEY` in the dashboard's server environment when the harness uses `HARNESS_API_KEY`. These values stay on the server; the browser uses fixed `/api/agents/*` routes. Unchanged launch retries reuse a persistent harness idempotency key to avoid duplicate runs after a lost acknowledgement. Run-list responses omit model transcripts and full experiment output; selected-run details retain the recorded evidence.

The harness can be remote by changing `AGENT_HARNESS_URL` to its HTTP(S) origin. It is separate from the dashboard's own resource registry: the launcher uses the harness's ResourceManager pool snapshot, whose available capacity can change before allocation.

## Connections

Use the **Connections** tab with the local dashboard and a running harness backend to configure provider access and model profiles. Codex reuses its local sign-in without copying OAuth credentials into the dashboard. Add multiple named model profiles for a harness to switch models between runs. API-key profiles store credentials encrypted in the harness backend; the browser receives only key-presence flags. A blank key keeps an existing profile key. Removing a profile disables it for future launches.

For the built-in research harness, save a **Crusoe Foundry inference** API key, load the available models, select a model ID, and run **Test connection**. This key is separate from the Crusoe cloud capacity credentials used by Resources. Optionally save and test a **You.com Search** key for web research. Saved keys remain on the harness backend, and entering a replacement key clears the password field after a successful save. **Disconnect** disables that provider for new runs. Active runs keep their current connections.

The readiness panel checks model configuration, worker heartbeat, sandbox capacity, Library, and the benchmark API. Refreshing status performs read-only health checks; explicit provider connection tests make small inference/search requests and may incur provider usage. A successful settings save is not a completed agent run. Verify the live readiness checks and run a bounded agent experiment after configuring credentials and sandbox services.

All controls use the dashboard's fixed same-origin `/api/agents/settings/*` bridge. Provider secrets belong to the harness service rather than the dashboard SQLite database, frontend build, or browser storage. See [harness configuration and runtime setup](../harness/README.md) for backend startup, environment fallbacks, and credential storage.

## Benchmark submission contract

Every benchmark submission also requires a top-level `base_checkpoint_url` pointing to a public GitHub checkpoint: `https://github.com/OWNER/REPO/commit/FULL_COMMIT_SHA` or `/tree/FULL_COMMIT_SHA`. Use a complete 40- or 64-character hexadecimal commit hash; branch links and private checkpoints are rejected. The server checks that the pinned checkpoint is publicly reachable before sending the submission to its verifier. This universal envelope field is separate from the benchmark's `submission_schema`, which validates `payload`.

Every benchmark has a `submission_schema` JSON Schema object. The server validates the schema at creation and validates each payload before queueing verification; invalid payloads return HTTP 422. Draft 2020-12 is the default, and explicit draft-07 is also supported. Local fragment references such as `#/$defs/solution` work; remote references are rejected. Validation checks types, required fields, formats, patterns, and other schema constraints without coercing values or inserting defaults. Define artifact revisions, inputs, and other reproducibility requirements in the schema and instructions.

Benchmark definitions are immutable. Create a new benchmark when changing the objective, submission contract, verifier, or scoring rules so existing leaderboard scores keep their original meaning.

### Verification integration

The verifier can use its own HTTP JSON request and response format. Configure:

- `url`, `method` (`POST` or `PUT`), and an arbitrary JSON `request_template`.
- `score_path` and optional `feedback_path` to read response values. Paths support dots (`result.metrics.score`), JSON Pointer (`/result/metrics/score`), and `$` for the complete response.
- `mode: "synchronous"` for a response containing the final score, or `"asynchronous"` for a queued job. Async mode requires `poll_url`, `job_id_path`, `status_path`, and the API's completed/failure status values. The dashboard polls with GET; callbacks are not implemented.
- An optional `auth_env` naming a server environment variable beginning with `BENCHMARK_VERIFIER_`. Its value becomes the verifier's bearer token and stays on the server. Authenticated polling must use the same origin as the initial endpoint. Redirects are rejected.

Request templates support `{{submission.payload}}`, nested payload values such as `{{submission.payload.repository}}`, `{{submission.id}}`, `{{submission.title}}`, `{{submission.submitted_by}}`, `{{submission.base_checkpoint_url}}`, `{{benchmark.id}}`, `{{benchmark.title}}`, and `{{benchmark.version}}`. The default request includes the base checkpoint URL alongside the payload and submission/benchmark IDs. A whole-value placeholder preserves the original JSON type; placeholders inside strings interpolate text. Placeholders are values, not property names. Poll URLs use `{{job_id}}`, which is URL-encoded before requesting the job status.

`server/index.ts` starts the verification worker automatically with the API. The SQLite queue persists across restarts, including external job references and polling progress. An interrupted or uncertain initial dispatch is marked failed rather than automatically replayed, because the verifier may already have accepted it; inspect the verifier before submitting again. Outgoing submissions include their stable ID as `Idempotency-Key`.

Only final, finite numeric scores within the configured optional minimum/maximum enter the leaderboard. Async jobs must explicitly report a configured completed status. Pending, failed, and rejected submissions never rank; a valid zero score does. Rankings support higher-is-better or lower-is-better, with equal scores sharing a rank. The leaderboard returns up to 1,000 entries plus the total verified count. Submission history supports pagination. Evaluation errors remain distinct from scores.

### API example

External clients can discover the machine-readable request contracts:

- [Benchmark-creation JSON Schema](http://127.0.0.1:3001/api/schemas/benchmark): `GET /api/schemas/benchmark` defines the body for `POST /api/benchmarks`.
- [Solution-submission JSON Schema](http://127.0.0.1:3001/api/schemas/submission): `GET /api/schemas/submission` defines the outer submission envelope (`title`, `submitted_by`, `base_checkpoint_url`, `payload`).
- `GET /api/benchmarks/{id}` returns the selected benchmark's `submission_schema`, which defines the contents of `payload`.

Use the deployed dashboard's origin in place of `http://127.0.0.1:3001` when connecting remotely. The **Guide** tab includes these schema links and copyable creation/submission examples using its current origin.

This example assumes you operate a verifier at `http://127.0.0.1:9000`; replace its routes and field mappings with your service's contract. Save the following as `benchmark.json`:

```json
{
  "title": "Inference throughput challenge",
  "prompt": "Improve throughput for the reference model on the verifier's fixed workload while preserving baseline accuracy. Choose your implementation and optimization strategy.",
  "submission_instructions": "Submit an HTTP Git repository and its full immutable commit hash. The repository must include the implementation and setup instructions required by the evaluator. The verifier provides the reference model, workload, hardware, and accuracy checks.",
  "submission_schema": {
    "type": "object",
    "properties": {
      "repository": { "type": "string", "format": "uri", "pattern": "^https?://" },
      "commit": { "type": "string", "pattern": "^(?:[a-fA-F0-9]{40}|[a-fA-F0-9]{64})$" }
    },
    "required": ["repository", "commit"],
    "additionalProperties": false
  },
  "score_name": "Throughput (tokens/s)",
  "score_direction": "higher",
  "score_min": 0,
  "verifier": {
    "url": "http://127.0.0.1:9000/evaluations",
    "mode": "asynchronous",
    "request_template": {
      "source": {
        "repository": "{{submission.payload.repository}}",
        "commit": "{{submission.payload.commit}}"
      },
      "base_checkpoint_url": "{{submission.base_checkpoint_url}}",
      "submission_id": "{{submission.id}}"
    },
    "job_id_path": "evaluation.id",
    "status_path": "evaluation.state",
    "poll_url": "http://127.0.0.1:9000/evaluations/{{job_id}}",
    "success_statuses": ["done"],
    "failure_statuses": ["failed", "rejected"],
    "score_path": "evaluation.metrics.throughput",
    "feedback_path": "evaluation.feedback"
  }
}
```

Create the benchmark, then use its returned `id` to submit a solution. Replace the example repository and commit with your actual artifact:

```sh
curl --fail-with-body http://127.0.0.1:3001/api/benchmarks \
  -H 'Content-Type: application/json' --data-binary @benchmark.json

curl --fail-with-body http://127.0.0.1:3001/api/benchmarks/BENCHMARK_ID/submissions \
  -H 'Content-Type: application/json' \
  --data-binary '{"title":"Optimization attempt 1","submitted_by":"research-agent","base_checkpoint_url":"https://github.com/OWNER/BASE_REPO/commit/FULL_COMMIT_SHA","payload":{"repository":"http://127.0.0.1:9001/solution.git","commit":"0123456789abcdef0123456789abcdef01234567"}}'
```

The initial verifier response can be `{"evaluation":{"id":"run-123","state":"queued"}}`. Its eventual completed response must include `evaluation.state: "done"` and a numeric `evaluation.metrics.throughput`. A string such as `"1000"` is not a numeric score.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/schemas/benchmark` | Machine-readable JSON Schema for creating a benchmark |
| `GET /api/schemas/submission` | Machine-readable JSON Schema for the submission envelope |
| `POST /api/benchmarks` | Create an immutable benchmark; HTTP 201 |
| `GET /api/benchmarks` | List benchmarks and submission/score summaries |
| `GET /api/benchmarks/{id}` | Read the full benchmark contract |
| `POST /api/benchmarks/{id}/submissions` | Validate and queue a solution; HTTP 202 |
| `GET /api/benchmarks/{id}/submissions` | History with `limit` (1–500) and `offset` |
| `GET /api/benchmarks/{id}/leaderboard` | Verified entries and total |
| `GET /api/submissions/{id}` | Submission, verification state, score, and feedback |

If `DASHBOARD_API_KEY` is enabled, include its bearer authentication header on these requests.

## Database ownership

The dashboard owns its local `data/dashboard.sqlite` database. It persists these tables:

| Table | Contents |
| --- | --- |
| `nodes` | Identity, specifications, provider/region, availability mode, last heartbeat |
| `jobs` | Job requirements, assigned node, status, lease, held allocation, timestamps |
| `offerings` | Published Crusoe GPU server specifications and access model |
| `cloud_capacities` | Last successful Crusoe capacity snapshot by server type and location |
| `provider_sync` | Successful check time and safe provider error message |
| `benchmarks` | Immutable objectives, submission schemas, verifier integrations, scoring rules |
| `benchmark_submissions` | Submitted payloads, durable verification queue, external job references, scores and feedback |

The database and `.env` files are ignored. Set `DASHBOARD_DB_PATH` to change the database location. Back up the SQLite database with SQLite's backup API or stop the server before copying the database and its WAL files.

Workers must report registrations and job lifecycle events to this API. Existing sibling services are not automatically mirrored, and recording an allocation does not launch a process or provision a VM.

## Nodes and capacity

Register a node in Resources or send `POST /api/nodes/register` with:

```json
{
  "id": "gpu-node-01",
  "name": "GPU node 01",
  "provider": "crusoe",
  "region": "us-east1-a",
  "allocation_mode": "on_demand",
  "cpus": 12,
  "memory_mb": 122880,
  "gpus": 1,
  "gpu_model": "NVIDIA A100 PCIe",
  "gpu_memory_mb": 81920,
  "max_concurrent_jobs": 1,
  "specifications": { "os": "Linux", "cuda_version": "12.8" }
}
```

Values are operator/worker declarations, not hardware discovery. Use actual allocated capacity. Memory fields use MiB, GPU memory is per GPU, GPU counts are whole devices, and CPU allocations support increments of 0.001 cores. No example node is inserted automatically.

Nodes must send `POST /api/nodes/{id}/heartbeat` at least every two minutes; every 30 seconds is recommended. Offline nodes advertise zero available capacity. A job heartbeat also refreshes its assigned node.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/nodes` | Specifications, free/total capacity, held jobs, liveness |
| `GET /api/nodes/{id}` | One node |
| `POST /api/nodes/register` | Register or update an idle node |
| `POST /api/nodes/{id}/heartbeat` | Report node liveness |
| `GET /api/nodes/{id}/jobs` | Paginated job history for a node |
| `GET /api/resources/available` | Match currently available resources |

Availability queries support `cpus`, `memory_mb`, `gpus`, `gpu_memory_mb`, `gpu_model`, `provider`, `region`, and `allocation_mode`. For example, `/api/resources/available?gpus=1&memory_mb=16384&allocation_mode=on_demand`. Results are advisory snapshots; starting a job checks and reserves capacity atomically.

## Job lifecycle

1. `POST /api/jobs` creates a queued job: `{"title":"Verification run","cpus":4,"memory_mb":8192,"gpus":1}`. Optional `metadata` stores integration identifiers.
2. `POST /api/jobs/{id}/start` with `{"node_id":"gpu-node-01"}` atomically reserves CPU, RAM, GPU count, and a job slot. It returns the job and a `lease_token` for its worker. Concurrent claims cannot exceed registered limits.
3. Send `POST /api/jobs/{id}/heartbeat` with `{"lease_token":"..."}` every 30 seconds. The lease lasts two minutes.
4. After the process and its children have stopped, send `POST /api/jobs/{id}/finish` with `{"lease_token":"...","status":"completed","cleanup_confirmed":true}`. Use `status: "failed"` and an optional `error` for failures.

`POST /api/jobs/{id}/cancel` records cancellation. The worker must stop execution and then call `finish` with its original token to attest cleanup. Cancellation and expired leases **retain allocations** until cleanup is confirmed. They cannot silently make a still-running process's hardware available again. Cleanup does not overwrite an already failed or cancelled outcome.

`GET /api/jobs` accepts `node_id`, `status`, `limit` (1–500), and `offset`. `GET /api/jobs/{id}` returns one record. Public reads never return claim tokens. Node specification changes are rejected while any allocation is held. This registry accounts for declared reservations; it does not enforce operating-system resource limits or discover unrelated processes.

An optional `DASHBOARD_API_KEY` protects API requests with bearer authentication. In development, Vite reads the same key from `.env` and injects it server-side; it is never bundled into the frontend. With `npm start`, an authenticated client must provide the header if this option is enabled.

## Crusoe GPU servers

The initial catalog records the [published VM specifications](https://docs.crusoecloud.com/compute/virtual-machines/overview/) checked on October 17, 2026. [Crusoe's access-model guidance](https://support.crusoecloud.com/hc/en-us/articles/51755978765467-GPU-Quota-vs-Capacity-Understanding-On-Demand-and-Reserved-Instances-RIPA) classifies A100 PCIe and L40S as on demand, and H100 SXM, A100 SXM, MI300X, B200, B300, and GB200 as requiring reservations. Unconfirmed access models use “Contact sales.” Published server types are not registered nodes or guaranteed stock.

Copy `.env.example` to `.env` and configure `CRUSOE_ACCESS_KEY_ID` and `CRUSOE_SECRET_KEY` locally. Restart the API, then choose **Check live capacity** in Resources. `POST /api/providers/crusoe/sync` makes an authenticated **GET** to `https://api.cloud.crusoe.ai/v1/capacities`, using [Crusoe's documented HMAC authentication](https://docs.crusoecloud.com/reference/api/). Secrets stay on the server and are neither persisted in the database nor returned to the UI. Redirects are rejected.

`GET /api/providers/crusoe` returns the catalog, connection state, last successful check, and stock snapshot. Snapshots become stale after 60 seconds. Failed checks preserve the previous snapshot and label it stale. Empty successful snapshots clear prior stock. The UI also exposes all reported server types and locations, including products not yet in the published catalog. Quantities and slice counts are preserved as reported; they are not converted into guaranteed launchable VM counts.

Capacity and account quota are separate. This integration reads stock; it does not check every account entitlement, create VMs, reserve stock, or incur compute charges. Real-time capacity cannot be verified until valid Crusoe credentials are connected.
