# Agent Harness

Two autonomous agent roles use the existing Library, benchmark API, and ResourceManager:

| Role | Work | Submission destination |
| --- | --- | --- |
| `researcher` | Read prior work, investigate ideas online, check novelty against Library, run experiments, write a reproducible paper | **Library `POST /papers`**, for independent verification |
| `attempt` | Read a benchmark and relevant Library papers, implement and test promising approaches, optimize within the run budget | **Benchmark `POST /api/benchmarks/{id}/submissions`**, for independent scoring |

The attempt agent is also the task agent. It seeks the best supported solution within its budget; it cannot promise a global optimum. Only the researcher needs a novel contribution. Both can research the internet through the same brokered tools.

```text
                    Harness API → durable run queue → model/tool orchestrator
                                                            │
                     ┌──────────────────────────────────────┴─────────────┐
                     │                                                    │
          researcher run: own sandbox                         attempt run: own sandbox
          Library + web → novelty review                     benchmark + Library + web
                     ↓                                                    ↓
              experiments/code                                     implementation/tests
                     ↓                                                    ↓
           paper → Library verification                       solution → benchmark verifier
```

## Isolation

Each run allocates its own ResourceManager compute job and fresh container. Commands and files stay in that container's `/tmp/workspace`; no run shares a workspace. ResourceManager enforces the read-only root, non-root user, disabled networking, CPU/memory/scratch/time limits, dropped capabilities and process limit. Generated commands never run on the harness host.

The model loop and tool broker run on the trusted orchestrator. They hold service/model/search credentials and expose only the run's fixed capabilities. Library reads, model calls, web research and publication go through those clients. Sandboxes receive commands, not credentials or network clients. Model context contains the user's objective, paper text, web excerpts and experiment outputs; those are sent to the configured model provider.

The included image has Python's standard library and a shell. Install experiment dependencies in an operator-approved derived image before configuring its ResourceManager pool. The current manager supports CPU allocations, and its writable `/tmp` is `noexec`: interpreted scripts work, while generated native binaries cannot execute there. GPU measurements remain with the external benchmark verifier. Docker containers share their host kernel; use suitable execution hosts/runtime for your workloads.

## Start

Requires Python 3.10+ and `uv`. Start the [benchmark API](../dashboard/README.md), [Library](../library/README.md), and ResourceManager first. Library is included in this repository and runs as a separate HTTP service. ResourceManager is configured independently; its source is not included here. Configure their URLs and credentials using the settings below. ResourceManager must provide the generic compute-job and input/output-channel contract used by `agent_harness/clients.py`; Library must provide the paper intake and verification contract described below.

Configure an eligible, registered ResourceManager pool with this image installed on its Docker host:

```sh
cd harness
docker build -t agent-harness-sandbox:local .
uv sync --python 3.12
```

Set that pool's private `image` to `agent-harness-sandbox:local`, bind its real `node_id`, set quotas, and enable it. Refresh node heartbeats using ResourceManager's deployment configuration. Do not use the echo image: the harness needs its execution protocol. No image selection, mounts or Docker settings are accepted from agent/model requests.

Export configuration in both the API and worker shells (`.env.example` is a reference, not automatically loaded):

```sh
export HARNESS_LIBRARY_URL=http://127.0.0.1:8001
export HARNESS_BENCHMARK_URL=http://127.0.0.1:3001
export HARNESS_RESOURCE_MANAGER_URL=http://127.0.0.1:8002
# Export service API keys if these services require authentication.
uv run uvicorn agent_harness.main:app --host 127.0.0.1 --port 8003
```

In another configured shell:

```sh
cd harness
uv run python -m agent_harness.worker
```

The dashboard defaults to **Codex**, using the worker machine's existing `codex login` with ChatGPT. The harness does not extract its OAuth token or send it to the browser. Run `codex login status` on the worker to check the login; model access is confirmed by an actual inference turn, rather than the model catalog. See [Codex authentication](https://developers.openai.com/codex/auth).

For the built-in research harness, save a Crusoe key and model in **Connections**, or export `HARNESS_MODEL_BASE_URL`, `HARNESS_MODEL`, and `HARNESS_MODEL_API_KEY` for a compatible endpoint. This endpoint must support tool-calling chat completions at `/chat/completions` and JSON object responses. Research and independent novelty/evidence reviews use separate model calls. Multiple workers can share the local harness database; each run has one renewable owner lease and a distinct sandbox. Workers can start before credentials are configured and pick up settings between runs. `--once` processes one run or cleanup reconciliation. API docs are at `http://127.0.0.1:8003/docs`.

## Connections and model profiles

The dashboard's **Connections** tab saves provider keys on the harness backend, loads the Crusoe catalog, tests model tools/JSON and You.com search, and checks worker, Library, benchmark and sandbox availability. Tests are explicit requests; dashboard polling does not generate inference or search traffic. Passing dependency checks is not proof of a completed scientific run or verifier deployment.

Provider and model-profile keys are encrypted in SQLite. A master key is generated next to the database as `<database>.settings.key` with owner-only permissions, or supplied by `HARNESS_SETTINGS_KEY`. Keep this key with your private database backups and share it with API/worker processes using the same database. Losing it makes saved credentials unreadable. Public responses expose only whether a key is set. Blank key edits retain the existing key; disconnect removes the saved credential and overrides its provider environment fallback. Credentials never enter model prompts, compute containers, run events, browser storage, or source control.

Add multiple named model profiles per harness, then select a harness and profile when launching either agent role:

| Harness | Authentication and execution |
| --- | --- |
| Codex | Existing local ChatGPT OAuth sign-in; select a model from its catalog or a saved Codex model profile. No portable OAuth-key input. |
| Claude Code | Installed `claude` CLI plus an Anthropic Messages-compatible endpoint/model and API key. Native tools, MCP, plugins and hooks are disabled for planning. |
| NVIDIA NemoClaw | Installed NemoClaw, OpenShell and working Docker, with `HARNESS_NEMOCLAW_ENABLED=1` after runtime/image preparation. Each run provisions its own OpenShell planner sandbox and inference route, then destroys it on close. |
| Research harness | Direct compatible chat completions, normally the saved Crusoe connection. A model profile can select another compatible model/endpoint. |

For **NVIDIA NemoClaw**, add an **API key** model profile with the inference provider's base URL, exact model ID and matching key. Examples:

| Inference provider | API base URL | Example model ID | Required key |
| --- | --- | --- | --- |
| OpenAI | `https://api.openai.com/v1` | `your-openai-model-id` | OpenAI Platform API key with access to that model |
| Crusoe | `https://api.inference.crusoecloud.com/v1` | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B` | Crusoe Foundry inference key with access to that model |

Use a model available to your provider account; the Crusoe catalog in Connections lists its exact IDs. NemoClaw forwards these settings through its custom compatible endpoint using `NEMOCLAW_ENDPOINT_URL`, `NEMOCLAW_MODEL` and `COMPATIBLE_API_KEY`. It selects `NEMOCLAW_PREFERRED_API=openai-completions`, as described in [NVIDIA's compatible API guide](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/inference/custom-endpoints/choose-compatible-inference-api). Keys do not need an NVIDIA prefix. Existing Codex ChatGPT sign-in applies to the Codex harness; an OpenAI NemoClaw profile requires its own API key. Installing/configuring NemoClaw, OpenShell and Docker on the worker is still required before launching a NemoClaw run.

Codex and Claude Code run as trusted planning clients with native tools disabled. Their structured tool proposals go through this harness's existing novelty, experiment and publication gates; every generated experiment still executes only in the run's ResourceManager sandbox. NemoClaw additionally isolates its planner in OpenShell. The CLI adapters are not allowed to run model-selected commands on the worker host. Runtime availability reports installation/sign-in prerequisites, while endpoint/model access can still fail on a turn. NemoClaw onboarding can require image downloads and takes longer than an inference request.

The model selected for a run is recorded in its request/runtime metadata. A worker takes a fresh connection snapshot at the start of each run; editing a connection affects later runs. Disconnecting a provider does not interrupt a run already using that connection. ChatGPT OAuth applies to supported Codex models; Claude Code, NemoClaw and Crusoe use their own authentication. These API families are not interchangeable.

Settings endpoints (all use the harness's authentication):

- `GET /settings/providers`, `POST /settings/providers/{crusoe|you}`, `POST /settings/providers/{provider}/test` and `/remove`.
- `GET /settings/providers/crusoe/models`, `GET /settings/readiness` and `GET /settings/harnesses`.
- `POST /settings/harnesses/profiles` and `POST /settings/harnesses/profiles/{id}/remove`.

Dashboard requests go through the existing authenticated server bridge. Hosted settings require the same operator allowlist as agent controls and a reachable HTTPS harness. Local hosting is sufficient; the public Site cannot reach a loopback harness on your machine.

## Launch the two roles

Use real `resource_id` and `node_id` values from `GET /resources`, with allocations appropriate for the work and inside that pool's limits. Save this as `researcher.json`:

```json
{
  "role": "researcher",
  "objective": "Investigate a new approach to the target problem, compare it with prior Library work, run reproducible experiments, and submit a paper to Library.",
  "max_steps": 40,
  "allocation": {
    "resource_id": "YOUR_POOL_ID",
    "node_id": "YOUR_REGISTERED_NODE_ID",
    "cpus": 1.0,
    "memory_mb": 1024,
    "scratch_mb": 128,
    "timeout_seconds": 1200
  }
}
```

```sh
curl --fail-with-body http://127.0.0.1:8003/runs \
  -H 'Content-Type: application/json' --data-binary @researcher.json
```

For an attempt run, change `role` to `attempt`, add the actual `benchmark_id`, and set the objective to improving that benchmark. The benchmark's public immutable GitHub base checkpoint and any required public solution artifacts must be available to the agent (for example, include their URLs in `objective`). This harness supplies no GitHub write credentials and cannot publish inaccessible local files as public artifacts.

Each accepted run returns its `id`. Poll `/runs/{id}` for progress and `/runs/{id}/events` for tool activity and recorded experiments. Once submitted, `/runs/{id}/publication` retrieves current Library verification or benchmark evaluation status. Continue polling that endpoint for asynchronous benchmark scoring, including on the hosted dashboard where reads advance verification work.

The dashboard's **Agents** tab provides these launch controls and live views directly. Its local server connects to this API on port 8003 by default; set `AGENT_HARNESS_URL` and, when needed, `AGENT_HARNESS_API_KEY` on the dashboard server. Run history uses lightweight summaries, while selected runs retain experiments, reviews and submission details. `/status` reports worker heartbeats separately from API health, so queued runs are not presented as active without a worker.

`POST /runs` accepts an optional `Idempotency-Key` of 1–200 visible ASCII characters. The same key and normalized settings return the original run, including after restart; reusing a key for different settings returns 409. The dashboard retains the same key for an unchanged launch retry. Paper and benchmark publication remain separate non-idempotent operations with the uncertainty rules below.

## Publication requirements

The researcher must receive a novelty approval before executing commands, then record at least one successful, nontruncated experiment with output. A separate reviewer compares the **entire readable Library snapshot**, including unverified submissions, with the proposed idea. Before publication, the final paper receives another novelty review and an evidence review against this run's actual commands and outputs. A snapshot change during review requires a fresh review. The submitted paper includes the recorded evidence appendix and is queued for the Library's independent verification. Library rechecks novelty independently against its complete current snapshot and records that verdict separately from experiments. It cannot mark a paper verified unless novelty and experiments both pass; a changed Library snapshot requires a fresh review before the final commit.

Novelty approval is an LLM assessment relative to the recorded Library snapshot, not proof of worldwide novelty or a scientific verdict. URL-only papers without `paper_text`, incomplete pagination, or a Library beyond the bounded review context prevent approval rather than silently skipping prior work. The Library currently has no atomic novelty reservation, so external concurrent submissions can still arrive between the final snapshot check and publication.

The attempt agent must read its benchmark, search the Library, read at least one matching paper if matches exist, record a successful sandbox test, and pass an evidence review. Its payload is validated against the benchmark's own JSON Schema, including non-object payloads. The benchmark server checks public checkpoint access and independently computes the official score. Model-generated scores do not enter the leaderboard.

Runs have bounded steps, runtime, model context, command output, web bodies, and API responses. `completed` may mean the agent finished without publication or exhausted a budget; inspect `result` and `error`. `submitted` means acknowledgement by Library/benchmark, not successful verification.

## Internet research

Save a You.com API key in **Connections** (or export `HARNESS_YOU_API_KEY`/`YDC_API_KEY`) to enable native [You.com Search](https://you.com/docs/api-reference/search/v1-search): `POST https://ydc-index.io/v1/search` with `X-API-Key`, bounded result excerpts and source URLs. The key is never sent to result pages. Crusoe uses its [managed inference endpoint](https://support.crusoecloud.com/hc/en-us/articles/55002524169499-How-To-List-Model-Features-via-curl-and-the-Inference-API), separate from Crusoe capacity/HMAC credentials.

Alternatively, configure optional `HARNESS_SEARCH_URL` to an operator-owned search gateway accepting `GET ?q=...` and returning JSON results with source URLs. `HARNESS_SEARCH_API_KEY`, if configured, is sent as its bearer token. Without either provider, `web_search` reports `search_not_configured`; known public pages can still be read with `web_read`.

Public page reads support text/HTML over HTTPS on port 443. They reject private/local destinations, credentials and redirects, pin connections to vetted public IPs, and use the original TLS hostname. PDF extraction, arbitrary downloads and network package installation are not implemented. Research results are untrusted data. Search gateway responses should contain sources and excerpts, not provider credentials.

## API and durability

External Library must implement `GET /papers?limit=100&offset=...`, returning `papers` and `total`, `GET /papers/{id}`, and `POST /papers`. Paper records have UUID `id`, `status`, `submission` (including `title` and readable `paper_text`), and optional verification `report`. Submission accepts `title`, `paper_text`, `authors`, `submitted_by`, and `notes`; acknowledgement is HTTP 202 with `status: pending_verification`.

External ResourceManager must implement `GET /resources`, idempotent `POST /jobs`, `GET /jobs/{id}`, `POST /jobs/{id}/stdin`, `GET /jobs/{id}/stdout?after=...`, `POST /jobs/{id}/stdin/close`, and `POST /jobs/{id}/cancel`. Allocation uses the six fields in the launch example and returns HTTP 202 with a UUID job ID. Input is `{ "id": "message-id", "line": "opaque UTF-8 line" }`; output is `{ "lines": [{ "sequence": 1, "line": "..." }], "next_cursor": 1 }`. Jobs expose their status, `sandbox_id`, and `cleanup_required`. The approved image supplies this harness's NDJSON execution protocol; the manager must actually enforce the isolation and resource limits described above.

| Request | Purpose |
| --- | --- |
| `GET /agents` | Discover the two roles and their tools |
| `GET /status` | Worker availability, last worker heartbeat, and queued/running counts |
| `GET /resources` | Read ResourceManager's available pools |
| `POST /runs` | Queue a researcher or attempt run |
| `GET /runs` | Paginated run history |
| `GET /runs/{id}` | State, publication acknowledgement, errors and cleanup status |
| `GET /runs/{id}/events?after=0` | Paginated audit events with a cursor |
| `POST /runs/{id}/cancel` | Request sandbox cancellation |
| `GET /runs/{id}/publication` | Current independent paper/submission verification status |
| `GET /health` | Process health |

Set `HARNESS_API_KEY` to protect all endpoints except health and API docs. With a key enabled, add `Authorization: Bearer ...` to the example requests. Keep the API private: it can schedule compute and submit papers/solutions. `HARNESS_DB_PATH` defaults to `harness/data/harness.sqlite3`; API and workers must share that local database.

Allocation uses a stable ResourceManager idempotency key. Publication intent is saved **before** either submission request. Library and benchmark submission endpoints do not promise idempotency: a lost acknowledgement or interrupted publication becomes `needs_attention` and is never automatically replayed. Inspect the destination before creating a replacement run. Interrupted runs stop and reconcile their sandbox rather than replaying generated commands. Cancellation retains `cleanup_required` until ResourceManager confirms termination; further worker passes reconcile pending cleanup without starting another agent.

## Checks

```sh
uv run pytest
uv run ruff check .
```

Tests exercise actual bounded shell execution in temporary test directories, simulated ResourceManager I/O, model/service contracts, role permissions, publication requirements, complete Library scans, uncertainty, cleanup, persistence and HTTP routes. A live end-to-end run additionally requires a reachable model provider and an enabled ResourceManager pool backed by a working Docker daemon and the installed harness image.

For one-click benchmark research, the You.com MCP broker, Library and Resource API access, and recorded paper evidence, see [One-click paper research](RESEARCH.md).
