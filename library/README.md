# Library

Library independently checks a submitted paper's novelty against the existing Library before allocating experimental compute. Its judge then extracts requirements and experimental conditions and chooses minimum sufficient resources from the separate ResourceManager service. A paper is accepted as verified only when its independent novelty review is `novel` and its experiments are `verified`. Library owns both decisions; ResourceManager supplies generic compute jobs and their input/output channels. Its [HTTP contract](api/resource-manager.openapi.json) is included here; that service and resource inventory are configured independently.

```text
Submitting agent → Library paper queue → Library LLM judge
                                             ↕
                                  ResourceManager GET /resources
                                             ↓
                                 Library verification orchestrator
                                             ↕
                              ResourceManager POST /jobs + channels
                                             ↕
                                  Isolated compute environment
                                  running Library's verification agent
                                             ↕ opaque job I/O
                                    Library verification API bridge
                                             ↕
                                        Verification API
```

ResourceManager does not interpret papers, verification conditions, or verdicts. A compute job's exit code is separate from a scientific result. The compute request contains resource allocations; Library sends the agent's task over the job's opaque input channel.

## Start Library

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```sh
cd library
uv sync --python 3.12
uv run uvicorn paper_library.main:app --host 127.0.0.1 --port 8001
```

API docs: <http://127.0.0.1:8001/docs>. Machine-readable contract: `/openapi.json`. Saved schemas are included for the [Library API](api/openapi.json), [novelty review](api/novelty-review.schema.json), [verification plan](api/verification-plan.schema.json), and [ResourceManager API](api/resource-manager.openapi.json).

In another terminal, configure and start the Library verification worker:

```sh
cd library
export LIBRARY_RESOURCE_MANAGER_URL=http://127.0.0.1:8002
export LIBRARY_PLANNER_BASE_URL=https://your-llm-endpoint.example/v1
export LIBRARY_PLANNER_MODEL=YOUR_MODEL_ID
# Set LIBRARY_PLANNER_API_KEY and LIBRARY_RESOURCE_MANAGER_API_KEY as needed.
export LIBRARY_VERIFICATION_API_URL=http://127.0.0.1:8000
# Set LIBRARY_VERIFICATION_API_KEY if the verification API requires authentication.
uv run python -m paper_library.worker
```

To use the same machine's existing Codex ChatGPT sign-in for Library-owned judging, install the optional sibling harness dependency and select the Codex backend. These are separate Library review calls; the submitting researcher's own novelty assessment is never accepted as verification.

```sh
cd library
# Run codex login separately on this worker if it is not already signed in.
uv run --with-editable ../harness python -m paper_library.worker \
  --planner-backend codex \
  --resource-manager-url http://127.0.0.1:8002
```

`--planner-model` optionally selects an available Codex model. This backend does not require an HTTP planner URL or provider key. The optional adapter uses the harness's native tool-disabled OAuth planner, and never reads or exports its login tokens.

Start a ResourceManager service separately and configure `LIBRARY_RESOURCE_MANAGER_URL` for its endpoint. It manages node registration, the operator-owned resource inventory, compute reservations, and container execution. Library communicates with it through the [ResourceManager HTTP API](api/resource-manager.openapi.json) and does not need Docker access.

## Submit a paper

```sh
curl --fail-with-body http://127.0.0.1:8001/papers \
  -H 'Content-Type: application/json' \
  --data-binary @examples/paper.json
```

Every submission requires `title` and at least one of `paper_url` or `paper_text`. Optional fields are `authors`, `submitted_by`, and `notes`. Requirements come from the paper itself; no separate requirements manifest is required. Replace the illustrative example with the actual paper before submission.

Submissions return HTTP 202, a unique `id`, `status: pending_verification`, and a `Location` header. Repeated submissions create separate entries. URLs are stored as references, not downloaded. Include `paper_text` so the judge can read the paper. File uploads are not implemented.

| Request | Purpose |
| --- | --- |
| `POST /papers` | Submit a paper |
| `GET /papers` | List submissions oldest first |
| `GET /papers?status=pending_verification` | List papers awaiting verification |
| `GET /papers/{id}` | Read the saved plan, compute progress, verification report, or error |
| `POST /papers/{id}/retry` | Requeue an eligible blocked or failed paper |
| `GET /health` | Check the Library API process |

List requests support `limit` (1–100, default 50) and `offset` (default 0), returning `papers`, `total`, `limit`, and `offset`. Unknown IDs return 404 and invalid input returns 422.

## Verification workflow

The independent novelty reviewer compares the complete candidate text with every other stored paper, including pending, blocked and failed submissions. The candidate's own ID is excluded. Unicode/case/whitespace-normalized exact duplicates are rejected without an LLM call. Other overlap receives a separate semantic review of the claimed contribution, mechanism and prior approaches. A new title, paraphrase or parameter change does not by itself establish novelty. Papers and titles are untrusted data; the reviewer cannot invoke tools or follow embedded instructions.

The public `novelty` field records `novel`, `not_novel` or `inconclusive`, a rationale, supporting `source_paper_ids`, the complete snapshot's SHA-256 fingerprint/count, review time and method. Library constructs this provenance itself and validates the reviewer's complete paper-ID coverage. Candidate and Library content are bound to the saved decision. A missing full text, URL-only reference, unreadable record, invalid reviewer output, or catalog exceeding 200 other papers / 100,000 serialized context characters yields `inconclusive` and blocks compute. No paper is silently dropped or summarized to fit the limit. These are novelty findings relative to this Library, not proof of worldwide originality.

Library rechecks its snapshot before allocating compute and before accepting a final verified result. If submissions arrive during review, it repeats the comparison; final acceptance checks the current fingerprint in the same database transaction that saves the verdict. A stale positive review cannot publish. Late overlap blocks acceptance while preserving the experiment in `experimental_report` and reconciling any allocated job. `report` contains only the accepted final outcome, so an experimental success cannot override a failed novelty gate. New blocked novelty decisions have no accepted report.

Library asks ResourceManager for available resources. The judge reads `paper_text` and records claims, procedures, exact conditions, measurable acceptance criteria, and verbatim `source_quotes` in `plan.verification_requirements`. It flags missing or ambiguous details in `plan.missing_information`, without inventing conditions or lowering thresholds. Quotes must occur in the source paper; quote membership establishes provenance rather than proving the interpretation is complete or correct.

The judge selects the minimum estimated sufficient CPU, memory, scratch space, and runtime. Resource capacities are ceilings. An agent that coordinates API calls does not need to duplicate the GPU resources used behind the verification API. Library validates the proposed allocation against ResourceManager's available snapshot and retains the plan with the paper.

Library requests a generic compute job using `POST /jobs` with only `resource_id`, `node_id`, `cpus`, `memory_mb`, `scratch_mb`, and `timeout_seconds`. A stable `Idempotency-Key` identifies the allocation attempt so a request timeout can be retried without creating another compute job. ResourceManager controls the approved image, host mapping, quotas, and fresh on-demand execution environment.

The paper's `compute_job_id` and `compute_job_status` track that allocation. Library sends its verification task through the job's input channel, reads agent output, and handles requests to the verification API itself. ResourceManager transports those opaque UTF-8 lines without applying a paper schema or deciding what they mean. Library records a verification outcome only after validating the agent's report and compute-job completion. A successful compute exit alone does not mark a paper verified.

Paper progress is `pending_verification → planning → verifying → completed`, with `blocked` and `failed` outcomes where appropriate. Reports contain a `verified`, `not_verified`, or `inconclusive` verdict with evidence. The original paper remains unchanged. Retries preserve that paper and begin a fresh assessment; active and completed papers cannot be retried. Unconfirmed remote cleanup must be resolved before a new attempt may start.

Historical `verified` results without an independent novelty record are moved to `blocked` when the updated Library opens its database. Their original report is preserved as `experimental_report`, together with timestamps and compute metadata; no experiment is automatically rerun. After confirming cleanup, use `POST /papers/{id}/retry` on that same ID for a fresh complete assessment. The retry clears the previous assessment and may allocate compute again. Creating a new submission instead would compare against the old record and can correctly trigger duplicate rejection. URL-only records need an operator to supply their full source text before review; the current HTTP API does not edit existing paper text.

## Restricted verification access

The verification agent runs in ResourceManager's isolated compute environment with direct networking disabled. Library mediates the agent's only external capability through its job I/O channel:

- `GET /health` and `GET /openapi.json` on the configured verification API.
- `POST /jobs` to request an inference verification job.
- `GET /jobs/{id}` and `POST /jobs/{id}/cancel` for verification jobs created by that agent run.

Library rejects arbitrary destinations, redirects, worker/admin routes, and access to other runs' jobs. It attaches the verification API credential on the trusted Library side. Planner credentials, ResourceManager credentials, and verification API keys never enter the agent task. ResourceManager's generic job API does not implement these verification-specific permissions.

The compute image must contain a compatible agent runtime and dependencies. The standard-library `paper_library/agent_protocol.py` client defines Library's task, API-request, and result messages. That protocol belongs to Library; ResourceManager's channels carry opaque UTF-8 lines. Any model used within the isolated environment must be available locally or through an explicitly supported verification API capability.

## Configuration

| Variable | Purpose |
| --- | --- |
| `LIBRARY_DB_PATH` | Library SQLite file; default `library/data/library.sqlite3` |
| `LIBRARY_API_KEY` | Optional bearer key for Library paper endpoints |
| `LIBRARY_RESOURCE_MANAGER_URL` | ResourceManager API base URL, for example `http://127.0.0.1:8002` |
| `LIBRARY_RESOURCE_MANAGER_API_KEY` | Credential for generic compute requests and job channels |
| `LIBRARY_PLANNER_BACKEND` | `http` (default) or optional `codex` with the sibling harness installed |
| `LIBRARY_PLANNER_BASE_URL` | Library's compatible LLM endpoint |
| `LIBRARY_PLANNER_MODEL` | Explicit planner model ID |
| `LIBRARY_PLANNER_API_KEY` | Planner provider credential |
| `LIBRARY_VERIFICATION_API_URL` | Inference verification API base URL |
| `LIBRARY_VERIFICATION_API_KEY` | Credential for Library's restricted verification bridge |

`.env.example` lists settings for reference; it is not loaded automatically. Paper contents are sent to the configured planner provider. Library API and worker share one local persistent database. ResourceManager uses its own independent database; do not share SQLite through a network filesystem.

All configured URLs are reached from the Library worker's machine. Use protected HTTPS endpoints between hosts. Keep these credentials on Library's trusted host. Without `LIBRARY_API_KEY`, anyone who can reach Library can submit and read papers. Health and API docs remain public.

Configure image selection, Docker contexts, node registration, and resource quotas through ResourceManager and its separate operator-owned resource inventory. Library selects allocations from the public resource snapshot; private image and execution-host settings stay inside ResourceManager.

## Deployment and checks

See [the Crusoe deployment notes](deploy/crusoe.md) for service placement.

```sh
uv run pytest
uv run ruff check .
```

Library tests cover paper intake, complete independent novelty comparisons, semantic and exact duplicates, context/source checks, concurrent Library changes, requirement extraction, resource selection, restricted verification access, cleanup and outcomes. Cross-service integration tests skip when a sibling ResourceManager checkout is unavailable.
