# Hosted benchmark dashboard

Published dashboard: https://crusoe-benchmark-dashboard.shardul711.chatgpt.site

This directory contains the hosted implementation: dashboard, API routes, durable D1 storage, JSON Schema validation, and GitHub checkpoint checks. The separate `dashboard/` package runs the local Node.js/SQLite version.

## Benchmark and solver instructions

Open the dashboard's **Guide** tab or read [the API guide](../dashboard/README.md).

- `GET /api/schemas/benchmark` describes the external benchmark-creation request.
- `POST /api/benchmarks` creates an immutable benchmark with an objective, complete submission schema, scoring rules, and verifier configuration.
- `GET /api/schemas/submission` describes the solution envelope.
- `POST /api/benchmarks/{id}/submissions` requires a title, public GitHub `base_checkpoint_url` pinned to a full commit SHA, and a payload matching the selected benchmark's schema. `submitted_by` identifies the solver.
- The server confirms public checkpoint access anonymously, then calls the benchmark verifier. HTTP JSON request templates and response paths support synchronous scores or asynchronous job polling.
- `GET /api/submissions/{id}` returns verification progress. External solvers poll until completion; the dashboard also refreshes automatically.
- `GET /api/benchmarks/{id}/leaderboard` returns verified scores only.

The verifier performs the actual evaluation. This application records its scores and feedback; it does not install or launch an autonomous solver.

An empty Benchmarks tab offers **Benchmark 01: Optimize Nemotron Lightning** on one RTX A5000 24 GB using the released NVFP4 checkpoint. Its setup form includes the objective, pinned model revision, submission contract, output throughput score, and MMLU / WikiText-2 PPL / GSM8K quality checks. Published BF16 base-model references are labeled separately from measured A5000 scores; PPL and the verifier's own baseline remain pending. Connect a reachable independent verifier before creating the immutable benchmark. The verifier must measure the reference and enforce all quality checks before returning ranked scores.

## Agent controls

The **Agents** tab can launch and monitor the separate [Agent Harness](../harness/README.md). It shows worker availability, run history, novelty reviews, recorded experiment output, cancellation, submitted papers, and benchmark verification results. The harness, model worker and ResourceManager execution service run on a separate compute host; Cloudflare does not execute these sandboxes.

Configure the hosted server's `AGENT_HARNESS_URL` with a reachable HTTPS origin and `AGENT_HARNESS_API_KEY` with the harness's bearer key when required. On a public Site, also configure `AGENT_HARNESS_ALLOWED_EMAILS` or `AGENT_HARNESS_ALLOWED_USER_IDS` as comma-separated operator allowlists. Agent routes require trusted ChatGPT sign-in identity and membership in one of those lists. No allowlist means no compute access. Keep these settings in hosted runtime secrets; neither service keys nor endpoint details enter the browser. A local harness at laptop localhost is available through the local dashboard, not through the hosted Worker.

An unconfigured hosted connection displays a setup state with disabled launch buttons. The local dashboard defaults to a harness on `http://127.0.0.1:8003`; see [local agent setup](../dashboard/README.md#agents).

The source also includes a **Connections** tab for harness availability, multiple named model profiles, Crusoe Foundry inference, optional You.com Search, and read-only readiness checks. Codex defaults to the existing sign-in on the worker; OAuth credentials are not copied to the hosted frontend. API keys are stored encrypted by the separate harness backend and are never returned in settings responses. Claude Code and NemoClaw require their external runtimes and saved model profiles; the UI reports missing setup instead of claiming a live provider is ready.

For local use, start `dashboard/` and the local harness, then enter credentials through Connections. These source changes do not publish a new Site or expose laptop localhost to Cloudflare. Hosted settings controls require the reachable HTTPS harness and operator allowlists described above. A settings save does not verify a live agent run; explicit connection tests make small provider requests and may incur usage charges. See [Connections usage](../dashboard/README.md#connections).

## Hosted execution

Hosted storage is independent of local SQLite. D1 records survive Worker restarts. API activity advances durable verification jobs for short background windows; keep polling a submission until it reaches a final state. Long evaluations remain queued in the external verifier and resume score checks on the next dashboard visit or API request. The local Node.js version runs a continuous verification worker.

Hosted verifier endpoints must be reachable over HTTP(S) from Cloudflare. Laptop localhost endpoints work only with the local Node.js dashboard. Public GitHub checkpoint checks use the unauthenticated GitHub API and are subject to its rate limits; unavailable checks fail explicitly before verifier dispatch.

Configure private verifier tokens as hosted environment secrets. Set the chosen `BENCHMARK_VERIFIER_*` bearer-token variable and `BENCHMARK_VERIFIER_ALLOWED_ORIGINS` as a comma-separated list of exact authorized origins. Public benchmark creators cannot redirect configured credentials to another destination. No secrets are included in this source.

## Development and deployment

Requires Node.js 22.13 or later. Install with `npm ci`, then run `npm run dev`. `npx tsc --noEmit` checks the code; `npm run build` builds the Worker and browser assets. The hosting manifest declares D1 as `DB` and no R2 binding.

D1 schema and metadata are under `drizzle/`; apply pending migrations in order. Publishing through Sites applies hosted migrations before the Worker becomes live. Local preview D1 and production D1 are separate. Do not commit environment secrets, local database state, source-write credentials, or build output.
