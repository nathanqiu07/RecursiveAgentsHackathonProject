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

## Hosted execution

Hosted storage is independent of local SQLite. D1 records survive Worker restarts. API activity advances durable verification jobs for short background windows; keep polling a submission until it reaches a final state. Long evaluations remain queued in the external verifier and resume score checks on the next dashboard visit or API request. The local Node.js version runs a continuous verification worker.

Hosted verifier endpoints must be reachable over HTTP(S) from Cloudflare. Laptop localhost endpoints work only with the local Node.js dashboard. Public GitHub checkpoint checks use the unauthenticated GitHub API and are subject to its rate limits; unavailable checks fail explicitly before verifier dispatch.

Configure private verifier tokens as hosted environment secrets. Set the chosen `BENCHMARK_VERIFIER_*` bearer-token variable and `BENCHMARK_VERIFIER_ALLOWED_ORIGINS` as a comma-separated list of exact authorized origins. Public benchmark creators cannot redirect configured credentials to another destination. No secrets are included in this source.

## Development and deployment

Requires Node.js 22.13 or later. Install with `npm ci`, then run `npm run dev`. `npx tsc --noEmit` checks the code; `npm run build` builds the Worker and browser assets. The hosting manifest declares D1 as `DB` and no R2 binding.

D1 schema and metadata are under `drizzle/`; apply pending migrations in order. Publishing through Sites applies hosted migrations before the Worker becomes live. Local preview D1 and production D1 are separate. Do not commit environment secrets, local database state, source-write credentials, or build output.
