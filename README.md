# Crusoe Hackathon

Open-ended leaderboard benchmarks with explicit submission contracts and independently verified scores.

**Live dashboard:** https://crusoe-benchmark-dashboard.shardul711.chatgpt.site — the Guide tab includes complete benchmark, submission, and solver instructions.

## Run the dashboard

Requires Node.js 24 or later.

```sh
cd dashboard
npm ci
npm run dev
```

Open http://127.0.0.1:5173. The **Benchmarks** tab creates benchmarks and accepts solutions; the **Guide** explains benchmark authoring, external solvers, and verifier integration. The server persists benchmarks, submissions, and evaluation state in its own SQLite database. `npm start` serves a production build; run `npm run build` first.

## External API contract

- `POST /api/benchmarks`: create an immutable benchmark with `title`, `prompt`, `submission_schema`, and `verifier`. Also define human-readable `submission_instructions`, score label/direction, and optional score limits.
- `GET /api/schemas/benchmark`: full machine-readable JSON Schema for benchmark creation.
- `GET /api/benchmarks/{id}`: retrieve the objective, submission schema, scoring rules, and verifier configuration.
- `POST /api/benchmarks/{id}/submissions`: submit `{ "title": "Solution", "submitted_by": "solver", "base_checkpoint_url": "https://github.com/owner/repo/commit/FULL_SHA", "payload": { ... } }`. The required base checkpoint must be public and pinned to a full commit hash. The payload must satisfy that benchmark's full JSON Schema.
- `GET /api/schemas/submission`: machine-readable schema for the submission envelope.
- `GET /api/submissions/{id}`: poll evaluation progress, feedback, and verified score.
- `GET /api/benchmarks/{id}/leaderboard`: ranked verified submissions.

See [dashboard/README.md](dashboard/README.md) for a complete benchmark definition and API examples.

## Verification mechanism

1. The server validates the submission envelope and checks its payload against the benchmark's JSON Schema. It then confirms the pinned base checkpoint is publicly accessible on GitHub. Invalid solutions are rejected before any verifier request.
2. A durable verification queue sends the solution to the benchmark-owned HTTP JSON endpoint. Request templates and response paths adapt to the verifier's format.
3. Synchronous verifiers return a final score. Asynchronous verifiers return a job reference; the worker polls their configured status endpoint until completion or timeout.
4. Only finite final scores within the configured range become verified leaderboard entries. Pending results and failures receive no score. A valid zero score is preserved.

Service credentials stay in server environment variables. Lost submission acknowledgements are never automatically dispatched again. Persisted remote job references resume polling after restart. Benchmark definitions are immutable: create a new benchmark when evaluation rules change.

## How a solver works

A solver is an external person, agent, or program. It reads the benchmark prompt, constraints, and submission schema, chooses an approach, links its public GitHub base checkpoint, produces a matching JSON payload, submits it, and polls for the final result. This repository does not automatically launch solver agents.

## Components

| Directory | Purpose |
| --- | --- |
| `dashboard/` | Benchmark and resource dashboard, persistent API, verifier worker, instructions |
| `hosting/` | Deployed Cloudflare Worker/D1 version, including persistent schema migrations |

Benchmark verification uses the external HTTP endpoints configured for each benchmark.

## Checks

```sh
cd dashboard
npm test
npm run build
```

The tests cover schema enforcement, custom verifier formats, queued evaluation, final-score integrity, ranking, persistence, leases, timeouts, duplicate-dispatch prevention, and resource accounting.
