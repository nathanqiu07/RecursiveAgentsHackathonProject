# Work on this checkout

This repository includes the dashboard and all local control-plane services. Each developer runs their own API servers, workers, private configuration, and databases. GPU inference runs in enrolled VM resource nodes; verifiers and agent harnesses run locally.

Use Node.js 24+, Python 3.12+, and `uv`. From the repository root, install each component's locked dependencies:

```sh
npm --prefix dashboard ci
uv sync --project ResourceManager --python 3.12
uv sync --project 'verification api' --python 3.12
uv sync --project library --python 3.12
uv sync --project harness --python 3.12
uv sync --project verifier --python 3.12 --extra evaluation --group dev
```

Inspect and start the local stack:

```sh
python3 dev/local_stack.py up --dry-run
python3 dev/local_stack.py init
python3 dev/local_stack.py up
python3 dev/local_stack.py status
```

Open the dashboard at `http://127.0.0.1:5173`. The stack can start while GPU nodes are disconnected; readiness and measured baselines are reported separately. `init` creates a private shared local service key. It never prints that key.

Inside a benchmark, use **Overview** to read the objective, **Verifier instructions** to set up the verifier and measure the reference, and **Agents** to launch benchmark-specific research/attempt runs. Submit a pinned implementation under **Submissions**, where its verification feedback appears. Only verified results appear on **Leaderboard**. For Nemotron, prepare the real GPU pool and baseline before submitting candidates; the dashboard does not invent a reference score.

Read [the local stack guide](dev/README.md) for addresses, environment configuration, startup ownership, and troubleshooting. Read [the mini verifier guide](verifier/README.md) before preparing datasets or measuring a baseline. Resource node provisioning scripts are in `nodes/`; their examples require operator configuration for the actual Linux server, IOMMU group, pinned VM image, SSH alias, and approved inference command.

If Codex is already running on each server, paste [the server Codex prompt](nodes/SERVER_CODEX_PROMPT.md) there. Its installer runs directly on the server and exports a private connection receipt. Import that receipt on the controller with `nodes/server_setup.py connect`; laptop SSH is needed for the eventual authenticated inference connection, not for the server's local setup.

## Source layout

| Directory | Responsibility |
| --- | --- |
| `dashboard/` | Local dashboard, benchmark API, verifier integration |
| `hosting/` | Separately hosted dashboard variant |
| `dev/` | Local process launcher and readiness checks |
| `ResourceManager/` | Registered compute capacity and job execution |
| `Resources/` | Disabled example compute inventory and format documentation |
| `verification api/` | Generic verification API and job worker |
| `library/` | Paper intake, verification planning, and reports |
| `harness/` | Local researcher and attempt agent orchestration |
| `nodes/` | Existing-server GPU VM enrollment, job lifecycle, and local SSH transport |
| `verifier/` | Nemotron mini quality and throughput evaluation |

Generated environments, datasets, model weights, VM disks, databases, process state, `.env` files, and node connection receipts stay local. Source and disabled examples are shared. Use `verifier/config.example.json` as a starting point; `verifier/config.json` is private and ignored. Keep real operator node inventory in ignored local configuration or `data/`, and never paste a scoped node key into a submission or committed JSON file.

Before sharing changes, run the relevant component tests and builds. Existing component READMEs contain their commands. No GPU benchmark number or memory-fit result is established merely by passing CPU-side tests.

Create a personal branch for changes, avoid committing runtime files, and push that branch for collaborators to review. The hosted application requires its separate hosting workflow; pushing source does not deploy or publish a site.
