# Local control plane

Run the dashboard, every verifier, and every agent harness on your computer. The
A5000 servers supply isolated GPU VMs for jobs and inference. They do not receive
your Codex sign-in, provider credentials, Library database, or local API servers.
Inference connections reach the GPU VMs through local SSH forwards configured by
the node tools. No public tunnel is needed for the local control plane.

The launcher requires Python 3.9+ on macOS or Linux. The components use their own
environments: Python 3.10+ for the core APIs and Python 3.12+ for Nemotron. The
dashboard requires Node.js 24+.

From the repository root:

```sh
# Inspect the complete plan; this writes nothing and runs no services.
python3 dev/local_stack.py up --dry-run

# Make a private, shared local verifier key. Its value is never printed.
python3 dev/local_stack.py init

# Start missing HTTP APIs; this does not start queue workers.
python3 dev/local_stack.py up --no-workers

# Inspect APIs, worker presence, registered capacity and verifier setup.
python3 dev/local_stack.py status
python3 dev/local_stack.py check

# Start APIs plus all local queue workers. Existing queues may begin processing.
python3 dev/local_stack.py up

# Alternatively keep a foreground supervisor until Ctrl-C.
python3 dev/local_stack.py run

# Stop only the processes this launcher created.
python3 dev/local_stack.py down
```

`up` is repeatable. Healthy services already listening on the expected ports are
reused. Existing workers are reused only when their module and working directory
match this checkout. An unknown or unhealthy service occupying a required port
blocks startup of that service. Existing processes are never adopted or stopped.

`check` performs read-only probes and checks installed runtimes; exit status 1
means at least one selected service or worker is unavailable or its dependencies
are missing. Its readiness report separately identifies missing GPU capacity,
model connections, suite configuration, and baseline measurements. A healthy API
or an existing worker process does not prove a GPU VM or benchmark is ready.

## Install component dependencies once

The launcher does not silently install dependencies. Existing environments in
this checkout can be reused. For a new installation:

```sh
uv sync --project ResourceManager --python 3.12
uv sync --project 'verification api' --python 3.12
uv sync --project library --python 3.12
uv sync --project harness --python 3.12
uv sync --project verifier --python 3.12 --extra evaluation
npm --prefix dashboard ci
python3 dev/install_resource_status.py
```

The Library worker uses the harness Python environment plus local source imports,
so its Codex planning adapter has the same dependencies and local sign-in as the
harness. It defaults to `LIBRARY_PLANNER_BACKEND=codex`; Codex chooses its existing
configured model unless `LIBRARY_PLANNER_MODEL` is explicitly supplied. Configure
other model runtimes and profiles in the dashboard's Connections tab. Starting the
worker does not install Claude Code or NemoClaw or establish their model access.

## Local addresses

| Component | Address | Worker |
| --- | --- | --- |
| Dashboard | `http://127.0.0.1:5173` | Vite |
| Dashboard API and benchmark queue | `http://127.0.0.1:3001` | Included in API process |
| Generic verification API | `http://127.0.0.1:8000` | `verification_api.worker` |
| Paper Library | `http://127.0.0.1:8001` | `paper_library.worker` |
| Resource Manager | `http://127.0.0.1:8002` | `resource_manager.worker` |
| Agent Harness | `http://127.0.0.1:8003` | `agent_harness.worker` |
| Nemotron mini verifier | `http://127.0.0.1:9000` | `nemotron_verifier.worker` |

The generic verifier executes trusted repository runner scripts on this local
machine; its registration is named `local-verifier`. This is not a GPU resource
registration. Resource Manager dispatches sandbox jobs to separately enrolled
nodes. The local harness worker drives built-in, Codex, Claude Code and NemoClaw
runs; those runtimes still need their own installed tools and configured profiles.

The Nemotron API starts even before a remote deployment and suite are configured,
and reports `ready: false`. Its worker waits for `verifier/config.json`. Create the
configuration using the verifier instructions, then rerun `up`. No reference score
is invented, and the launcher does not submit a baseline request or download a model.

## Shared private configuration

To supervise the local enrollment heartbeat after bringing servers online, add
`LOCAL_STACK_NODE_STATE=/absolute/path/to/your/node-state` to the private environment
file. Only then does the launcher include `node-heartbeat`, which rechecks enrolled
nodes and refreshes Resource Manager registration every 30 seconds. This process
stays local and does not provision new VMs. Without this setting no heartbeat is
started and no node-state location is assumed.

Install `dev/install_resource_status.py` (also invoked by the nodes bridge
installer) to expose the same pool in the Agents tab's Experiment resources panel.
The panel includes busy, offline and unreachable enrolled nodes. Launch choices
continue to show only allocatable pools. Resource Manager checks the daemon's live
busy state as well as its own held allocations, so a VM held by benchmark
verification is not shown as available for a new experiment. The node daemon's
exclusive VM lock remains the final arbiter if requests race. Restart the local
Resource Manager API and harness API after installing the new discovery routes;
the launcher leaves externally owned processes running until their owner does so.

`init` creates `dev/data/local-stack.env` with permissions `0600`, generating one
random token shared by `NEMOTRON_VERIFIER_API_KEY` and
`BENCHMARK_VERIFIER_NEMOTRON_KEY`. Logs and process state also live under the ignored
`dev/data/` directory, with private permissions. Keep these files local.

Add existing API credentials or explicit planner/model configuration to this
private file as literal `KEY=VALUE` lines. Quotes around a whole value are allowed;
shell expansion and commands are never evaluated. Exported environment variables
take precedence. Individual component `.env` files are not merged by the launcher;
the dashboard's own optional `.env` remains subject to its normal loader, with
the launcher's explicit environment taking precedence.

The launcher keeps API and worker database paths identical and requires absolute
paths for overrides. It maps these consumer keys from their service key whenever
the service key is supplied:

| Service key | Consumer key(s) |
| --- | --- |
| `HARNESS_API_KEY` | `AGENT_HARNESS_API_KEY` |
| `LIBRARY_API_KEY` | `HARNESS_LIBRARY_API_KEY` |
| `DASHBOARD_API_KEY` | `HARNESS_BENCHMARK_API_KEY` |
| `RESOURCE_MANAGER_API_KEY` | `HARNESS_RESOURCE_MANAGER_API_KEY`, `LIBRARY_RESOURCE_MANAGER_API_KEY` |
| `VERIFIER_API_KEY` | `LIBRARY_VERIFICATION_API_KEY` |
| `NEMOTRON_VERIFIER_API_KEY` | `BENCHMARK_VERIFIER_NEMOTRON_KEY` |

Reused processes keep their original environment. If the dashboard API was
already running before key creation, restart it manually with the shared private
configuration, or stop it in its original terminal and let this launcher start
the replacement. The launcher will not stop that external process for you.

Control-plane links are forced to the matching HTTP loopback endpoints. A remote
`HARNESS_LIBRARY_URL`, `HARNESS_BENCHMARK_URL`,
`HARNESS_RESOURCE_MANAGER_URL`, `LIBRARY_VERIFICATION_API_URL`,
`LIBRARY_RESOURCE_MANAGER_URL`, `AGENT_HARNESS_URL`, or
`NEMOTRON_VERIFIER_URL` causes an explicit error. External model-provider endpoints
are independent of these local service links.

Port overrides are `DASHBOARD_API_PORT`, `LOCAL_STACK_WEB_PORT`,
`LOCAL_STACK_LIBRARY_PORT`, `LOCAL_STACK_RESOURCE_MANAGER_PORT`,
`LOCAL_STACK_HARNESS_PORT`, `LOCAL_STACK_VERIFICATION_PORT`, and
`LOCAL_STACK_NEMOTRON_PORT`. Local inter-service links are derived from these
values. `LOCAL_STACK_NODE_BIN` selects Node; component Python overrides follow
`LOCAL_STACK_<DIRECTORY>_PYTHON`, e.g. `LOCAL_STACK_HARNESS_PYTHON` and
`LOCAL_STACK_VERIFICATION_API_PYTHON`.

Use `--env-file /absolute/path/private.env`, `--data-dir /absolute/path/local-state`,
or `--only nemotron-api,nemotron-worker` to select a setup. The default list is
shown by `up --dry-run`. `--timeout 30` allows more startup or graceful shutdown
time. Shutdown sends a graceful signal only to recorded session leaders whose
start identity and random process marker still match. It never searches by port,
kills unrelated processes, or force-kills a worker during cleanup. A worker that
is still stopping remains in the state file for a later `down` or inspection.

## Checks for the launcher

```sh
python3 -m unittest discover -s dev -p 'test_*.py' -v
python3 dev/local_stack.py up --dry-run
```

The tests use temporary local servers and temporary state. They check process
ownership, stale PID handling, repeated startup, safe reuse, port collisions,
private keys, loopback restrictions, and truthful missing-capacity reporting.
They do not submit GPU work or call a model provider.
