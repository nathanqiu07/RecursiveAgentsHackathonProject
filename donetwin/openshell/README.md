# NVIDIA OpenShell sandbox for agent-written layout programs

DoneTwin's Layout agent writes Python programs. Those programs run in an **NVIDIA OpenShell** sandbox (the secure
runtime inside NVIDIA NemoClaw) under a strict policy (`layout-sandbox-policy.yaml`):

- **no network egress** (no `network_policies` section),
- **filesystem**: system dirs read-only, only `/tmp` writable,
- **process**: runs as the unprivileged `sandbox` user, 1 CPU, 512 MiB.

One warm sandbox (`donetwin-sbx`, image `python:3.12-slim`) is reused for every iteration. Each layout program is sent as
a JSON file bundle over stdin to a single `openshell sandbox exec`, so there's no upload or download step per iteration.
The result comes back on stdout behind a marker line. See `run_layout(..., mode="openshell")` in `donetwin/sandbox.py`.

## Setup (Windows, WSL 2: experimental per the OpenShell support matrix)

```powershell
# 1. Ubuntu on WSL (asks for a Linux username/password on first launch)
wsl --install -d Ubuntu
# 2. Start Docker Desktop, then Settings -> Resources -> WSL integration -> enable "Ubuntu"
```

```bash
# 3. Inside Ubuntu: install the OpenShell CLI + local gateway (official NVIDIA script)
curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh | sh
openshell status
```

Linux/macOS: skip steps 1–2. Docker Engine 28+ (or Podman 5) is required.

## Use it

```powershell
$env:SANDBOX_MODE="openshell"; .\.venv\Scripts\python.exe server.py   # server auto-selects openshell when `openshell status` works
.\.venv\Scripts\python.exe run.py --sandbox openshell
```

Environment overrides: `OPENSHELL_CMD` (full CLI prefix), `OPENSHELL_WSL_DISTRO` (default `Ubuntu`),
`OPENSHELL_SANDBOX` (default `donetwin-sbx`), `OPENSHELL_IMAGE`.

## Next step: run the whole agent team inside OpenShell

The worker loop (`donetwin/loop.py`) only needs outbound HTTPS to two hosts. A second policy would allow egress solely
to `api.inference.crusoecloud.com:443` (Crusoe Managed Inference / Nemotron) and `ydc-index.io:443` (You.com), with
both API keys held by OpenShell **providers**, so the agents never see the raw credentials.
