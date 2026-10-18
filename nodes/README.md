# Existing A5000 servers → resource nodes

## Explicit Windows/WSL Docker alternative

`docker_host.py` and `docker_setup.py` provide an operator-selected Docker alternative
for a Windows host exposing its GPU through WSL. This is **not KVM/VFIO isolation**.
Its runtime is `docker-wsl2-nvidia`, with `boot_verified: false`,
`container_verified: true`, and a real CUDA kernel probe. VM-only consumers must
continue rejecting this runtime unless deliberately extended by their operator.

Build `nodes/Dockerfile.gpu-node` on the WSL Docker Engine and pin the resulting
immutable image ID in the private Docker target. The generic image includes the
bounded JSONL runner and CUDA smoke test, not inference runtimes or model weights.
Jobs have no network, read-only roots, a non-root user, dropped capabilities,
CPU/RAM/PID limits, bounded writable scratch, and no host/socket/credential mounts.
The service permits one job at a time; GPU access is shared with the Windows host.

Install using `python3 nodes/docker_setup.py --help`. The installer reuses its
private scoped key on identical reruns and verifies real container CUDA execution,
runner self-test, teardown, and wrong-key rejection before writing readiness.
Only the API hash is installed with the service; receipts and SSH credentials stay
outside Git. The node API listens on loopback. SSH should permit forwarding only
to that API port. `server_setup.py connect` can import an explicit Docker receipt
without `--resources-dir`. ResourceManager and the verifier accept explicitly labeled Docker receipts only after checking container, CUDA, runner, cleanup, GPU and pinned-image identity. Inference remains pending until a reference deployment is configured.

This alternative does not assert native NVFP4 support or readiness of the pinned
Nemotron deployments. Those require separate installation and real inference
validation. Do not mark generic CUDA resource readiness as inference readiness.

These tools enroll **existing NVIDIA RTX A5000 servers**, not a fleet of 5,000 servers. They do not purchase cloud capacity. Each server runs a small private resource service; each job gets a fresh **libvirt/QEMU/KVM VM with the dedicated GPU passed through using VFIO**. Agent harness orchestration, benchmark scheduling, quality checks and verifier services run on your local computer. Codex already running on each server can provision it directly using [the ready-to-paste server prompt](SERVER_CODEX_PROMPT.md).

No hosts have been supplied or brought online in this workspace. `examples/inventory.json` is deliberately empty and returns a machine-readable `blocked` result. The templates contain `REPLACE_...` values and are not live capacity.

## Bring three servers online with local Codex agents

Requires local Python 3.10+, OpenSSH, a signed-in Codex CLI, and trusted SSH aliases for your servers. Inspect the pinned guest image and approve its fixed commands before enrolling it.

```sh
mkdir -p "$HOME/.crusoe-nodes"
chmod 700 "$HOME/.crusoe-nodes"
cp nodes/examples/three-nodes.template.json "$HOME/.crusoe-nodes/inventory.json"
chmod 600 "$HOME/.crusoe-nodes/inventory.json"
```

Fill the three records with your real SSH aliases, dedicated GPU PCI functions/UUIDs, and an approved standalone qcow2 image path and SHA256 on each server. Do not place passwords, API keys, or tokens in commands or the inventory. The GPU PCI addresses are host-specific; the example addresses are illustrative. Each physical host may appear only once in this version. Local forwarding ports must be unique.

Check prerequisites without installing packages, writing a service, detaching devices, or booting a VM:

```sh
./nodes/enroll.sh --inventory "$HOME/.crusoe-nodes/inventory.json" --check-only
```

Install the local ResourceManager adapter, then launch one local Codex agent per server in parallel:

```sh
python3 nodes/install_rm_bridge.py
./nodes/enroll.sh --agents \
  --inventory "$HOME/.crusoe-nodes/inventory.json" \
  --parallel 3 \
  --state "$HOME/.crusoe-nodes" \
  --resources-dir "$PWD/Resources"
```

The agents run the deterministic bootstrap tool once. A parent process independently validates the real endpoint, its node/GPU identity, VM boot receipt, and rejection of incorrect keys. Agent prose alone cannot mark a server ready. Omit `--agents` to use the same deterministic enrollment directly. Add `--install-packages` to install the supported Ubuntu/Debian virtualization packages; this never enables IOMMU, rebinds your GPU, or reboots the server. Existing services with a different configuration are rejected.

Codex reuses its configured authentication and model. The CLI invocation uses `exec`, `--ephemeral`, `--output-schema`, and `--output-last-message`, verified against the installed CLI and [official noninteractive Codex documentation](https://developers.openai.com/codex/noninteractive). The provisioning agents use `--sandbox danger-full-access` to run SSH/bootstrap commands; their prompt restricts them to the deterministic approved enrollment command. Keys are not included in that prompt, subprocess arguments, or agent logs.

## Actual output

After a successful boot, live `nvidia-smi`, guest runner shell self-test, and confirmed VM teardown, stdout contains JSON with `status: "ready"` and each node's **actual** `address` and newly generated `key`. The address is a private local HTTP forward over an authenticated SSH tunnel, not a public IP.

```json
{
  "status": "ready",
  "nodes": [{
    "node_id": "your-configured-node-id",
    "status": "ready",
    "address": "http://127.0.0.1:YOUR_CONFIGURED_PORT",
    "key": "rn_GENERATED_ONLY_AFTER_REAL_ENROLLMENT",
    "transport": "ssh-tunnel",
    "verification": {"boot_verified": true, "runner_verified": true}
  }]
}
```

The example above shows the shape only. A blocked/error result has `address: null`, `key: null`, and a concrete error. `--check-only` returns `prerequisites_ready` with no address/key and does not claim a VM has booted. Exit code 0 means every selected node is ready (or prerequisites ready for check-only); exit code 2 means at least one is blocked.

Local credentials and reports live in `$HOME/.crusoe-nodes/<node-id>/connection.json`, `<node-id>/result.json`, and `results.json`, mode **0600** under private directories. Use `--quiet` to suppress credential output. A partially completed enrollment keeps recoverable credentials locally but never emits them as a ready connection. Rerunning with the identical inventory reconnects/reuses the installed node and rechecks readiness. Recreate a different configuration only after stopping its jobs and explicitly removing the old host service/configuration and local receipt; enrollment does not silently rotate a running node's key.

## Required server and guest preparation

The supported host is an existing Ubuntu/Debian Linux x86_64 machine with passwordless `sudo -n` through a trusted SSH alias. Install `qemu-kvm libvirt-daemon-system libvirt-clients cloud-image-utils openssh-client python3`, ensure libvirt is running, and configure these prerequisites:

- CPU virtualization, `/dev/kvm`, and kernel/firmware IOMMU support must be enabled.
- The explicitly selected GPU functions must already use `vfio-pci`. The complete IOMMU group must contain only the dedicated NVIDIA GPU's display/audio/USB functions. Shared/unsafe groups are rejected. Host display GPUs are never detached automatically; arrange a separate host display/management path first.
- Create an isolated libvirt network with DHCP from `examples/isolated-network.xml` (`virsh net-define`, `net-start`, `net-autostart`). It has no forwarding element. Choose an unused private subnet appropriate for your server. The service refuses a network with forwarding enabled. This keeps guest workloads off the internet while allowing the host to reach guest SSH and inference.
- Supply a standalone qcow2 cloud image with **cloud-init, SSH, Python, a working compatible NVIDIA guest driver**, the desired inference runtime and pre-cached model/source/dependencies. Bake `guest_runner.py` into it as executable `/usr/local/bin/gpu-job-runner`. Clean cloud-init state before sealing it. The configured `guest_user` must exist or be creatable by cloud-init. Its default job command must support the provided `--self-test` protocol.
- Pin the exact image bytes with `sha256sum`. The node rechecks that digest at startup and before every job; templates do not download or mutate model weights. The image's virtual disk size must fit `scratch_mb`. The writable qcow2 overlay shares the image's virtual size; `scratch_mb` is a virtual disk allowance, not a host filesystem quota. Provision sufficient host disk space.
- `guest_memory_mb` is **host RAM**, separate from the A5000's 24 GB VRAM. Leave at least 2 GiB of host RAM available outside the VM; choose smaller values on smaller hosts. VM CPUs/RAM/virtual disk and total job timeout are checked against approved pool limits.

For example, bake the runner into an **offline copy** of a prepared image using your approved image-building process or `virt-customize --upload nodes/guest_runner.py:/usr/local/bin/gpu-job-runner --chmod 0755:/usr/local/bin/gpu-job-runner -a APPROVED_COPY.qcow2`, then pin the resulting SHA256. This does not install NVIDIA drivers, inference engines or model weights for you. GPU passthrough/driver/model compatibility must pass the real boot probe; local protocol tests cannot establish it.

[libvirt domain documentation](https://libvirt.org/formatdomain.html) defines the KVM/hostdev boundary; [cloud-init SSH configuration](https://docs.cloud-init.io/en/latest/reference/modules.html#ssh) supplies fresh guest client and host keys. The guest host key is pinned for each VM; guest SSH does not disable host-key checks. PCI hostdev entries use `managed="no"` so libvirt does not detach host drivers for us.

## The same capacity in Experiments and verification

`install_rm_bridge.py` installs tracked adapter source into the local `ResourceManager/` service. The repository includes the service source for collaborators. It also installs `dev/install_resource_status.py` when present. Restart the local ResourceManager API and worker after installation. Existing Docker resources keep their original runtime.

Enrollment with `--resources-dir` writes `Resources/<resource-id>.json` with `runtime: "remote_vm"` and a private `node_manifest` path, and registers only the real verified node with the local ResourceManager at `http://127.0.0.1:8002`. That same connection receipt/key/tunnel can be used by the local verifier pool. There is one authoritative node-side busy lock, so an Experiments job and a verifier job cannot own the same GPU simultaneously. The RM adapter waits on busy nodes within the job's configured timeout. An uncertain VM cleanup retains its resource reservation and blocks another assignment. A lost libvirt create acknowledgement also preserves the disk/state and quarantines the node across restart; an operator must reconcile the recorded VM UUID with libvirt before releasing that capacity.

Keep node registration fresh from the local machine:

```sh
python3 nodes/controller.py heartbeat \
  --state "$HOME/.crusoe-nodes" \
  --resources-dir "$PWD/Resources"
```

This refreshes every 30 seconds only after probing the authenticated real node. Use `RESOURCE_MANAGER_API_KEY` in the local environment, or `--resource-manager-key-file /absolute/private/key.txt` (0600) for an authenticated local manager. Neither key is placed in process arguments. `GET /resources` lists allocatable pools; `/resources/status` includes busy/offline pools and live node state when the discovery extension is installed. The local supervisor can start the heartbeat with `LOCAL_STACK_NODE_STATE` configured.

## Approved inference deployments

Docker/WSL nodes can also launch approved inference aliases. Each alias includes
the same immutable source/model metadata as a VM deployment, plus an internal
Docker network, an environment allowlist, and up to four read-only runtime/model
mounts. Each mount has a SHA256-pinned manifest of its files and symlinks; the node
checks file contents and rejects escaping symlinks before launch. Generic jobs
retain network isolation and non-executable temporary storage. Inference jobs
use executable, size-limited private temporary storage for Triton compilation.

Build `Dockerfile.inference-toolchain` on top of the GPU runner to supply Triton's
C compiler, then pin the resulting image ID in the node target. Updating an idle
node with `docker_setup.py --update-image --update-deployments` preserves its
identity and scoped key. Re-enroll the controller with the new receipt and image
verification before measuring a baseline.

The first working A5000 Nemotron setup uses vLLM source
`f98ae6a9fd2c73a2f80dd9fe4c534676f689394d`, the pinned checkpoint revision
`bee7596271d1495f6992ae224aefde4410e816b8`, Marlin W4A16, BF16 activations and KV
cache, Triton attention, eager execution, context 4096, four active sequences,
GPU memory utilization 0.9, and prefix caching disabled. WSL required the native
CUDA allocator and `checkpoint_page_cache=evict_during_load`: the node advises
away clean checkpoint pages until the engine becomes healthy. It does not change
or delete checkpoint files. Expandable allocations failed on the tested WSL
driver; automatic FP8 KV cache was incompatible with Triton on this A5000.

Real generation and teacher-forced prompt-logprob checks passed on that setup.
These installation checks are not benchmark scores. The controller measures the
first valid mini MMLU/GSM8K/PPL and C20 throughput results as the baseline, with
no minimum reference quality threshold. Record the exact live settings and image
in its private enrollment/configuration; do not reuse a baseline after changing
the hardware, runtime, model, or settings.

The default VM job command is the generic JSONL executor. A verifier can instead choose an operator-approved inference launch command using an alias declared in each target's `deployments` map:

```json
{
  "deployments": {
    "baseline": {
      "command": ["/opt/vllm/bin/vllm", "serve", "/models/approved-model", "--host", "0.0.0.0", "--port", "8000"],
      "metadata": {
        "repository": "https://github.com/YOUR_ORG/vllm",
        "commit": "FULL_IMMUTABLE_SOURCE_COMMIT",
        "model_revision": "FULL_IMMUTABLE_MODEL_REVISION",
        "execution": "W4A16",
        "settings": {"max_model_len": 3072}
      }
    }
  }
}
```

Replace revision placeholders with immutable hex hashes. Source/model files and runtime must already be installed inside the pinned guest image. Metadata is an **operator assertion tied to the approved command/image**, not independent proof that a mutable checkout contains those source bytes. The verifier must compare approved registry metadata with the active guest and fail on mismatches. Do not use Ampere A5000 receipts to claim native NVFP4 execution.

For a manual inference launch:

```sh
python3 nodes/controller.py start-job --node YOUR_NODE_ID \
  --deployment baseline --state "$HOME/.crusoe-nodes"
```

This returns the real job ID in `provisioning` state. Poll job status, `/v1/models`, and fresh `/v1/hardware` before measurement. The inference base URL is the node's local address plus `/v1`, with the same scoped key from the private receipt. Use `stop-job --node ... --job ...` and poll `cleanup_confirmed` before reallocating the GPU. Timeouts are configurable (up to 86,400 seconds), with no fixed ten-minute cap; boot, execution and teardown consume actual wall time.

## Private node protocol

All routes require `Authorization: Bearer <node-scoped-key>`. The key authorizes jobs, live hardware reads and inference on exactly one installed node; it cannot administer the host through this API. Administrative changes remain on the operator SSH path.

| Route | Behavior |
| --- | --- |
| `GET /v1/health` | Node ID, verified readiness, busy flag, limits, last actual boot verification; never a live GPU measurement |
| `POST /v1/probe` | Boot disposable VM, verify GPU+runner, tear down; rejects while busy |
| `POST /v1/jobs` | Body exactly `resource_id,cpus,memory_mb,scratch_mb,timeout_seconds`; required `Idempotency-Key` UUID; optional `X-Deployment-Alias` must be approved |
| `GET /v1/jobs/{id}` | `provisioning/running/completed/failed/cancelled`, exit code and `cleanup_confirmed` |
| `POST /v1/jobs/{id}/stdin` | One `{id,line}` JSONL input; bounded/idempotent per message ID |
| `POST /v1/jobs/{id}/stdin/close` | Close stdin after pending messages drain |
| `GET /v1/jobs/{id}/stdout?after=N` | Bounded `{lines:[{sequence,line}],next_after,closed}` |
| `POST /v1/jobs/{id}/cancel` | Request cancellation; poll status to confirm teardown |
| `GET /v1/hardware` | **Fresh** `nvidia-smi` inside active running VM: `{fresh:true,gpu:{name,uuid,memory_total_mb,power_limit_w,driver_version,vm_id,observed_at},deployment,image_sha256}`; 409 when idle |
| `GET /v1/models`, `POST /v1/completions`, `POST /v1/chat/completions` | Bounded non-streaming proxy to active guest's fixed inference port |

Guest executor input is `{type:"execute",id,command,timeout_seconds}`; output is `{type:"execution_result",id,exit_code,output,truncated}`. Command execution is isolated inside its job VM. The runner is not a Codex agent and does not schedule/evaluate jobs remotely.

## Verification performed here

```sh
ResourceManager/.venv/bin/python -m unittest discover -s nodes/tests -v
ResourceManager/.venv/bin/python -m pytest ResourceManager/tests -q
```

Tests exercise actual local HTTP, scoped auth, one-GPU ownership, deployment/idempotency checks, JSONL exchange through the durable local manager, guest command execution/timeouts, VFIO/IOMMU preflight, VM XML, private credentials and lost-ack cleanup. VM/GPU operations are mocked in protocol tests. No live SSH, KVM/VFIO boot, driver compatibility, model fit or baseline benchmark score is claimed without real hosts.

## Existing GPU VM: direct execution

When the operator explicitly chooses to use an existing GPU VM directly, install
`direct_host.py` with `install_direct.py`. This mode needs a working NVIDIA driver
and systemd, but no nested KVM, VFIO, or guest image. It reports
`direct-host-systemd`, the GPU's actual model/UUID/memory, and `boot_verified:false`.
It must not be advertised as the A5000 benchmark resource when the device is an A4000.

On the server, from this repository:

```sh
sudo python3 nodes/install_direct.py --node-id a4000-01 --port 18100 --cpus 4 --memory-mb 8192
```

Choose CPU/RAM values within the existing VM's capacity. The installer saves a
private scoped connection key in
`/var/lib/crusoe-direct-node/server-connection.json`; it never prints that key.
The API listens only on server loopback. Import the receipt over trusted SSH,
forward its port, test authenticated `/v1/health` plus a real job, then create a
private controller receipt with `status:ready` and `runtime:direct-host-systemd`.
Run `python3 nodes/install_rm_bridge.py` and restart the local ResourceManager
API/worker when safe. Its `remote_vm` transport adapter supports both runtimes;
the explicit direct runtime in the receipt is required to accept direct execution.

Jobs execute as the dedicated unprivileged `resourcejob` user in individual
systemd cgroups with CPU/RAM limits and a private workspace. Systemd kills the
entire job cgroup on cancellation, including child sessions. Host system and home
paths are read-only/hidden; job workspace and private temp space are writable.
One GPU job runs at a time. This is trusted direct execution, not a new VM/security
boundary per job. Scratch capacity is admission metadata, not a disk quota.
The current direct mode supports the JSONL experiment executor; approved inference
deployments for verifier submissions are not yet configured. GPU availability does
not claim model fit or measured benchmark results.

## Persistent controller connection

The controller can supervise a private SSH forward and `direct_heartbeat.py` with launchd. The heartbeat advertises capacity only after authenticated node readiness passes; a disconnected node expires from the scheduler. The Windows/WSL node supervises its outbound reverse connection with systemd. No VPN account or public node API is required. Keep receipt keys, host keys and machine-specific launch configurations outside Git.
