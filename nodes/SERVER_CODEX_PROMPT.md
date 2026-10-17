# Paste this into Codex on each existing GPU server

The source branch is public: [codex/local-a5000-resource-pool](https://github.com/nathanqiu07/RecursiveAgentsHackathonProject/tree/codex/local-a5000-resource-pool). Run the prompt below separately on each server. Provisioning runs directly on that server; it does not depend on the laptop being able to SSH into the server first.

```text
Turn THIS existing NVIDIA RTX A5000 server into an isolated GPU VM resource node for our local benchmark controller. You already run on the server. Complete the work here and return the real connection details and a private scoped node key. Do not purchase cloud resources.

Get the public source from:
  https://github.com/nathanqiu07/RecursiveAgentsHackathonProject
Branch:
  codex/local-a5000-resource-pool

Use a fresh checkout if needed:
  git clone --branch codex/local-a5000-resource-pool --single-branch https://github.com/nathanqiu07/RecursiveAgentsHackathonProject.git
  cd RecursiveAgentsHackathonProject

Preserve any existing checkout changes. Read nodes/README.md, nodes/common.py, nodes/host.py, and nodes/server_setup.py. The supported host is Ubuntu/Debian Linux x86_64 with KVM/libvirt. Use the real scripts and their --help; do not invent commands.

1. Inspect actual hardware and prerequisites:
     python3 nodes/server_setup.py inspect
   Identify the RTX A5000's actual UUID, full PCI/IOMMU group, VRAM, available host RAM/CPUs/disk, virtualization and IOMMU support, GPU processes, SSH service port, and an actual SSH hostname/IP and account for this machine. Choose a unique node ID derived from the hostname plus GPU UUID suffix. Do not assume example PCI addresses or claim a private interface address is reachable from the controller. Export the SSH host public key/fingerprint so the controller can pin it.

2. Prepare the dedicated GPU host and immutable guest image. Install supported host virtualization packages when necessary. Configure an isolated DHCP libvirt network with no forwarding, using nodes/examples/isolated-network.xml only after selecting an unused subnet. The complete dedicated GPU IOMMU group must be bound to vfio-pci before enrollment. Confirm that it is idle and is not the active display or another user's job before changing its driver. Keep a working management path. If firmware settings, shared IOMMU devices, active jobs, or a required reboot prevent safe progress, report the exact blocker; never fabricate readiness.

   Use an existing suitable approved standalone qcow2 image, or build a separate image through a verified Ubuntu/Debian image-building process. The image must contain cloud-init, SSH, Python, a working compatible NVIDIA GPU guest driver, and nodes/guest_runner.py installed as executable /usr/local/bin/gpu-job-runner. Do not modify an image used by another running VM. Install/cache the desired inference runtimes, reviewed source commits, model weights and dependencies before sealing the isolated image; these VMs have no internet forwarding. Clean cloud-init state, seal the standalone image and compute its real SHA256. No secrets or Codex sign-in may be baked into the guest.

   Benchmark target: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4, model revision bee7596271d1495f6992ae224aefde4410e816b8, on one 24 GB RTX A5000. Ampere needs a supported W4A16 execution path for those stored NVFP4 weights; do not claim native FP4 arithmetic. Begin with a small active sequence limit such as four and context sufficient for 2048 input plus 1024 output. The local clients remain C20 even when the engine queues requests. Consult verifier/pool.example.json for the exact reference/candidate aliases and source/settings contract, and validate the actual installed backend. Do not invent unsupported flags or claim model fit from weight-size arithmetic.

3. Create private configuration outside Git, for example $HOME/.crusoe-node-server/target.json, using nodes/examples/a5000.example.json or nodes/examples/three-nodes.template.json as the schema. Fill REAL values, including the unique node/resource IDs, actual GPU group/UUID, approved absolute guest image path and SHA256, host RAM/CPU limits, isolated libvirt network and guest runner. remote_port 18090 is a normal per-server loopback port; choose an unused one. local_port is a controller preference and can later be overridden during import. ssh_host is a future controller SSH alias, not a user@host string.

   Configure approved inference deployment aliases only for commands and metadata that actually exist inside the guest image. Their repository, full commit, model revision, W4A16 execution and settings must exactly match the local verifier registry. Operator-approved metadata is tied to the sealed image/command, not proof of an arbitrary mutable checkout. Keep all credentials out of inventory, launch arguments, model settings and Git. chmod the private target file to 600.

4. Run the local installer directly on THIS server, substituting actual values:
     python3 nodes/server_setup.py install \
       --target "$HOME/.crusoe-node-server/target.json" \
       --node ACTUAL_NODE_ID \
       --ssh-hostname ACTUAL_SSH_HOSTNAME_OR_IP \
       --ssh-user ACTUAL_SSH_ACCOUNT \
       --ssh-port ACTUAL_SSH_PORT \
       --install-packages

   This reuses the deterministic node installer locally through sudo; no laptop SSH is needed for setup. It checks prerequisites, installs a loopback-only systemd node service and performs a real disposable GPU VM boot/runner/GPU smoke test with teardown. It also checks that the API rejects the wrong key. A failed check must remain blocked. It never automatically rebinds a display GPU or reboots the host. Identical reruns preserve the scoped key; do not rotate an active node by deleting its receipt.

5. Independently check the installed service and receipt. The private receipt is normally:
     $HOME/.crusoe-node-server/ACTUAL_NODE_ID/server-connection.json
   It must be mode 0600 and status ready, with boot_verified and runner_verified from a real VM. The service listens only on 127.0.0.1; do not expose its HTTP port publicly. Verify approved inference commands and model support in an actual job when configured. If GPU VM execution works but model/backend support is unfinished, distinguish resource readiness from inference readiness and report the remaining inference blocker honestly. Do not invent MMLU, GSM8K, PPL, throughput or memory-fit numbers.

6. Return a concise private handoff containing:
   - Actual node/resource IDs and source branch commit.
   - Actual SSH hostname/IP, user and SSH port; SSH host public key/fingerprint; whether controller reachability still needs verification.
   - The actual loopback node service port and suggested unique controller forwarding port.
   - Actual GPU name, UUID and VRAM; real VM boot/runner/teardown verification.
   - Path to the private server-connection.json receipt, and the actual scoped key from that receipt through THIS PRIVATE CHAT or another private transfer. Never put the key in GitHub, an issue, public logs or the public repository.
   - Readiness of each approved inference deployment and any concrete blocker.

The controller will privately transfer/import the receipt and establish a strict authenticated SSH forward. All benchmark verifiers, quality/performance harnesses, scheduling and agent orchestration stay on the controller laptop. This server runs only the private node service and per-job GPU VMs/inference. Do not install a remote verifier/harness service or copy your Codex sign-in into a guest.
```

## Import the server handoff on the controller

Transfer `server-connection.json` privately. It contains a scoped secret, so keep mode `0600`; never add it to the repository. Configure the reported SSH hostname/user/port under a trusted local SSH alias and pin the server's reported public host key through the trusted handoff. The importer retains strict host-key checking.

Then, on the controller checkout:

```sh
python3 nodes/install_rm_bridge.py
python3 nodes/server_setup.py connect \
  --receipt /PRIVATE/PATH/server-connection.json \
  --ssh-alias YOUR_TRUSTED_SERVER_ALIAS \
  --local-port 18100 \
  --state "$HOME/.crusoe-nodes" \
  --resources-dir "$PWD/Resources"
```

Use a distinct local port for each server, for example `18100`, `18101`, and `18102`. ResourceManager must already be running locally for `--resources-dir` registration. Omit that option to establish and verify the connection first. The importer checks the actual node identity, GPU boot receipt and key enforcement through the live tunnel before saving the controller's private `<node-id>/connection.json`. Point `verifier/config.json` at those receipts using `verifier/pool.example.json`; public examples contain no real key.

`server_setup.py inspect` is read-only. `install` changes the local Linux server and performs a real VM smoke test only after its existing hardware/image prerequisites pass. CPU-side tests exercise installer/input/privacy and import behavior; they do not establish real GPU readiness.
