# Crusoe Cloud compute for KernelBench-mini

One GPU VM on Crusoe Cloud runs our CUDA kernel evaluation (`benchmarks/kernelbench_mini`,
remote mode via `KB_REMOTE=user@host`) and can also run model-written solver programs
(QAPLIB, ALE-Bench) off the laptop. Everything goes through `scripts/crusoe_gpu_vm.sh`.

## Cost

| Short name | VM type | GPU | On-demand |
|---|---|---|---|
| `l40s` (default) | `l40s-48gb.1x` | 1x L40S 48GB PCIe | $1.50/hr |
| `a100` | `a100-80gb.1x` | 1x A100 80GB PCIe | $2.00/hr |
| `h100` | `h100-80gb-sxm-ib.8x` | 8x H100 80GB SXM | $36.00/hr (8 x $4.50) |

Billing is per second while the VM is in the running state. Crusoe documents no 1x H100
VM, so the script refuses `h100` unless `CRUSOE_ALLOW_8X=1`. A 3 hour L40S session is
about $4.50.

## Steps

1. **Install the CLI** (1 min): `brew install crusoecloud/cli/crusoe`
2. **Create Cloud API keys** (2 min): console, Admin, Security, Cloud API keys at
   https://console.crusoecloud.com/security/cloud-api-keys. The secret is shown once.
   These are not the Intelligence (Managed Inference) API keys.
3. **Configure** (1 min): `crusoe config init` asks for the access key ID, secret key and
   default project (https://console.crusoecloud.com/projects). It writes `~/.crusoe/config`.
   Check with `crusoe whoami`.
4. **SSH key** (0 min if present): the script uses `~/.ssh/id_ed25519.pub`; override with
   `CRUSOE_SSH_PUBLIC_KEY_FILE`.
5. **Create the VM** (about 2 to 5 min): `scripts/crusoe_gpu_vm.sh create` (or `create a100`).
   It prints the hourly cost, creates `hackathon-kb-mini` in `us-east1-a` with image
   `ubuntu22.04-nvidia-pcie-docker:latest`, waits for `STATE_RUNNING`, then for SSH.
6. **Set up the toolchain** (about 5 to 10 min): `scripts/crusoe_gpu_vm.sh setup`. It runs
   `benchmarks/kernelbench_mini/setup_gpu_vm.sh` on the VM if that file exists, otherwise
   installs `~/kb-venv` with torch (cu124), ninja and the CUDA 12.4 toolkit for nvcc.
7. **Point the evaluator at it**: copy the `export KB_REMOTE=ubuntu@<ip>` line that
   `setup` prints and run it in your shell.
8. **Work**: `scripts/crusoe_gpu_vm.sh status` shows state, IP and rough cost so far;
   `scripts/crusoe_gpu_vm.sh ssh` opens a shell.
9. **Delete when done: `scripts/crusoe_gpu_vm.sh delete`. This stops billing and destroys
   the VM disk, so copy results back first. Raw CLI: `crusoe compute vms delete hackathon-kb-mini --yes`.**

## Notes

- SSH on port 22 works out of the box: the default VPC has the `default-allow-ssh` rule.
  Right after start, `Connection refused` for a minute or two is normal.
- `crusoe compute vms stop hackathon-kb-mini` pauses compute billing and keeps the disk,
  but the public IP changes on the next start, so `KB_REMOTE` must be re-exported.
- GPU quota is per project (console, Usage, Quotas). Quota does not guarantee capacity;
  if L40S is unavailable in `us-east1-a`, try `CRUSOE_LOCATION=us-southcentral1-a`.
- Unknown from the docs: hackathon credits or quota grants, the driver and CUDA version
  baked into the image, and exact create to SSH time. Treat the minute estimates as guesses.

Sources: docs.crusoecloud.com (installing-the-cli, compute/virtual-machines/overview,
accessing-vms, compute/images/overview, networking/firewall-rules/overview,
usage-billing/viewing-quotas, identity-and-security/managing-api-keys,
reference/cli/crusoe_compute_vms_create and _delete), crusoe.ai/cloud/pricing.
