#!/usr/bin/env bash
# Manage one Crusoe Cloud GPU VM for KernelBench-mini kernels and solver runs.
#
# Usage: scripts/crusoe_gpu_vm.sh <command> [args]
#   create [l40s|a100|h100|<raw-type>]  create the VM (default l40s) and wait for SSH
#   status                              state, type, IP, rough cost so far
#   ip                                  print the public IP only
#   ssh [cmd...]                        ssh in (or run a command) as ubuntu@<ip>
#   setup                               install the GPU toolchain on the VM, print KB_REMOTE
#   delete                              delete the VM (stops billing, destroys its disk)
#
# Credentials are never read or stored here. The Crusoe CLI finds them itself in
# ~/.crusoe/config (written by `crusoe config init`) or in CRUSOE_ACCESS_KEY_ID /
# CRUSOE_SECRET_KEY / CRUSOE_DEFAULT_PROJECT.
#
# Env overrides:
#   CRUSOE_VM_NAME              default hackathon-kb-mini
#   CRUSOE_LOCATION             default us-east1-a (L40S also in us-southcentral1-a)
#   CRUSOE_SSH_PUBLIC_KEY_FILE  default ~/.ssh/id_ed25519.pub (same var the CLI reads)
#   CRUSOE_IMAGE                default ubuntu22.04-nvidia-pcie-docker:latest (sxm for h100)
#   CRUSOE_ALLOW_8X=1           required for h100: Crusoe only lists an 8x H100 VM
#   TORCH_INDEX_URL             fallback setup only, default cu124 wheels
#   CUDA_TOOLKIT_PKG            fallback setup only, default cuda-toolkit-12-4
#
# Sources: docs.crusoecloud.com (installing-the-cli, compute/virtual-machines/overview,
# compute/images/overview, reference/cli/crusoe_compute_vms_*), crusoe.ai/cloud/pricing.
set -euo pipefail

VM_NAME="${CRUSOE_VM_NAME:-hackathon-kb-mini}"
LOCATION="${CRUSOE_LOCATION:-us-east1-a}"
SSH_PUB_KEY="${CRUSOE_SSH_PUBLIC_KEY_FILE:-$HOME/.ssh/id_ed25519.pub}"
SSH_USER="ubuntu"   # default user on all Crusoe ubuntu images
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KB_DIR="$REPO_ROOT/benchmarks/kernelbench_mini"
KB_SETUP="$KB_DIR/setup_gpu_vm.sh"
SELF="scripts/crusoe_gpu_vm.sh"

die() { echo "error: $*" >&2; exit 1; }
log() { echo "[crusoe] $*" >&2; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o ServerAliveInterval=30)
if [[ -f "${SSH_PUB_KEY%.pub}" ]]; then
  SSH_OPTS+=(-i "${SSH_PUB_KEY%.pub}")
fi

# ---------------------------------------------------------------- preflight

require_cli() {
  command -v crusoe >/dev/null 2>&1 || die "crusoe CLI not found.
  Install (macOS):  brew install crusoecloud/cli/crusoe
  Then configure:   crusoe config init
  Keys come from https://console.crusoecloud.com/security/cloud-api-keys
  (Cloud API keys, NOT the Intelligence / Managed Inference API key)."
  command -v python3 >/dev/null 2>&1 || die "python3 is required to parse CLI JSON output."
  if ! crusoe whoami >/dev/null 2>&1; then
    die "crusoe CLI is installed but not configured or the keys are rejected.
  Run: crusoe config init   (asks for access key ID, secret key, default project)
  Keys:     https://console.crusoecloud.com/security/cloud-api-keys
  Projects: https://console.crusoecloud.com/projects
  Check with: crusoe whoami"
  fi
}

# ---------------------------------------------------------------- type table

# Sets TYPE, IMAGE, RATE (USD per hour, on-demand) from a short name or raw type.
resolve_type() {
  local want="${1:-l40s}"
  case "$want" in
    l40s|L40S|l40s-48gb.1x)
      TYPE="l40s-48gb.1x"; IMAGE="ubuntu22.04-nvidia-pcie-docker:latest"; RATE="1.50" ;;
    a100|A100|a100-80gb.1x)
      TYPE="a100-80gb.1x"; IMAGE="ubuntu22.04-nvidia-pcie-docker:latest"; RATE="2.00" ;;
    h100|H100|h100-80gb-sxm-ib.8x)
      [[ "${CRUSOE_ALLOW_8X:-0}" == "1" ]] || die "Crusoe documents no 1x H100 VM; the only H100 type is
  h100-80gb-sxm-ib.8x at 8 x \$4.50 = \$36.00/hr. Use l40s (\$1.50/hr) or a100 (\$2.00/hr),
  or set CRUSOE_ALLOW_8X=1 if you really want the 8-GPU box."
      TYPE="h100-80gb-sxm-ib.8x"; IMAGE="ubuntu22.04-nvidia-sxm-docker:latest"; RATE="36.00" ;;
    *.*)
      TYPE="$want"; IMAGE="ubuntu22.04-nvidia-pcie-docker:latest"; RATE="$(rate_for_type "$want")" ;;
    *)
      die "unknown type '$want'. Use l40s, a100, h100, or a raw type from: crusoe compute vms types" ;;
  esac
  IMAGE="${CRUSOE_IMAGE:-$IMAGE}"
}

rate_for_type() {
  case "$1" in
    l40s-48gb.1x) echo "1.50" ;;
    a100-80gb.1x) echo "2.00" ;;
    h100-80gb-sxm-ib.8x) echo "36.00" ;;
    *) echo "unknown" ;;
  esac
}

# ---------------------------------------------------------------- JSON helpers

# Reads `crusoe compute vms get --json` on stdin and prints one field.
# Field "ip" walks network_interfaces[].ips[].public_ipv4.address (API schema
# InstanceV1Alpha5); the walk is recursive so a wrapper object does not break it.
read -r -d '' PY_FIELD <<'PY' || true
import json, sys
want = sys.argv[1]
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)
def walk(o):
    if isinstance(o, dict):
        yield o
        for v in o.values():
            yield from walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from walk(v)
def norm(k):
    return k.lower().replace("_", "")
if want == "ip":
    for o in walk(data):
        for k, v in o.items():
            if norm(k) in ("publicipv4", "publicip"):
                if isinstance(v, dict) and v.get("address"):
                    print(v["address"]); sys.exit(0)
                if isinstance(v, str) and v:
                    print(v); sys.exit(0)
    sys.exit(0)
inst = next((o for o in walk(data) if "state" in o and ("name" in o or "type" in o)), None)
if inst is None:
    sys.exit(0)
if want == "age_hours":
    from datetime import datetime, timezone
    raw = str(inst.get("created_at", ""))
    try:
        raw = raw.replace("Z", "+00:00")
        if "." in raw:
            head, tail = raw.split(".", 1)
            frac, sep, tz = tail.partition("+")
            raw = head + "." + frac[:6].ljust(6, "0") + sep + tz
        t = datetime.fromisoformat(raw)
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        print("%.3f" % ((datetime.now(timezone.utc) - t).total_seconds() / 3600))
    except Exception:
        pass
    sys.exit(0)
print(inst.get(want, ""))
PY

vm_json() { crusoe compute vms get "$VM_NAME" --json 2>/dev/null; }
vm_exists() { crusoe compute vms get "$VM_NAME" >/dev/null 2>&1; }
field() { vm_json | python3 -c "$PY_FIELD" "$1"; }

get_ip() {
  local ip tries="${1:-1}"
  while :; do
    ip="$(field ip || true)"
    [[ -n "$ip" ]] && { echo "$ip"; return 0; }
    tries=$((tries - 1))
    (( tries > 0 )) || die "no public IP found for $VM_NAME. Is it running? Try: $SELF status"
    sleep 10
  done
}

# ---------------------------------------------------------------- waiting

wait_running() {
  local deadline=$((SECONDS + 900)) started=0 state
  while :; do
    state="$(field state || true)"
    case "$state" in
      STATE_RUNNING) log "state: STATE_RUNNING after $((SECONDS - T0))s"; return 0 ;;
      STATE_SHUTOFF|STATE_SHUTDOWN)
        # vms create docs: "Call crusoe compute vms start on the created VM before accessing it."
        if [[ $started -eq 0 ]]; then
          log "VM is $state, starting it"
          crusoe compute vms start "$VM_NAME" >/dev/null
          started=1
        fi ;;
      STATE_CRASHED) die "VM crashed. Inspect with: crusoe compute vms get $VM_NAME" ;;
    esac
    (( SECONDS < deadline )) || die "timed out after 15 min waiting for STATE_RUNNING (last: ${state:-unknown})"
    log "state: ${state:-pending} ($((SECONDS - T0))s)"
    sleep 10
  done
}

wait_ssh() {
  local ip="$1" deadline=$((SECONDS + 600))
  until ssh "${SSH_OPTS[@]}" -o BatchMode=yes -o ConnectTimeout=5 "$SSH_USER@$ip" true >/dev/null 2>&1; do
    (( SECONDS < deadline )) || die "SSH to $SSH_USER@$ip not ready after 10 min. Check the key and the default-allow-ssh firewall rule."
    log "waiting for sshd on $ip ($((SECONDS - T0))s)"
    sleep 10
  done
  log "SSH ready after $((SECONDS - T0))s"
}

# ---------------------------------------------------------------- commands

cmd_create() {
  require_cli
  resolve_type "${1:-l40s}"
  [[ -f "$SSH_PUB_KEY" ]] || die "SSH public key not found at $SSH_PUB_KEY.
  Create one with ssh-keygen -t ed25519, or set CRUSOE_SSH_PUBLIC_KEY_FILE."
  if vm_exists; then
    log "$VM_NAME already exists; not creating a second one."
    cmd_status
    return 0
  fi
  if [[ "$RATE" == "unknown" ]]; then
    echo "cost: $TYPE on-demand rate not in this script's table; check https://crusoe.ai/cloud/pricing"
  else
    echo "cost: $TYPE on-demand \$$RATE/hr (\$$(python3 -c "print('%.3f' % ($RATE/60))")/min), billed per second while running. Stop billing with: $SELF delete"
  fi
  log "creating $VM_NAME type=$TYPE location=$LOCATION image=$IMAGE (expect ~2-5 min to SSH)"
  T0=$SECONDS
  crusoe compute vms create \
    --name "$VM_NAME" \
    --type "$TYPE" \
    --location "$LOCATION" \
    --image "$IMAGE" \
    --keyfile "$SSH_PUB_KEY"
  wait_running
  local ip
  ip="$(get_ip 12)"   # allow up to 2 min for the public IP to appear
  wait_ssh "$ip"
  echo "VM $VM_NAME is up at $ip"
  echo "next: $SELF setup"
}

cmd_status() {
  require_cli
  if ! vm_exists; then
    echo "no VM named $VM_NAME (nothing is billing under this name)"
    return 0
  fi
  local json state type loc ip age rate
  json="$(vm_json)"
  state="$(python3 -c "$PY_FIELD" state <<<"$json")"
  type="$(python3 -c "$PY_FIELD" type <<<"$json")"
  loc="$(python3 -c "$PY_FIELD" location <<<"$json")"
  ip="$(python3 -c "$PY_FIELD" ip <<<"$json")"
  age="$(python3 -c "$PY_FIELD" age_hours <<<"$json")"
  rate="$(rate_for_type "$type")"
  echo "name:     $VM_NAME"
  echo "state:    ${state:-unknown}"
  echo "type:     ${type:-unknown}   location: ${loc:-unknown}"
  echo "ip:       ${ip:-none}"
  if [[ -n "$age" && "$rate" != "unknown" ]]; then
    echo "age:      ${age} h  (upper-bound cost if never stopped: \$$(python3 -c "print('%.2f' % ($age*$rate))") at \$$rate/hr)"
  fi
}

cmd_ip() {
  require_cli
  get_ip
}

cmd_ssh() {
  require_cli
  local ip
  ip="$(get_ip)"
  exec ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" "$@"
}

cmd_setup() {
  require_cli
  local ip
  ip="$(get_ip)"
  T0=$SECONDS
  wait_ssh "$ip"
  if [[ -f "$KB_SETUP" ]]; then
    log "copying $KB_DIR to ~/kernelbench_mini on the VM"
    ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" 'rm -rf ~/kernelbench_mini'
    scp "${SSH_OPTS[@]}" -q -r "$KB_DIR" "$SSH_USER@$ip:kernelbench_mini"
    log "running setup_gpu_vm.sh on the VM"
    ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" 'cd ~/kernelbench_mini && bash setup_gpu_vm.sh'
  else
    log "$KB_SETUP not found; running fallback: venv ~/kb-venv with torch+CUDA and ninja"
    ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" \
      "TORCH_INDEX_URL='${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}' CUDA_TOOLKIT_PKG='${CUDA_TOOLKIT_PKG:-cuda-toolkit-12-4}' bash -s" <<'REMOTE'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
APT="sudo -E apt-get -o DPkg::Lock::Timeout=600 -y -q"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || { echo "nvidia-smi failed: wrong image?" >&2; exit 1; }
$APT update
$APT install python3-venv python3-dev build-essential wget
# torch.utils.cpp_extension.load_inline needs nvcc; the image docs list drivers only.
if ! command -v nvcc >/dev/null 2>&1 && [ ! -x /usr/local/cuda/bin/nvcc ]; then
  . /etc/os-release
  wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION_ID//./}/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb
  sudo dpkg -i /tmp/cuda-keyring.deb
  $APT update
  $APT install "$CUDA_TOOLKIT_PKG"
fi
python3 -m venv ~/kb-venv
~/kb-venv/bin/pip install -q --upgrade pip wheel
~/kb-venv/bin/pip install -q torch --index-url "$TORCH_INDEX_URL"
~/kb-venv/bin/pip install -q ninja numpy
grep -q 'kb-venv' ~/.profile || cat >> ~/.profile <<'EOF'
export CUDA_HOME=/usr/local/cuda
export PATH="$HOME/kb-venv/bin:/usr/local/cuda/bin:$PATH"
EOF
/usr/local/cuda/bin/nvcc --version | tail -n 1
~/kb-venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
REMOTE
  fi
  echo
  echo "setup done in $((SECONDS - T0))s. For remote kernel evaluation run:"
  echo "export KB_REMOTE=$SSH_USER@$ip"
}

cmd_delete() {
  require_cli
  if ! vm_exists; then
    echo "no VM named $VM_NAME; nothing to delete"
    return 0
  fi
  local ip
  ip="$(field ip || true)"
  log "deleting $VM_NAME (its disk and any results on it are destroyed)"
  crusoe compute vms delete "$VM_NAME" --yes
  # Dynamic IPs get recycled; drop the old host key so a future VM on this IP is not refused.
  if [[ -n "$ip" ]]; then ssh-keygen -R "$ip" >/dev/null 2>&1 || true; fi
  local deadline=$((SECONDS + 180))
  while vm_exists; do
    (( SECONDS < deadline )) || die "delete issued but $VM_NAME still listed after 3 min. Check: crusoe compute vms list"
    sleep 5
  done
  echo "deleted $VM_NAME; on-demand billing for it has stopped"
}

usage() { sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

T0=$SECONDS
cmd="${1:-}"
[[ $# -gt 0 ]] && shift
case "$cmd" in
  create) cmd_create "$@" ;;
  status) cmd_status ;;
  ip)     cmd_ip ;;
  ssh)    cmd_ssh "$@" ;;
  setup)  cmd_setup ;;
  delete) cmd_delete ;;
  -h|--help|help|"") usage ;;
  *) usage; die "unknown command: $cmd" ;;
esac
