#!/usr/bin/env bash
# Prepare a fresh Ubuntu CUDA VM (e.g. a Crusoe L40S or A100) to score mini-KernelBench kernels.
#
# From the laptop (repo root), one command does everything:
#     bash benchmarks/kernelbench_mini/setup_gpu_vm.sh ubuntu@<vm-ip>
#   1. copies evaluate.py, the static checkers, problems/ and examples/ to ~/kb_mini on the VM   (~10 s)
#   2. runs this script there with --on-vm:
#        python venv at ~/kb_venv, torch with CUDA (wheel matched to the VM's nvcc), ninja      (~3-6 min, a 2.5 GB download)
#        nvidia-smi and nvcc checks (nvcc is required: load_inline compiles with it)             (~5 s; +5 min if apt must install it)
#        smoke test: a known-good ReLU kernel must score correct, a try/except hack must be rejected  (~1 min, first nvcc build)
#   Estimated total on a Crusoe image with drivers + CUDA toolkit: 5-8 min; ~12 min if nvcc has to be installed.
# Then on the laptop:  export KB_REMOTE=ubuntu@<vm-ip>   and run scripts/run_kernelbench_evolve.py --live --one
#
# Nothing here reads or sends API keys; the VM only ever sees kernel source files.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_DIR="kb_mini"            # relative to the VM user's home
VENV="$HOME/kb_venv"
TORCH_VERSION="${KB_TORCH_VERSION:-2.5.1}"

if [[ "${1:-}" != "--on-vm" ]]; then
  TARGET="${1:?usage: setup_gpu_vm.sh user@host   (or --on-vm on the VM itself)}"
  SSH_OPTS=(-o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new ${KB_SSH_OPTS:-})
  start=$(date +%s)
  echo "[laptop] copying evaluator to $TARGET:~/$REMOTE_DIR"
  env -i PATH="$PATH" HOME="$HOME" SSH_AUTH_SOCK="${SSH_AUTH_SOCK:-}" \
    ssh "${SSH_OPTS[@]}" "$TARGET" "mkdir -p $REMOTE_DIR/problems $REMOTE_DIR/examples $REMOTE_DIR/cands"
  env -i PATH="$PATH" HOME="$HOME" SSH_AUTH_SOCK="${SSH_AUTH_SOCK:-}" \
    scp -q "${SSH_OPTS[@]}" "$HERE/evaluate.py" "$HERE/static_checks.py" "$HERE/kb_static_checker.py" \
    "$HERE/setup_gpu_vm.sh" "$TARGET:$REMOTE_DIR/"
  env -i PATH="$PATH" HOME="$HOME" SSH_AUTH_SOCK="${SSH_AUTH_SOCK:-}" \
    scp -q "${SSH_OPTS[@]}" "$HERE"/problems/*.py "$HERE"/problems/LICENSE "$TARGET:$REMOTE_DIR/problems/"
  env -i PATH="$PATH" HOME="$HOME" SSH_AUTH_SOCK="${SSH_AUTH_SOCK:-}" \
    scp -q "${SSH_OPTS[@]}" "$HERE"/examples/*.py "$TARGET:$REMOTE_DIR/examples/"
  env -i PATH="$PATH" HOME="$HOME" SSH_AUTH_SOCK="${SSH_AUTH_SOCK:-}" \
    ssh "${SSH_OPTS[@]}" "$TARGET" "bash $REMOTE_DIR/setup_gpu_vm.sh --on-vm"
  echo "[laptop] done in $(( $(date +%s) - start )) s. Next:"
  echo "  export KB_REMOTE=$TARGET"
  echo "  .venv/bin/python scripts/run_kernelbench_evolve.py --live --one"
  exit 0
fi

# ---------------------------------------------------------------- on the VM
start=$(date +%s)
cd "$HOME/$REMOTE_DIR"
export PATH="/usr/local/cuda/bin:$PATH"

echo "[vm] GPU:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || {
  echo "[vm] ERROR: nvidia-smi failed; this image has no working NVIDIA driver"; exit 1; }

if ! command -v nvcc >/dev/null 2>&1; then
  echo "[vm] nvcc not found; installing the CUDA toolkit from apt (~5 min)"
  sudo apt-get update -y -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nvidia-cuda-toolkit
fi
NVCC_VER="$(nvcc --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1.\2/p')"
echo "[vm] nvcc $NVCC_VER at $(command -v nvcc)"
case "$NVCC_VER" in
  11.*) WHEEL=cu118 ;;
  12.0|12.1|12.2|12.3) WHEEL=cu121 ;;
  *) WHEEL=cu124 ;;
esac

if ! python3 -m venv --help >/dev/null 2>&1 || ! python3 -c "import ensurepip" >/dev/null 2>&1; then
  sudo apt-get update -y -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-dev
fi
[[ -x "$VENV/bin/python" ]] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip wheel setuptools
echo "[vm] installing torch==$TORCH_VERSION ($WHEEL) + ninja"
"$VENV/bin/pip" install -q "torch==$TORCH_VERSION" --index-url "https://download.pytorch.org/whl/$WHEEL"
"$VENV/bin/pip" install -q ninja numpy
export PATH="$VENV/bin:$PATH"   # load_inline needs ninja on PATH (evaluate.py also adds it)

"$VENV/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available(), "torch cannot see the GPU"
print(f"[vm] torch {torch.__version__}, CUDA {torch.version.cuda}, device {torch.cuda.get_device_name(0)}")
PY

echo "[vm] smoke test 1: known-good ReLU kernel (first nvcc build ~40 s)"
OK_JSON="$("$VENV/bin/python" evaluate.py --problem problems/19_ReLU.py --candidate examples/relu_ok.py --trials 20 | tail -1)"
echo "$OK_JSON" | "$VENV/bin/python" -c "import json,sys; r=json.load(sys.stdin); print('   ', {k: r[k] for k in ('compiled','correct','speedup','category')}); print('   ', r['reasons'][:2]); sys.exit(0 if r['correct'] else 1)"

echo "[vm] smoke test 2: try/except fallback must be rejected"
sed 's/        return relu_ext.relu_cuda(x.contiguous())/        try:\n            return relu_ext.relu_cuda(x)\n        except RuntimeError:\n            return torch.relu(x)/' \
  examples/relu_ok.py > cands/smoke_hack.py
HACK_JSON="$("$VENV/bin/python" evaluate.py --problem problems/19_ReLU.py --candidate cands/smoke_hack.py | tail -1)"
echo "$HACK_JSON" | "$VENV/bin/python" -c "import json,sys; r=json.load(sys.stdin); print('   ', r['category']); sys.exit(0 if str(r['category']).startswith('static_reject') else 1)"

echo "[vm] ready in $(( $(date +%s) - start )) s: $VENV/bin/python ~/$REMOTE_DIR/evaluate.py"
