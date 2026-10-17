"""Execute layout programs in a throwaway sandbox: a locked-down Docker container
(no network, read-only root, capped memory/CPU/pids) or, as a fallback, an isolated subprocess."""
import base64
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DOCKER_IMAGE = os.getenv("SANDBOX_IMAGE", "python:3.12-slim")


def docker_available():
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


# --------------------------------------------------------------------------- NVIDIA OpenShell
OPENSHELL_SANDBOX = os.getenv("OPENSHELL_SANDBOX", "donetwin-sbx")
OPENSHELL_IMAGE = os.getenv("OPENSHELL_IMAGE", "docker.io/library/python:3.12-slim")
POLICY_PATH = os.path.join(os.path.dirname(HERE), "openshell", "layout-sandbox-policy.yaml")
RESULT_MARKER = "@@DONETWIN_RESULT@@"
_openshell_cmd = None


def _wsl_path(p):
    p = os.path.abspath(p).replace("\\", "/")
    return f"/mnt/{p[0].lower()}{p[2:]}" if len(p) > 1 and p[1] == ":" else p


def openshell_cmd():
    """CLI prefix for OpenShell: native on Linux/macOS, through WSL on Windows (OPENSHELL_CMD overrides)."""
    global _openshell_cmd
    if _openshell_cmd is not None:
        return _openshell_cmd
    if os.getenv("OPENSHELL_CMD"):
        _openshell_cmd = os.getenv("OPENSHELL_CMD").split()
    elif shutil.which("openshell"):
        _openshell_cmd = ["openshell"]
    elif os.name == "nt":
        distro = os.getenv("OPENSHELL_WSL_DISTRO", "Ubuntu")
        _openshell_cmd = ["wsl", "-d", distro, "--exec", "openshell"]
    else:
        _openshell_cmd = []
    return _openshell_cmd


def _os_path(p):
    return _wsl_path(p) if openshell_cmd()[:1] == ["wsl"] else p


def openshell_available():
    cmd = openshell_cmd()
    if not cmd:
        return False
    try:
        return subprocess.run(cmd + ["status"], capture_output=True, timeout=20).returncode == 0
    except Exception:
        return False


def ensure_openshell_sandbox():
    """Keep one warm, locked-down sandbox; create it on first use."""
    cmd = openshell_cmd()
    if subprocess.run(cmd + ["sandbox", "get", OPENSHELL_SANDBOX], capture_output=True, timeout=30).returncode == 0:
        return
    p = subprocess.run(cmd + ["sandbox", "create", "--name", OPENSHELL_SANDBOX, "--from", OPENSHELL_IMAGE,
                              "--policy", _os_path(POLICY_PATH), "--cpu", "1", "--memory", "512Mi",
                              "--label", "app=donetwin", "--detach", "--", "sleep", "infinity"],
                       capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        raise RuntimeError(f"openshell sandbox create failed: {(p.stderr or p.stdout)[-600:]}")


# Bootstrap run inside the sandbox: unpack the file bundle from stdin into /tmp and run the runner.
_BOOT = ("import sys,json,os,tempfile\n"
         "b=json.load(sys.stdin)\n"
         "d=tempfile.mkdtemp(dir='/tmp')\n"
         "[open(os.path.join(d,n),'w').write(c) for n,c in b.items()]\n"
         "os.execv(sys.executable,[sys.executable,'-I',os.path.join(d,'sandbox_runner.py'),d,'--stdout'])\n")


def _run_openshell(workdir, timeout):
    ensure_openshell_sandbox()
    bundle = {}
    for f in ("scene.py", "sandbox_runner.py", "layout.py", "base_scene.json"):
        with open(os.path.join(workdir, f), encoding="utf-8") as fh:
            bundle[f] = fh.read()
    boot = base64.b64encode(_BOOT.encode()).decode()
    p = subprocess.run(openshell_cmd() + ["sandbox", "exec", "-n", OPENSHELL_SANDBOX, "--no-login-shell", "--",
                                          "python3", "-I", "-c", f"import base64;exec(base64.b64decode('{boot}'))"],
                       input=json.dumps(bundle), capture_output=True, text=True, timeout=timeout)
    for line in reversed(p.stdout.splitlines()):
        if line.startswith(RESULT_MARKER):
            return json.loads(line[len(RESULT_MARKER):])
    return {"ok": False, "error": f"openshell exec failed (rc={p.returncode}): {(p.stderr or p.stdout)[-800:]}",
            "scene": None, "stdout": ""}


def run_layout(code, base_scene, workdir, mode="subprocess", timeout=20):
    os.makedirs(workdir, exist_ok=True)
    for f in ("scene.py", "sandbox_runner.py"):
        shutil.copy(os.path.join(HERE, f), os.path.join(workdir, f))
    with open(os.path.join(workdir, "layout.py"), "w", encoding="utf-8") as f:
        f.write(code)
    with open(os.path.join(workdir, "base_scene.json"), "w", encoding="utf-8") as f:
        json.dump(base_scene, f)
    result_path = os.path.join(workdir, "result.json")
    if os.path.exists(result_path):
        os.remove(result_path)

    if mode == "openshell":
        try:
            res = _run_openshell(workdir, max(timeout, 60))
        except subprocess.TimeoutExpired:
            res = {"ok": False, "error": "layout program exceeded the OpenShell time limit and was killed",
                   "scene": None, "stdout": ""}
        except Exception as e:
            res = {"ok": False, "error": f"OpenShell sandbox error: {e}", "scene": None, "stdout": ""}
        res["mode"] = mode
        return res

    if mode == "docker":
        cmd = ["docker", "run", "--rm", "--network", "none", "--read-only", "--memory", "256m",
               "--cpus", "0.5", "--pids-limit", "64", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
               "-v", f"{os.path.abspath(workdir)}:/work", DOCKER_IMAGE,
               "python", "-I", "/work/sandbox_runner.py", "/work"]
    else:
        cmd = [sys.executable, "-I", os.path.join(workdir, "sandbox_runner.py"), workdir]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"layout program exceeded {timeout}s and was killed", "scene": None,
                "stdout": "", "mode": mode}
    if not os.path.exists(result_path):
        return {"ok": False, "error": f"sandbox crashed: {(p.stderr or p.stdout)[-800:]}", "scene": None,
                "stdout": "", "mode": mode}
    with open(result_path, encoding="utf-8") as f:
        res = json.load(f)
    res["mode"] = mode
    return res
