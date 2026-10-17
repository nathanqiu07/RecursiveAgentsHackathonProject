"""Runs one agent-written layout program in isolation. Usage: runner.py <workdir>

Reads <workdir>/base_scene.json and <workdir>/layout.py, writes <workdir>/result.json.
Only the Scene API and `math` are reachable from the program.
"""
import builtins
import io
import json
import math
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene import ROBOT_CATALOG, Scene, SceneError  # noqa: E402

SAFE_BUILTINS = {n: getattr(builtins, n) for n in (
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "isinstance", "len", "list",
    "max", "min", "print", "range", "round", "set", "sorted", "str", "sum", "tuple", "zip",
    "ValueError", "Exception", "KeyError")}


def _import(name, *args, **kwargs):
    if name == "math":
        return math
    raise ImportError(f"import of '{name}' is blocked in the layout sandbox (only math is allowed)")


SAFE_BUILTINS["__import__"] = _import


def main(workdir):
    with open(os.path.join(workdir, "base_scene.json"), encoding="utf-8") as f:
        scene = Scene(json.load(f))
    with open(os.path.join(workdir, "layout.py"), encoding="utf-8") as f:
        code = f.read()
    out = io.StringIO()
    g = {"__builtins__": SAFE_BUILTINS, "scene": scene, "math": math,
         "ROBOT_CATALOG": ROBOT_CATALOG, "SceneError": SceneError}
    real_stdout = sys.stdout
    sys.stdout = out
    try:
        exec(compile(code, "layout.py", "exec"), g)
        result = {"ok": True, "error": None, "scene": scene.to_dict()}
    except Exception:
        tb = traceback.format_exc().strip().splitlines()
        result = {"ok": False, "error": "\n".join(tb[-4:]), "scene": None}
    finally:
        sys.stdout = real_stdout
    result["stdout"] = out.getvalue()[-2000:]
    with open(os.path.join(workdir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f)
    if "--stdout" in sys.argv:  # OpenShell mode: hand the result back over the exec stream
        print(RESULT_MARKER + json.dumps(result))


RESULT_MARKER = "@@DONETWIN_RESULT@@"

if __name__ == "__main__":
    main(sys.argv[1])
