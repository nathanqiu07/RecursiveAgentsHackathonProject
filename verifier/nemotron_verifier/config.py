"""Operator-owned configuration; submissions never select hosts or credentials."""

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from urllib.parse import urlsplit
from collections import Counter

from . import MODEL, REVISION

RECIPE = {
    "name": "nemotron-a5000-mini-v2", "seed": 42,
    "mmlu_count": 57, "mmlu_per_subject": 1, "mmlu_subject_count": 57,
    "mmlu_shots": 0, "mmlu_max_tokens": 8,
    "gsm8k_count": 16, "gsm8k_shots": 0, "gsm8k_max_tokens": 512,
    "ppl_token_count": 4096, "ppl_window": 2048, "ppl_stride": 1024,
    "workload": {"concurrency": 20, "input_tokens": 2048, "output_tokens": 1024,
                 "warmup_requests": 1, "timed_requests": 20},
    "quality_concurrency": 4, "runtime_target_seconds": [600, 1200],
    "gate": {"mmlu_extra_incorrect": 1, "gsm8k_extra_incorrect": 1, "ppl_ratio": 1.01},
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def loopback_url(value, path="/v1"):
    url = urlsplit(value)
    if (url.scheme != "http" or url.hostname != "127.0.0.1" or not url.port
            or url.username or url.password or url.query or url.fragment or url.path != path):
        raise ValueError("Services must use a local SSH forward at http://127.0.0.1:PORT" + path)
    return value


def key_headers(definition):
    if definition.get("connection_file"):
        # Resource enrollment receipts contain the scoped node key. Read them
        # directly so keys never need to enter a submitted payload or registry.
        descriptor = os.open(Path(definition["connection_file"]).expanduser(), os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "r") as receipt:
            info = os.fstat(receipt.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_size > 65536:
                raise ValueError("Connection receipts must be private regular files")
            key = json.load(receipt).get("key")
        if not isinstance(key, str) or not key or "\n" in key or "\r" in key:
            raise ValueError("Connection receipt has no valid scoped key")
        return {"Authorization": "Bearer " + key}
    name = definition.get("api_key_env")
    if not name:
        return {}
    key = os.environ.get(name)
    if not key:
        raise ValueError("A locally configured service key is missing")
    return {"Authorization": "Bearer " + key}


def load_config(path=None):
    path = Path(path or os.environ.get("NEMOTRON_VERIFIER_CONFIG", "config.json"))
    config = json.loads(path.read_text())
    config["_config_root"] = str(path.resolve().parent)
    if config.get("model") != MODEL or config.get("revision") != REVISION:
        raise ValueError("The verifier requires the pinned Nemotron NVFP4 checkpoint")
    if not config.get("reference") or not isinstance(config.get("deployments"), dict):
        raise ValueError("Configure a reference deployment and a deployment registry")
    if type(config.get("vm_safety_timeout_seconds", 21600)) is not int or not 60 <= config.get("vm_safety_timeout_seconds", 21600) <= 86400:
        raise ValueError("VM safety timeout must be an integer between 60 and 86400 seconds")
    if not config.get("nodes"):
        hardware = config.get("hardware", {})
        loopback_url(hardware.get("base_url", ""))
        if (not hardware.get("connection_file")
                and not re.fullmatch(r"[A-Z_][A-Z0-9_]*", hardware.get("api_key_env", ""))):
            raise ValueError("Configure the trusted resource-node key environment variable")
    for alias, deployment in config["deployments"].items():
        loopback_url(deployment["base_url"])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", alias) or deployment.get("runtime") not in {"vllm", "sglang"}:
            raise ValueError("Deployments must identify their runtime")
        if deployment.get("api_key_env") and not re.fullmatch(r"[A-Z_][A-Z0-9_]*", deployment["api_key_env"]):
            raise ValueError("Service credentials must be environment variable references")
        if (not re.fullmatch(r"[a-fA-F0-9]{40}|[a-fA-F0-9]{64}", deployment.get("commit", ""))
                or not re.fullmatch(r"https://github\.com/[A-Za-z0-9-]+/[A-Za-z0-9_.-]+/?", deployment.get("repository", ""))):
            raise ValueError("Register the exact public source repository and full commit")
        if deployment.get("model_revision") != REVISION or deployment.get("execution") != "W4A16":
            raise ValueError("Register pinned NVFP4 weights with W4A16 execution on Ampere")
        if not isinstance(deployment.get("settings"), dict) or not deployment["settings"]:
            raise ValueError("Register the engine launch settings")
        if "api_key" in deployment or "key" in deployment:
            raise ValueError("Keep service keys in local environment variables")
    if config["reference"] not in config["deployments"]:
        raise ValueError("Reference deployment is not registered")
    suite = Path(config["suite_dir"]).expanduser()
    config["suite_dir"] = str((path.parent / suite).resolve() if not suite.is_absolute() else suite.resolve())
    return config


def suite_identity(config):
    root = Path(config["suite_dir"])
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("model") != MODEL or manifest.get("revision") != REVISION
            or manifest.get("recipe") != RECIPE or manifest.get("suite") != "mini"):
        raise ValueError("Prepare the fixed mini suite for this checkpoint")
    if (manifest.get("mmlu_count") != 57 or manifest.get("gsm8k_count") != 16
            or manifest.get("ppl_token_count") != 4096):
        raise ValueError("Mini suite sample counts do not match the benchmark")
    for name in ["mmlu.json", "gsm8k.json", "ppl_tokens.json", "workload_tokens.json"]:
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != manifest["files"][name]:
            raise ValueError("Evaluation suite checksum mismatch")
    mmlu = json.loads((root / "mmlu.json").read_text())
    gsm = json.loads((root / "gsm8k.json").read_text())
    tokens = json.loads((root / "ppl_tokens.json").read_text())
    workload = json.loads((root / "workload_tokens.json").read_text())
    subjects = Counter(row["subject"] for row in mmlu)
    if (len(mmlu) != 57 or len(subjects) != 57 or set(subjects.values()) != {1}
            or any(row["answer"] not in "ABCD" or len(row["answer"]) != 1 for row in mmlu)
            or len(gsm) != 16 or len(tokens) != 4096 or len(workload) != 21
            or any(len(prompt) != 2048 for prompt in workload)):
        raise ValueError("Prepared mini data does not match the frozen workload shape")
    return digest(manifest)
