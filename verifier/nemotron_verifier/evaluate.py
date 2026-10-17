"""Local harness for trusted remote inference; submitted code is never executed."""

import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, InvalidOperation
from pathlib import Path

import httpx

from . import MODEL, REVISION
from .config import RECIPE, digest, key_headers, suite_identity


def hardware(config, client=None):
    """Probe the active GPU VM through its authenticated SSH-forwarded node service."""
    definition = config["hardware"]
    owned = client is None
    client = client or httpx.Client(timeout=15, trust_env=False, follow_redirects=False)
    try:
        response = client.get(definition["base_url"] + "/hardware", headers=key_headers(definition))
        if response.status_code != 200:
            raise ValueError("An active resource-node GPU VM is required")
        document = response.json()
        gpu = document["gpu"]
        # A node must probe its running guest; cached enrollment smoke tests are insufficient.
        if document.get("fresh") is not True:
            raise ValueError("Resource node must return fresh active-VM hardware metadata")
        if gpu.get("name") != "NVIDIA RTX A5000" or not str(gpu.get("uuid", "")).startswith("GPU-"):
            raise ValueError("This benchmark requires one physical RTX A5000")
        memory = float(gpu["memory_total_mb"])
        power = float(gpu["power_limit_w"])
        driver = str(gpu["driver_version"])
        if not (24000 <= memory <= 25000 and 0 < power <= 300 and driver):
            raise ValueError("RTX A5000 memory, power, or driver metadata is invalid")
        if definition.get("gpu_uuid") and gpu["uuid"] != definition["gpu_uuid"]:
            raise ValueError("Resource-node GPU differs from the registered device")
        return {"name": gpu["name"], "uuid": gpu["uuid"], "memory_mb": memory,
                "power_limit_w": power, "driver": driver}
    finally:
        if owned:
            client.close()


def identity(config, gpu):
    return digest({"suite": suite_identity(config), "gpu": gpu, "model": MODEL,
                   "revision": REVISION, "recipe": RECIPE,
                   "reference": config["deployments"][config["reference"]]})


def numeric_answer(text):
    match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\Z", text.strip())
    if not match:
        return None
    try:
        return Decimal(match.group(1).replace(",", ""))
    except InvalidOperation:
        return None


def ppl_from_logprobs(tokens, logprobs, first):
    if len(logprobs) != len(tokens):
        raise ValueError("Missing prompt-token log probabilities; PPL cannot be measured")
    total, count = 0.0, 0
    for index in range(max(first, 1), len(tokens)):
        value = logprobs[index]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value > 0:
            raise ValueError("Missing or invalid teacher-forced token log probability")
        total -= value
        count += 1
    return total, count


def validate_metrics(metrics, throughput=False):
    for key in ["mmlu", "gsm8k", "ppl"] + (["throughput"] if throughput else []):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Evaluation metrics must be finite numbers")
    if not (0 <= metrics["mmlu"] <= 100 and 0 <= metrics["gsm8k"] <= 100 and metrics["ppl"] >= 1):
        raise ValueError("Evaluation metrics are outside their valid range")
    for prefix, count in [("mmlu", 57), ("gsm8k", 16)]:
        correct = metrics.get(prefix + "_correct")
        if (isinstance(correct, bool) or not isinstance(correct, int) or not 0 <= correct <= count
                or metrics.get(prefix + "_count") != count
                or not math.isclose(metrics[prefix], correct * 100 / count, abs_tol=1e-10)):
            raise ValueError("Mini accuracy must match its exact correct-answer count")
    if throughput and metrics["throughput"] <= 0:
        raise ValueError("Throughput must be positive")


def quality_failures(reference, candidate):
    validate_metrics(reference)
    validate_metrics(candidate)
    failures = []
    if candidate["mmlu_correct"] < reference["mmlu_correct"] - 1:
        failures.append("Mini MMLU added more than one incorrect answer (57 questions)")
    if candidate["gsm8k_correct"] < reference["gsm8k_correct"] - 1:
        failures.append("Mini GSM8K added more than one incorrect answer (16 problems)")
    if candidate["ppl"] > reference["ppl"] * 1.01:
        failures.append("Mini Wikitext PPL rose more than 1% above reference")
    return failures


class Inference:
    def __init__(self, deployment, client=None):
        self.deployment = deployment
        self.client = client or httpx.Client(base_url=deployment["base_url"] + "/",
            headers=key_headers(deployment), timeout=httpx.Timeout(1800, connect=10),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=20),
            trust_env=False, follow_redirects=False)

    def request(self, route, body):
        with self.client.stream("POST", route, json={"model": MODEL, **body}) as response:
            if response.status_code != 200:
                raise ValueError("Inference endpoint rejected evaluation; check engine support")
            chunks, size = [], 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > 8 * 1024 * 1024:
                    raise ValueError("Inference response exceeded the evaluation limit")
                chunks.append(chunk)
        return json.loads(b"".join(chunks))

    def check_model(self):
        response = self.client.get("models", timeout=15)
        if response.status_code != 200 or MODEL not in [row.get("id") for row in response.json().get("data", [])]:
            raise ValueError("The registered engine is not serving the pinned model name")

    def chat(self, prompt, maximum):
        result = self.request("chat/completions", {
            "messages": [{"role": "user", "content": prompt}], "temperature": 0,
            "max_tokens": maximum, "seed": 42,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        choice = result["choices"][0]
        # A truncated reasoning trace never counts as a completed answer.
        return (choice["message"].get("content") or "") if choice.get("finish_reason") == "stop" else ""

    def score(self, tokens):
        result = self.request("completions", {"prompt": tokens, "max_tokens": 0,
            "echo": True, "logprobs": 1, "temperature": 0, "add_special_tokens": False})
        if result.get("usage", {}).get("prompt_tokens") != len(tokens):
            raise ValueError("Teacher-forced prompt token count changed")
        return result["choices"][0]["logprobs"]["token_logprobs"]

    def generate_workload(self, tokens):
        if len(tokens) != 2048:
            raise ValueError("Performance inputs must contain exactly 2048 tokens")
        result = self.request("completions", {"prompt": tokens, "max_tokens": 1024,
            "ignore_eos": True, "temperature": 0, "seed": 42, "add_special_tokens": False})
        usage = result.get("usage", {})
        if usage.get("prompt_tokens") != 2048 or usage.get("completion_tokens") != 1024:
            raise ValueError("Performance requires exactly 2048 input and 1024 output tokens")
        return 1024

    def performance(self, prompts):
        if len(prompts) != 21:
            raise ValueError("Performance requires one warmup and 20 timed prompts")
        # Twenty worker clients form a closed loop. Engines may schedule fewer active
        # sequences, but requests waiting in the engine are included in elapsed time.
        with ThreadPoolExecutor(max_workers=20) as pool:
            for future in as_completed([pool.submit(self.generate_workload, p) for p in prompts[:1]]):
                future.result()
            started = time.perf_counter()
            completed = [future.result() for future in as_completed(
                [pool.submit(self.generate_workload, p) for p in prompts[1:]])]
            elapsed = time.perf_counter() - started
        return {"throughput": sum(completed) / elapsed, "elapsed_seconds": elapsed,
                "output_tokens": sum(completed), "requests": len(completed),
                "concurrency": 20, "input_tokens_per_request": 2048,
                "output_tokens_per_request": 1024, "warmup_requests": 1,
                "measurement": "aggregate_output_tokens_per_wall_second"}


def measure(config, alias, expected_identity, reference=None):
    started = time.perf_counter()
    gpu = hardware(config)
    if identity(config, gpu) != expected_identity:
        raise ValueError("Hardware, reference deployment, or evaluation suite changed")
    deployment = config["deployments"][alias]
    engine = Inference(deployment)
    root = Path(config["suite_dir"])
    mmlu = json.loads((root / "mmlu.json").read_text())
    gsm8k = json.loads((root / "gsm8k.json").read_text())
    tokens = json.loads((root / "ppl_tokens.json").read_text())
    workload = json.loads((root / "workload_tokens.json").read_text())
    if len(mmlu) != 57 or len(gsm8k) != 16 or len(tokens) != 4096:
        raise ValueError("Evaluation requires the fixed prepared mini datasets")
    try:
        engine.check_model()
        # Refuse to substitute generated-token likelihoods or approximate PPL.
        ppl_from_logprobs(tokens[:16], engine.score(tokens[:16]), 1)
        def mmlu_answer(example):
            answer = engine.chat(example["prompt"], 8).strip()
            match = re.fullmatch(r"\(?([ABCD])\)?[.)]?", answer)
            return bool(match and match.group(1) == example["answer"])

        def gsm_answer(example):
            return numeric_answer(engine.chat(example["prompt"], 512)) == Decimal(example["answer"])

        # Four local quality clients reduce API overhead while respecting a small
        # active-sequence budget on a 24 GB card. No sample is skipped or shortened.
        with ThreadPoolExecutor(max_workers=4) as pool:
            mmlu_correct = sum(pool.map(mmlu_answer, mmlu))
            gsm_correct = sum(pool.map(gsm_answer, gsm8k))
        total_nll, scored, previous_end = 0.0, 0, 0
        for start in range(0, len(tokens) - 1, 1024):
            end = min(start + 2048, len(tokens))
            if end <= previous_end:
                break
            window = tokens[start:end]
            loss, count = ppl_from_logprobs(window, engine.score(window), previous_end - start)
            total_nll += loss
            scored += count
            previous_end = end
        if scored != 4095:
            raise ValueError("PPL token coverage was incomplete")
        metrics = {"suite": "mini", "recipe": RECIPE["name"], "mmlu": 100 * mmlu_correct / 57,
                   "mmlu_correct": mmlu_correct, "mmlu_count": 57,
                   "gsm8k": 100 * gsm_correct / 16, "gsm8k_correct": gsm_correct, "gsm8k_count": 16,
                   "ppl": math.exp(total_nll / scored), "identity": expected_identity,
                   "gpu": gpu, "runtime": deployment["runtime"], "execution": "W4A16",
                   "commit": deployment["commit"], "settings": deployment["settings"],
                   "ppl_tokens": scored, "ppl_input_tokens": 4096,
                   "quality_seconds": time.perf_counter() - started,
                   "note": "Mini regression check; not a full MMLU or GSM8K benchmark score."}
        validate_metrics(metrics)
        # Rejected quality candidates never perform warmups or timing and get no score.
        if reference is None or not quality_failures(reference, metrics):
            performance = engine.performance(workload)
            metrics.update({"throughput": performance["throughput"], "performance": performance})
            validate_metrics(metrics, throughput=True)
        if hardware(config) != gpu:
            raise ValueError("GPU configuration changed during evaluation")
        runtime = time.perf_counter() - started
        metrics.update({"runtime_seconds": runtime,
                        "runtime_target_seconds": [600, 1200],
                        "runtime_within_target": runtime <= 1200,
                        "runtime_target_is_advisory": True})
        return metrics
    finally:
        engine.client.close()
