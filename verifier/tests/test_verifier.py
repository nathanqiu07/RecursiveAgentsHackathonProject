import json
import math
import threading
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from nemotron_verifier import GPU, MODEL, REVISION
from nemotron_verifier.config import RECIPE, digest, key_headers, load_config, suite_identity
from nemotron_verifier.evaluate import Inference, hardware, measure, numeric_answer, ppl_from_logprobs, quality_failures, validate_metrics
from nemotron_verifier.main import app
from nemotron_verifier.prepare import select_examples
from nemotron_verifier.queue import Queue
from nemotron_verifier.worker import tick


def metrics(**changes):
    result = {"identity": "frozen-suite", "suite": "mini", "mmlu_correct": 45, "mmlu_count": 57,
              "gsm8k_correct": 13, "gsm8k_count": 16, "ppl": 5.0, "throughput": 100.0, **changes}
    result["mmlu"] = 100 * result["mmlu_correct"] / 57
    result["gsm8k"] = 100 * result["gsm8k_correct"] / 16
    return result


def test_teacher_forced_ppl_masks_overlap_and_weights_token_losses():
    first = ppl_from_logprobs([1, 2, 3, 4], [None, -1.0, -2.0, -3.0], 0)
    second = ppl_from_logprobs([3, 4, 5], [None, -99.0, -4.0], 2)
    assert first == (6, 3) and second == (4, 1)
    assert math.exp((first[0] + second[0]) / (first[1] + second[1])) == math.exp(2.5)
    for invalid in [[None, -1], [None, None, -1, -1], [None, float("nan"), -1, -1], [None, True, -1, -1]]:
        with pytest.raises(ValueError):
            ppl_from_logprobs([1, 2, 3, 4], invalid, 1)


def test_quality_uses_exact_one_answer_tolerance():
    assert quality_failures(metrics(), metrics(mmlu_correct=44, gsm8k_correct=12, ppl=5.05)) == []
    assert len(quality_failures(metrics(), metrics(mmlu_correct=43, gsm8k_correct=11, ppl=5.051))) == 3
    rejected = metrics(mmlu_correct=40)
    rejected.pop("throughput")
    assert quality_failures(metrics(), rejected)
    for value in [None, "100", True, float("nan"), float("inf"), -1, 0]:
        with pytest.raises(ValueError):
            validate_metrics(metrics(throughput=value), throughput=True)
    with pytest.raises(ValueError):
        quality_failures(metrics(), {**metrics(), "mmlu": 1})


def test_gsm8k_requires_final_numeric_answer():
    assert numeric_answer("Work: 12 + 4 = 16.\n#### 16") == 16
    assert numeric_answer("#### 1,234.50") == 1234.5
    for text in ["16", "The calculation was 16, but", "#### 16\nmore reasoning", "#### NaN"]:
        assert numeric_answer(text) is None


def test_hardware_comes_from_authenticated_fresh_remote_vm(monkeypatch):
    monkeypatch.setenv("TEST_NODE_KEY", "node-private-key")
    config = {"hardware": {"base_url": "http://127.0.0.1:8810/v1", "api_key_env": "TEST_NODE_KEY"}}
    gpu = {"name": GPU, "uuid": "GPU-abc", "memory_total_mb": 24564, "power_limit_w": 230, "driver_version": "580.1"}
    document = {"fresh": True, "gpu": gpu}
    def respond(request):
        assert str(request.url) == "http://127.0.0.1:8810/v1/hardware"
        assert request.headers["Authorization"] == "Bearer node-private-key"
        return httpx.Response(200, json=document)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        assert hardware(config, client)["uuid"] == "GPU-abc"
        document["fresh"] = False
        with pytest.raises(ValueError):
            hardware(config, client)
        document["fresh"] = True
        gpu["name"] = "NVIDIA A100"
        with pytest.raises(ValueError):
            hardware(config, client)


def test_durable_queue_binding_and_baseline_immutability(tmp_path):
    path = str(tmp_path / "jobs.sqlite")
    queue = Queue(path)
    job = queue.submit("key", "candidate", {"input": 1})
    assert queue.submit("key", "candidate", {"input": 1}) == job
    with pytest.raises(ValueError):
        queue.submit("key", "candidate", {"input": 2})
    queue.bind_benchmark("challenge", "identity1")
    queue.bind_benchmark("challenge", "identity1")
    with pytest.raises(ValueError):
        queue.bind_benchmark("challenge", "identity2")
    assert queue.claim()["id"] == job["id"] and queue.claim() is None
    queue.db.close()
    restored = Queue(path)
    restored.recover()
    assert restored.get(job["id"])["status"] == "failed"
    assert "score" not in restored.get(job["id"])
    baseline = restored.submit_baseline("frozen-suite", "reference")
    assert restored.submit_baseline("frozen-suite", "reference")["id"] == baseline["id"]
    assert restored.claim()["id"] == baseline["id"]
    restored.finish(baseline["id"], "failed", {"feedback": "failed"})
    retry = restored.submit_baseline("frozen-suite", "reference")
    assert retry["id"] != baseline["id"]
    assert restored.claim()["id"] == retry["id"]
    restored.finish(retry["id"], "completed", {"metrics": metrics()})
    assert restored.submit_baseline("frozen-suite", "reference")["id"] == retry["id"]
    assert restored.baseline("frozen-suite")["ppl"] == 5
    restored.db.close()


def test_api_worker_baseline_quality_rejection_and_source_pin(monkeypatch, tmp_path):
    import nemotron_verifier.main as module
    config = {"reference": "reference", "deployments": {"reference": {"runtime": "vllm"},
              "candidate": {"runtime": "sglang", "repository": "https://github.com/ShardulAgg/sglang", "commit": "a" * 40}}}
    monkeypatch.setenv("NEMOTRON_VERIFIER_DB", str(tmp_path / "jobs.sqlite"))
    monkeypatch.setenv("NEMOTRON_VERIFIER_API_KEY", "private-test-key")
    monkeypatch.setattr(module, "setup", lambda: (config, "frozen-suite"))
    headers = {"Authorization": "Bearer private-test-key"}
    payload = {"repository": "https://github.com/ShardulAgg/sglang", "commit": "a" * 40,
               "deployment": "candidate", "model": MODEL, "gpu": GPU, "precision": "NVFP4"}
    request = {"submission": payload, "submission_id": "submission-1", "benchmark_id": "benchmark-1",
               "base_checkpoint_url": "https://github.com/ShardulAgg/sglang/commit/" + "b" * 40}
    with TestClient(app) as client:
        assert client.get("/health").json()["baseline_status"] == "unmeasured"
        assert client.post("/baseline").status_code == 401
        assert client.post("/evaluate", json=request, headers=headers).status_code == 409
        baseline = client.post("/baseline", headers=headers).json()
        assert "score" not in baseline
        assert client.get("/health").json()["baseline_status"] == "queued"
        assert tick(app.state.queue, config, lambda *a: metrics())
        assert client.get("/health").json()["baseline_ready"] is True
        assert client.get("/baseline", headers=headers).json()["metrics"]["ppl"] == 5
        assert client.post("/baseline", headers=headers).json()["id"] == baseline["id"]
        assert client.post("/evaluate", json=request, headers=headers).status_code == 400
        candidate = client.post("/evaluate", json=request, headers={**headers, "Idempotency-Key": "submission-1"}).json()
        assert candidate["status"] == "queued" and "score" not in candidate
        assert tick(app.state.queue, config, lambda *a: metrics(mmlu_correct=43))
        result = client.get("/evaluations/" + candidate["id"], headers=headers).json()
        assert result["status"] == "rejected" and "score" not in result
        request["submission_id"] = "submission-2"
        accepted = client.post("/evaluate", json=request, headers={**headers, "Idempotency-Key": "submission-2"}).json()
        assert tick(app.state.queue, config, lambda *a: metrics(throughput=200))
        result = client.get("/evaluations/" + accepted["id"], headers=headers).json()
        assert result["status"] == "completed" and result["score"] == 200
        request["submission_id"] = "submission-3"
        changed = client.post("/evaluate", json=request, headers={**headers, "Idempotency-Key": "submission-3"}).json()
        config["deployments"]["candidate"]["settings"] = {"changed": True}
        assert tick(app.state.queue, config, lambda *a: pytest.fail("Changed settings must not be evaluated"))
        assert client.get("/evaluations/" + changed["id"], headers=headers).json()["status"] == "failed"


def test_inference_never_invents_ppl_or_token_counts():
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(400)), base_url="http://127.0.0.1:8100/v1/") as client:
        with pytest.raises(ValueError):
            Inference({}, client).score([1, 2, 3])
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"usage": {"prompt_tokens": 2048, "completion_tokens": 1023}})), base_url="http://127.0.0.1:8100/v1/") as client:
        with pytest.raises(ValueError):
            Inference({}, client).generate_workload([1] * 2048)


def test_workload_uses_twenty_closed_loop_clients_and_aggregate_wall_time():
    engine = Inference.__new__(Inference)
    lock = threading.Lock()
    state = {"active": 0, "peak": 0, "calls": 0}
    def generate(tokens):
        assert len(tokens) == 2048
        with lock:
            state["active"] += 1
            state["peak"] = max(state["active"], state["peak"])
            state["calls"] += 1
        time.sleep(0.01)
        with lock:
            state["active"] -= 1
        return 1024
    engine.generate_workload = generate
    performance = engine.performance([[i] * 2048 for i in range(21)])
    assert state["calls"] == 21 and 1 < state["peak"] <= 20
    assert performance["requests"] == 20 and performance["output_tokens"] == 20480
    assert performance["throughput"] == 20480 / performance["elapsed_seconds"]


def test_quality_failure_skips_all_performance_requests(monkeypatch, tmp_path):
    import nemotron_verifier.evaluate as module
    (tmp_path / "mmlu.json").write_text(json.dumps([{"prompt": "mmlu", "answer": "A"}] * 57))
    (tmp_path / "gsm8k.json").write_text(json.dumps([{"prompt": "gsm", "answer": "1"}] * 16))
    (tmp_path / "ppl_tokens.json").write_text(json.dumps(list(range(4096))))
    (tmp_path / "workload_tokens.json").write_text("[]")
    gpu = {"name": GPU}
    monkeypatch.setattr(module, "hardware", lambda c: gpu)
    monkeypatch.setattr(module, "identity", lambda c, g: "frozen-suite")
    state = {"active": 0, "peak": 0, "calls": 0}
    lock = threading.Lock()
    class FakeEngine:
        def __init__(self, deployment):
            self.client = type("Client", (), {"close": lambda _: None})()
        def check_model(self):
            pass
        def score(self, tokens):
            return [None] + [-math.log(5)] * (len(tokens) - 1)
        def chat(self, prompt, cap):
            assert cap == (8 if prompt == "mmlu" else 512)
            with lock:
                state["active"] += 1
                state["peak"] = max(state["active"], state["peak"])
                state["calls"] += 1
            time.sleep(0.001)
            with lock:
                state["active"] -= 1
            return "B" if prompt == "mmlu" else "#### 1"
        def performance(self, prompts):
            pytest.fail("Rejected quality must stop before any throughput warmup")
    monkeypatch.setattr(module, "Inference", FakeEngine)
    config = {"suite_dir": str(tmp_path), "deployments": {"test": {"runtime": "vllm", "commit": "a" * 40, "settings": {"max_num_seqs": 4}}}}
    result = measure(config, "test", "frozen-suite", metrics())
    assert result["mmlu_correct"] == 0 and result["gsm8k_correct"] == 16
    assert result["ppl_tokens"] == 4095 and result["ppl"] == pytest.approx(5)
    assert "throughput" not in result
    assert state["calls"] == 73 and 1 < state["peak"] <= 4


def test_fixed_sampling_stratifies_mmlu_and_freezes_zero_shot_prompts():
    mmlu = [{"subject": f"subject-{i}", "question": f"question-{j}", "choices": ["a", "b", "c", "d"], "answer": j % 4}
            for i in range(57) for j in range(6)]
    gsm = [{"question": str(i), "answer": f"Reasoning\n#### {i}"} for i in range(100)]
    first = select_examples(mmlu, gsm)
    assert first == select_examples(mmlu, gsm)
    assert len(first[0]) == 57 and len(first[1]) == 16
    assert len({row["test_index"] for row in first[0]}) == 57
    assert all(sum(row["subject"] == subject for row in first[0]) == 1 for subject in {row["subject"] for row in first[0]})
    assert all(row["prompt"].count("Answer:") == 1 for row in first[0] + first[1])


def test_no_gpu_or_suite_reports_not_ready(monkeypatch, tmp_path):
    monkeypatch.setenv("NEMOTRON_VERIFIER_DB", str(tmp_path / "db.sqlite"))
    monkeypatch.setenv("NEMOTRON_VERIFIER_CONFIG", str(tmp_path / "missing.json"))
    monkeypatch.setenv("NEMOTRON_VERIFIER_API_KEY", "test")
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["ready"] is False and health["baseline_ready"] is False
        assert client.post("/baseline", headers={"Authorization": "Bearer test"}).status_code == 503


def test_registry_cannot_send_requests_to_caller_chosen_hosts(tmp_path):
    config = json.loads((Path(__file__).parents[1] / "config.example.json").read_text())
    path = tmp_path / "config.json"
    for url in ["https://attacker.example/v1", "http://127.0.0.1:8000/v1?key=secret", "http://user:pass@127.0.0.1:8000/v1"]:
        config["deployments"]["vllm-reference"]["base_url"] = url
        path.write_text(json.dumps(config))
        with pytest.raises(ValueError):
            load_config(path)


def test_prepared_manifest_identity_detects_modified_samples(tmp_path):
    files = {"mmlu.json": [{"subject": str(i), "answer": "A"} for i in range(57)],
             "gsm8k.json": [{}] * 16, "ppl_tokens.json": [1] * 4096,
             "workload_tokens.json": [[1] * 2048] * 21}
    checksums = {}
    import hashlib
    for name, data in files.items():
        content = json.dumps(data).encode()
        (tmp_path / name).write_bytes(content)
        checksums[name] = hashlib.sha256(content).hexdigest()
    manifest = {"suite": "mini", "model": MODEL, "revision": REVISION, "recipe": RECIPE,
                "mmlu_count": 57, "gsm8k_count": 16, "ppl_token_count": 4096, "files": checksums}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert suite_identity({"suite_dir": str(tmp_path)}) == digest(manifest)
    (tmp_path / "mmlu.json").write_text("tampered")
    with pytest.raises(ValueError):
        suite_identity({"suite_dir": str(tmp_path)})


def test_parallel_baseline_requests_coalesce_without_duplicate_jobs(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    queue = Queue(str(tmp_path / "jobs.sqlite"))
    with ThreadPoolExecutor(max_workers=8) as pool:
        requests = list(pool.map(lambda _: queue.submit_baseline("identity", "reference"), range(30)))
    assert len({job["id"] for job in requests}) == 1
    assert queue.claim()["id"] == requests[0]["id"]
    assert queue.claim() is None
    queue.db.close()


def test_private_receipt_keys_are_read_locally_and_symlinks_rejected(tmp_path):
    receipt = tmp_path / "node.json"
    receipt.write_text(json.dumps({"key": "private-scoped-node-key"}))
    receipt.chmod(0o600)
    assert key_headers({"connection_file": str(receipt)}) == {"Authorization": "Bearer private-scoped-node-key"}
    receipt.chmod(0o644)
    with pytest.raises(ValueError):
        key_headers({"connection_file": str(receipt)})
    receipt.chmod(0o600)
    link = tmp_path / "link.json"
    link.symlink_to(receipt)
    with pytest.raises(OSError):
        key_headers({"connection_file": str(link)})


def test_runtime_target_is_advisory_and_never_rejects_a_slow_valid_run(monkeypatch, tmp_path):
    import nemotron_verifier.evaluate as module
    (tmp_path / "mmlu.json").write_text(json.dumps([{"prompt": "mmlu", "answer": "A"}] * 57))
    (tmp_path / "gsm8k.json").write_text(json.dumps([{"prompt": "gsm", "answer": "1"}] * 16))
    (tmp_path / "ppl_tokens.json").write_text(json.dumps(list(range(4096))))
    (tmp_path / "workload_tokens.json").write_text(json.dumps([[1] * 2048] * 21))
    monkeypatch.setattr(module, "hardware", lambda c: {"name": GPU})
    monkeypatch.setattr(module, "identity", lambda c, g: "frozen-suite")
    clock = iter([0.0, 1300.0, 1301.0])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(clock))
    class FakeEngine:
        def __init__(self, deployment):
            self.client = type("Client", (), {"close": lambda _: None})()
        def check_model(self):
            pass
        def score(self, tokens):
            return [None] + [-math.log(5)] * (len(tokens) - 1)
        def chat(self, prompt, cap):
            return "A" if prompt == "mmlu" else "#### 1"
        def performance(self, prompts):
            assert len(prompts) == 21
            return {"throughput": 100.0}
    monkeypatch.setattr(module, "Inference", FakeEngine)
    config = {"suite_dir": str(tmp_path), "deployments": {"test": {"runtime": "vllm", "commit": "a" * 40, "settings": {"max_num_seqs": 4}}}}
    result = measure(config, "test", "frozen-suite", metrics())
    assert result["runtime_seconds"] == 1301
    assert result["runtime_within_target"] is False and result["runtime_target_is_advisory"] is True
    assert result["throughput"] == 100.0
    assert quality_failures(metrics(), result) == []
