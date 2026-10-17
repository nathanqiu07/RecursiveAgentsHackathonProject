"""Runs on the controller laptop: one independent evaluation lane per GPU node."""

import logging
import multiprocessing
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .config import digest, load_config
from .evaluate import measure, quality_failures, validate_metrics
from .pool import NodeBusy, NodeClient, inspect_pool, profile, wait_for_engine


def _measure_child(sender, config, alias, fingerprint, reference):
    try:
        sender.send({"ok": True, "metrics": measure(config, alias, fingerprint, reference)})
    except Exception:
        sender.send({"ok": False})
    finally:
        sender.close()


def local_measure(config, alias, fingerprint, reference, cancelled, stop_event):
    """A subprocess permits explicit cancellation. Elapsed time never kills a run."""
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_measure_child, args=(sender, config, alias, fingerprint, reference))
    process.start()
    sender.close()
    try:
        while True:
            if cancelled() or stop_event.is_set():
                raise InterruptedError("Evaluation cancelled")
            if receiver.poll(.2):
                result = receiver.recv()
                if not result.get("ok"):
                    raise ValueError("Local measurement failed")
                return result["metrics"]
            if not process.is_alive():
                raise ValueError("Local measurement stopped without a result")
    finally:
        receiver.close()
        process.join(timeout=1)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join(timeout=2)


def cleanup(client, job_id, attempts=20):
    """Quarantine unconfirmed cleanup; never assign a second VM on that GPU."""
    try:
        client.cancel(job_id)
        for _ in range(attempts):
            if client.job(job_id).get("cleanup_confirmed") is True:
                return True
            time.sleep(.5)
    except Exception:
        pass
    return False


def run_job(queue, config, node, pool_identity, client, health, job,
            stop_event, evaluator=local_measure):
    definition, job_id = job["definition"], job["id"]
    started = time.monotonic()
    accepted, busy = False, False
    status, result = "failed", {"feedback": "Evaluation could not complete. No score recorded."}
    cancelled = lambda: bool((queue.get(job_id) or {}).get("cancel_requested"))
    try:
        alias = definition["deployment"]
        deployment = config["deployments"][alias]
        if definition.get("pool_identity") != pool_identity:
            raise ValueError("Pool identity changed")
        if job["kind"] == "candidate":
            payload = definition["submission"]
            if (definition["deployment_identity"] != digest(deployment)
                    or any(payload[field] != deployment[field] for field in ("repository", "commit"))):
                raise ValueError("Deployment changed after submission")
        reference = queue.baseline(node["identity"]) if job["kind"] == "candidate" else None
        if job["kind"] == "candidate" and reference is None:
            raise ValueError("Node requires a matching frozen reference")
        if cancelled() or stop_event.is_set():
            raise InterruptedError()
        # Persisted assignment uses the same UUID remotely, allowing crash recovery.
        # Treat uncertain POST outcomes as accepted: cancel by id before any reuse.
        accepted = True
        try:
            remote = client.start(job_id, alias, health, config)
        except NodeBusy:
            accepted, busy = False, True
            if not queue.requeue(job_id):
                queue.finish(job_id, "cancelled", {"feedback": "Evaluation cancelled before a VM was accepted. No score recorded."})
            return
        if remote.get("id") != job_id:
            raise ValueError("Remote job identity mismatch")
        wait_for_engine(client, job_id, deployment, cancelled, stop_event)
        boot_seconds = time.monotonic() - started
        node_config = profile(config, node, client.receipt)
        candidate = evaluator(node_config, alias, node["identity"], reference, cancelled, stop_event)
        if candidate.get("identity") != node["identity"]:
            raise ValueError("Node measurement identity changed")
        quality_failures(candidate, candidate)
        if cancelled() or stop_event.is_set():
            raise InterruptedError()
        failures = quality_failures(reference, candidate) if reference else []
        candidate.update(node_id=node["id"], pool_identity=pool_identity,
                         vm_boot_seconds=boot_seconds, queue_seconds=max(0, time.time() - job["created"] - (time.monotonic() - started)))
        result = {"metrics": candidate, "feedback": "Reference baseline measured. No leaderboard score."}
        if failures:
            status, result["feedback"] = "rejected", "; ".join(failures)
        else:
            validate_metrics(candidate, throughput=True)
            status = "completed"
            if reference is not None:
                result.update(score=candidate["throughput"], feedback="Mini quality checks passed against this node's frozen reference.")
    except InterruptedError:
        status, result = "cancelled", {"feedback": "Evaluation cancelled. No score recorded."}
    except Exception as error:
        logging.error("Evaluation %s on node %s failed (%s)", job_id, node["id"], type(error).__name__)
    finally:
        if not busy:
            cleaned = not accepted or cleanup(client, job_id)
            if not cleaned:
                status, result = "failed", {"feedback": "VM cleanup is unconfirmed; node remains unavailable. No score recorded."}
            elapsed = time.monotonic() - started
            result["elapsed_seconds"] = elapsed
            result["runtime_target_seconds"] = [600, 1200]
            result["runtime_within_target"] = elapsed <= 1200
            result["runtime_target_is_advisory"] = True
            result["cleanup_confirmed"] = cleaned
            queue.finish(job_id, status, result)


def tick_node(queue, config, node_id, stop_event, client_factory=NodeClient, evaluator=local_measure):
    pool_identity, states = inspect_pool(config, client_factory)
    node = next((entry for entry in states if entry["id"] == node_id), None)
    if not node or node["status"] != "available":
        return False
    client = client_factory(node)
    try:
        health = client.health()
        if health.get("busy"):
            return False
        job = queue.claim_node(node_id, pool_identity, node["identity"])
        if not job:
            return False
        run_job(queue, config, node, pool_identity, client, health, job, stop_event, evaluator)
        return True
    finally:
        client.close()


def recover(queue, config, client_factory=NodeClient):
    from .pool import configured_nodes
    nodes = {node["id"]: node for node in configured_nodes(config)}
    for job in queue.interrupted():
        node = nodes.get(job["node_id"])
        confirmed = False
        if node:
            try:
                client = client_factory(node)
                try:
                    confirmed = cleanup(client, job["id"])
                finally:
                    client.close()
            except Exception:
                pass
        queue.finish(job["id"], "failed", {
            "feedback": "Coordinator interrupted. No score recorded; explicit resubmission is required.",
            "cleanup_confirmed": confirmed,
        })


def run(queue, config, stop_event=None):
    from .pool import configured_nodes
    stop_event = stop_event or threading.Event()
    if threading.current_thread() is threading.main_thread():
        for name in (signal.SIGTERM, signal.SIGINT):
            signal.signal(name, lambda *_: stop_event.set())
    recover(queue, config)
    def lane(node_id):
        while not stop_event.is_set():
            try:
                current = load_config()
                worked = tick_node(queue, current, node_id, stop_event)
            except Exception as error:
                logging.error("Node %s coordinator waiting (%s)", node_id, type(error).__name__)
                worked = False
            if not worked:
                stop_event.wait(2)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(lane, node["id"]) for node in configured_nodes(config)]
        try:
            for future in futures:
                future.result()
        finally:
            stop_event.set()
