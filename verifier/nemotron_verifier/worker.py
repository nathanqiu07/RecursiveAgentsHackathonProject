import fcntl
import json
import logging
import os
import time
from pathlib import Path

from .config import digest, load_config
from .evaluate import measure, quality_failures, validate_metrics
from .queue import Queue


def tick(queue, config, evaluator=measure):
    job = queue.claim()
    if not job:
        return False
    try:
        definition = job["definition"]
        deployment = config["deployments"][definition["deployment"]]
        if job["kind"] == "candidate":
            payload = definition["submission"]
            if (any(payload[field] != deployment[field] for field in ["repository", "commit"])
                    or definition["deployment_identity"] != digest(deployment)):
                raise ValueError("Registered source changed after this evaluation was queued")
        reference = queue.baseline(definition["identity"]) if job["kind"] == "candidate" else None
        if job["kind"] == "candidate" and reference is None:
            raise ValueError("A matching immutable baseline is required")
        candidate = evaluator(config, definition["deployment"], definition["identity"], reference)
        if candidate.get("identity") != definition["identity"]:
            raise ValueError("Measurement did not match the queued evaluation identity")
        # This also validates baseline metrics before freezing a reference.
        quality_failures(candidate, candidate)
        if job["kind"] == "baseline":
            validate_metrics(candidate, throughput=True)
            queue.finish(job["id"], "completed", {"metrics": candidate,
                                                  "feedback": "Reference baseline measured. No leaderboard score."})
        else:
            if reference is None or candidate["identity"] != reference["identity"]:
                raise ValueError("A matching immutable baseline is required")
            failures = quality_failures(reference, candidate)
            feedback = json.dumps({"reference": reference, "candidate": candidate,
                                   "quality_failures": failures})
            result = {"feedback": feedback, "metrics": candidate}
            if failures:
                queue.finish(job["id"], "rejected", result)
            else:
                validate_metrics(candidate, throughput=True)
                queue.finish(job["id"], "completed", {**result, "score": candidate["throughput"]})
    except Exception as error:
        # Never include endpoint bodies, request headers, credentials, or submitted notes in logs.
        logging.error("Evaluation %s failed (%s)", job["id"], type(error).__name__)
        queue.finish(job["id"], "failed", {"feedback": "Evaluation could not complete. No score recorded. Check the worker and engine configuration."})
    return True


def main():
    path = os.environ.get("NEMOTRON_VERIFIER_DB", "data/verifier.sqlite")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path + ".worker.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Only one verifier worker may evaluate this GPU") from None
        queue = Queue(path)
        try:
            initial = load_config()
        except (OSError, ValueError, KeyError, TypeError):
            initial = None
        if initial and initial.get("nodes"):
            from .coordinator import run
            try:
                run(queue, initial)
            finally:
                queue.db.close()
            return
        recovered = False
        waiting_for_config = False
        try:
            while True:
                try:
                    config = load_config()
                    if config.get("nodes"):
                        from .coordinator import run
                        run(queue, config)
                        return
                    if not recovered:
                        queue.recover()
                        recovered = True
                    waiting_for_config = False
                except (OSError, ValueError, KeyError, TypeError):
                    if not waiting_for_config:
                        logging.warning("Worker waiting for local verifier configuration")
                    waiting_for_config = True
                    time.sleep(5)
                    continue
                if not tick(queue, config):
                    time.sleep(1)
        finally:
            queue.db.close()


if __name__ == "__main__":
    main()
