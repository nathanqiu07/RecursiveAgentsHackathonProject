"""Model control plane; all generated commands execute in the run's compute sandbox."""

import hashlib
import json
import queue
import re
import threading
import time

from jsonschema import Draft202012Validator, FormatChecker, validators

from .clients import ServiceError
from .prompts import EVIDENCE_REVIEW, NOVELTY_REVIEW, instruction, tools_for
from .store import TERMINAL, LostLease, RunCancelled

CHECKPOINT = re.compile(
    r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/"
    r"(?:commit|tree)/(?:[a-fA-F0-9]{40}|[a-fA-F0-9]{64})/?\Z"
)


class GateError(Exception):
    pass


class EndRun(Exception):
    def __init__(self, status, result=None, error=None):
        self.status, self.result, self.error = status, result, error


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class Engine:
    def __init__(self, store, run, token, services, model, web, stop=None):
        self.store, self.run, self.token = store, run, token
        self.services, self.model, self.web = services, model, web
        self.stop = stop or threading.Event()
        self.id = run["id"]
        self.role = run["request"]["role"]
        self.state = run["state"]
        self.job_id = run["compute_job_id"]
        self.deadline = run["started_at"] + run["request"]["allocation"]["timeout_seconds"]
        self.tools = {item["function"]["name"]: item for item in tools_for(self.role)}

    def save(self):
        self.store.update(self.id, self.token, state=self.state)

    def event(self, kind, data):
        self.store.event(self.id, self.token, kind, data)

    def check(self):
        self.store.heartbeat(self.id, self.token)
        if self.stop.is_set():
            raise EndRun("needs_attention", error="worker_stopped")
        if self.store.now() >= self.deadline:
            raise EndRun("failed", error="run_time_budget_exceeded")

    def call(self, function):
        """Keep ownership and cancellation responsive during bounded network requests."""
        self.check()
        output = queue.Queue(maxsize=1)

        def invoke():
            try:
                output.put((True, function()))
            except Exception as exc:
                output.put((False, exc))

        threading.Thread(target=invoke, daemon=True).start()
        while True:
            try:
                success, value = output.get(timeout=0.5)
                self.check()
                if success:
                    return value
                raise value
            except queue.Empty:
                self.check()

    def allocate(self):
        self.check()
        self.store.update(self.id, self.token, allocation_started=True, cleanup_required=True)
        try:
            job = self.call(
                lambda: self.services.allocate(
                    self.run["request"]["allocation"],
                    "harness-" + self.id,
                )
            )
        except ServiceError as exc:
            if not exc.uncertain:
                self.store.update(
                    self.id, self.token, allocation_started=False, cleanup_required=False
                )
            raise
        self.job_id = job["id"]
        self.store.update(
            self.id,
            self.token,
            compute_job_id=self.job_id,
            sandbox_id=job.get("sandbox_id"),
            cleanup_required=True,
        )
        self.event(
            "sandbox_allocated",
            {"compute_job_id": self.job_id, "sandbox_id": job.get("sandbox_id")},
        )
        while job["status"] != "running":
            if job["status"] in {"completed", "failed", "cancelled"}:
                raise GateError("sandbox_unavailable_or_wrong_image")
            time.sleep(0.25)
            job = self.call(lambda: self.services.job(self.job_id))

    def cleanup(self):
        """Never release a run's cleanup flag based solely on a disconnected worker."""
        self.store.owned(self.id, self.token)
        run = self.store.get(self.id)
        if not run["allocation_started"]:
            self.store.update(
                self.id,
                self.token,
                cleanup_required=bool(self.state.get("runtime_cleanup_required")),
            )
            return
        try:
            if not self.job_id:
                # Recover an uncertain allocation with the original idempotency key.
                job = self.services.allocate(
                    self.run["request"]["allocation"], "harness-" + self.id
                )
                self.job_id = job["id"]
                self.store.update(
                    self.id,
                    self.token,
                    compute_job_id=self.job_id,
                    sandbox_id=job.get("sandbox_id"),
                )
            job = self.services.cancel(self.job_id)
            confirmed = (
                job["status"] in {"completed", "failed", "cancelled"}
                and job.get("cleanup_required") is False
            )
            self.store.update(
                self.id,
                self.token,
                cleanup_required=not confirmed or bool(self.state.get("runtime_cleanup_required")),
            )
            self.event("sandbox_cleanup", {"confirmed": confirmed, "compute_job_id": self.job_id})
        except ServiceError:
            self.store.update(self.id, self.token, cleanup_required=True)

    def close_runtime(self, model=None):
        reference = self.state.get("runtime_cleanup")
        if model is None and not reference:
            return
        if model is None:
            from .cli_models import reconcile_runtime_cleanup

            def operation():
                return reconcile_runtime_cleanup(reference)
        else:
            operation = model.close
        finished = queue.Queue(maxsize=1)

        def close():
            try:
                finished.put((True, operation()))
            except Exception as exc:
                finished.put((False, exc))

        threading.Thread(target=close, daemon=True).start()
        while True:
            # Teardown can outlast an ownership lease. Cancellation must not skip it.
            try:
                self.store.heartbeat(self.id, self.token)
            except RunCancelled:
                pass
            try:
                ok, value = finished.get(timeout=0.5)
                break
            except queue.Empty:
                continue
        if not ok:
            raise value
        if reference:
            self.state.pop("runtime_cleanup", None)
            self.state["runtime_cleanup_required"] = False
            self.save()
            self.event("runtime_cleanup", {"confirmed": True})

    def execute(self, args):
        if self.role == "researcher" and not self.state.get("novelty", {}).get("approved"):
            raise GateError("Review the idea's novelty against the Library before experiments.")
        command_id = f"execute-{self.state['steps']}"
        message = {
            "type": "execute",
            "id": command_id,
            "command": args["command"],
            "timeout_seconds": args["timeout_seconds"],
        }
        self.event(
            "experiment_started",
            {**message, "purpose": args["purpose"], "command_sha256": digest(args["command"])},
        )
        self.call(lambda: self.services.send(self.job_id, command_id, json.dumps(message)))
        cursor = self.state.get("cursor", 0)
        while True:
            output = self.call(lambda: self.services.output(self.job_id, cursor))
            for line in output["lines"]:
                try:
                    result = json.loads(line["line"])
                except (TypeError, ValueError):
                    raise GateError("sandbox_returned_invalid_protocol") from None
                if (
                    result.get("type") != "execution_result"
                    or result.get("id") != command_id
                    or type(result.get("exit_code")) is not int
                    or not isinstance(result.get("output"), str)
                    or len(result["output"].encode()) > 65000
                    or type(result.get("truncated")) is not bool
                ):
                    raise GateError("sandbox_returned_invalid_protocol")
                cursor = line["sequence"]
                self.state["cursor"] = cursor
                evidence = {
                    **message,
                    **result,
                    "purpose": args["purpose"],
                    "step": self.state["steps"],
                    "command_sha256": digest(args["command"]),
                }
                self.state.setdefault("experiments", []).append(evidence)
                self.save()
                self.event("experiment_result", evidence)
                return result
            cursor = output["next_cursor"]
            self.state["cursor"] = cursor
            job = self.call(lambda: self.services.job(self.job_id))
            if job["status"] in {"completed", "failed", "cancelled"}:
                raise GateError("sandbox_exited_before_command_result")
            time.sleep(0.25)

    def library_snapshot(self):
        papers = self.call(self.services.papers)
        snapshot = []
        for paper in papers:
            submission = paper.get("submission", {})
            text = submission.get("paper_text")
            if not isinstance(text, str) or not text.strip():
                raise GateError(
                    "Library novelty review is incomplete: a paper has no readable text."
                )
            snapshot.append(
                {
                    "id": paper["id"],
                    "title": submission.get("title"),
                    "paper_text": text,
                    "status": paper.get("status"),
                    "report": paper.get("report"),
                }
            )
        snapshot.sort(key=lambda paper: paper["id"])
        if len(json.dumps(snapshot)) > 180000:
            raise GateError("Library exceeds the complete novelty review context limit.")
        return snapshot

    def novelty(self, idea, final=False):
        snapshot = self.library_snapshot()
        result = self.call(
            lambda: self.model.complete_json(
                NOVELTY_REVIEW,
                {"candidate": idea, "library_papers": snapshot, "final_paper": final},
            )
        )
        ids = {paper["id"] for paper in snapshot}
        related = result.get("related_paper_ids")
        if (
            type(result.get("novel")) is not bool
            or not isinstance(result.get("rationale"), str)
            or not result["rationale"].strip()
            or not isinstance(related, list)
            or any(item not in ids for item in related)
        ):
            raise GateError("Novelty reviewer returned an invalid assessment.")
        assessment = {
            "approved": result["novel"],
            "rationale": result["rationale"][:10000],
            "related_paper_ids": related,
            "compared_paper_ids": sorted(ids),
            "snapshot_sha256": digest(snapshot),
            "idea": idea,
            "step": self.state["steps"],
            "scope": "Library snapshot only",
        }
        if not final:
            self.state["novelty"] = assessment
            self.save()
        self.event("final_novelty_review" if final else "novelty_review", assessment)
        return assessment

    def evidence(self, ids, purpose):
        if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
            raise GateError("Provide unique recorded evidence IDs.")
        records = {item["id"]: item for item in self.state.get("experiments", [])}
        if any(item not in records for item in ids):
            raise GateError("Evidence must refer to this run's recorded sandbox executions.")
        selected = [records[item] for item in ids]
        successes = [
            item
            for item in selected
            if item["purpose"] == purpose
            and item["exit_code"] == 0
            and not item["truncated"]
            and item["output"].strip()
        ]
        if self.role == "researcher":
            successes = [
                item
                for item in successes
                if item["step"] > self.state.get("novelty", {}).get("step", float("inf"))
            ]
        if not successes:
            raise GateError(f"At least one successful, nontruncated {purpose} record is required.")
        return selected

    def evidence_review(self, candidate, evidence):
        review = self.call(
            lambda: self.model.complete_json(
                EVIDENCE_REVIEW,
                {
                    "role": self.role,
                    "candidate": candidate,
                    "recorded_executions": evidence,
                    "benchmark": self.state.get("benchmark"),
                },
            )
        )
        self.event("evidence_review", review)
        if (
            review.get("supported") is not True
            or not isinstance(review.get("rationale"), str)
            or not review["rationale"].strip()
        ):
            raise GateError(
                "Evidence reviewer did not approve publication: "
                + str(review.get("rationale", "invalid assessment"))[:2000]
            )

    def publish(self, kind, body):
        if self.state.get("publication"):
            raise EndRun("needs_attention", error="publication_already_started_do_not_replay")
        self.state["publication"] = {
            "kind": kind,
            "body_sha256": digest(body),
            "status": "dispatching",
        }
        self.save()  # Durable intent precedes either non-idempotent publication endpoint.
        self.event("publication_started", self.state["publication"])
        try:
            if kind == "paper":
                result = self.call(lambda: self.services.submit_paper(body))
            else:
                result = self.call(
                    lambda: self.services.submit_attempt(
                        self.run["request"]["benchmark_id"],
                        body,
                    )
                )
        except ServiceError as exc:
            if exc.uncertain:
                raise EndRun("needs_attention", error="publication_outcome_unknown_do_not_resubmit")
            self.state.pop("publication")
            self.save()
            raise
        self.state["publication"].update(status="acknowledged", id=result["id"])
        self.save()
        self.event(
            "publication_accepted",
            {"kind": kind, "id": result["id"], "status": result.get("status")},
        )
        raise EndRun(
            "submitted",
            result={"kind": kind, "id": result["id"], "verification_status": result.get("status")},
        )

    def tool(self, name, args):
        definition = self.tools.get(name)
        if not definition:
            raise GateError("This tool is not available to this agent role.")
        errors = list(
            Draft202012Validator(
                definition["function"]["parameters"],
                format_checker=FormatChecker(),
            ).iter_errors(args)
        )
        if errors:
            raise GateError("Invalid tool arguments: " + errors[0].message[:300])
        if name == "search_library":
            papers = self.call(self.services.papers)
            words = args["query"].lower().split()
            matches = []
            for paper in papers:
                data = paper.get("submission", {})
                text = str(data.get("paper_text") or "")
                if any(word in (str(data.get("title")) + " " + text).lower() for word in words):
                    matches.append(
                        {
                            "id": paper["id"],
                            "title": data.get("title"),
                            "excerpt": text[:1000],
                            "status": paper.get("status"),
                            "report": paper.get("report"),
                        }
                    )
            self.state["library_searched"] = True
            matched = set(self.state.get("matched_paper_ids", []))
            matched.update(paper["id"] for paper in matches)
            self.state["matched_paper_ids"] = sorted(matched)
            self.save()
            return {"papers": matches[:20], "matches": len(matches), "library_total": len(papers)}
        if name == "read_paper":
            paper = self.call(lambda: self.services.paper(args["paper_id"]))
            text = paper.get("submission", {}).get("paper_text")
            if not isinstance(text, str) or not text.strip():
                raise GateError("This Library paper has no readable paper_text.")
            self.state.setdefault("read_paper_ids", []).append(args["paper_id"])
            self.save()
            return paper
        if name == "read_benchmark":
            benchmark_id = self.run["request"].get("benchmark_id")
            if not benchmark_id:
                raise GateError("This run has no assigned benchmark.")
            benchmark = self.call(lambda: self.services.benchmark(benchmark_id))
            self.state["benchmark"] = benchmark
            self.save()
            return benchmark
        if name == "web_search":
            return self.call(lambda: self.web.search(args["query"]))
        if name == "web_read":
            return self.call(lambda: self.web.read(args["url"]))
        if name == "execute":
            if self.role == "attempt" and not self.state.get("benchmark"):
                raise GateError("Read the benchmark contract before implementing an attempt.")
            return self.execute(args)
        if name == "review_novelty":
            return self.novelty(args["idea"])
        if name == "submit_paper":
            if not self.state.get("novelty", {}).get("approved"):
                raise GateError("An approved Library novelty review is required.")
            evidence = self.evidence(args["evidence_ids"], "experiment")
            review = self.novelty(args["paper_text"], final=True)
            if not review["approved"]:
                raise GateError("Final paper novelty review failed: " + review["rationale"][:2000])
            self.evidence_review(args["paper_text"], evidence)
            if digest(self.library_snapshot()) != review["snapshot_sha256"]:
                raise GateError("The Library changed during review; recheck before submitting.")
            appendix = "\n\n## Harness experiment records\n\n" + json.dumps(evidence, indent=2)
            return self.publish(
                "paper",
                {
                    "title": args["title"],
                    "paper_text": args["paper_text"] + appendix,
                    "authors": args["authors"],
                    "submitted_by": "researcher:" + self.id,
                    "notes": f"Harness run {self.id}. Novelty assessed against Library snapshot "
                    f"{review['snapshot_sha256']} ({len(review['compared_paper_ids'])} papers). "
                    "Pending independent Library verification.",
                },
            )
        if name == "submit_attempt":
            benchmark = self.state.get("benchmark")
            if not benchmark or not self.state.get("library_searched"):
                raise GateError("Read the benchmark and search the Library before submitting.")
            matched = set(self.state.get("matched_paper_ids", []))
            if matched and not matched.intersection(self.state.get("read_paper_ids", [])):
                raise GateError(
                    "Read at least one relevant matched Library paper before submitting."
                )
            if not CHECKPOINT.fullmatch(args["base_checkpoint_url"]):
                raise GateError(
                    "A public GitHub checkpoint pinned to a full commit SHA is required."
                )
            schema = benchmark["submission_schema"]
            validator = validators.validator_for(schema)
            validator.check_schema(schema)
            # Remote schema references are not allowed to trigger host-side network access.
            if re.search(r'"\$(?:ref|dynamicRef|recursiveRef)"\s*:\s*"(?!#)', json.dumps(schema)):
                raise GateError("Benchmark schema has unsupported external references.")
            if not validator(schema, format_checker=FormatChecker()).is_valid(args["payload"]):
                raise GateError("Payload does not satisfy the benchmark's submission_schema.")
            evidence = self.evidence(args["evidence_ids"], "test")
            self.evidence_review(args, evidence)
            return self.publish(
                "attempt",
                {
                    "title": args["title"],
                    "base_checkpoint_url": args["base_checkpoint_url"],
                    "payload": args["payload"],
                    "submitted_by": "attempt:" + self.id,
                },
            )
        if name == "finish":
            raise EndRun("completed", result={"summary": args["summary"], "published": False})
        raise GateError("Unknown tool")

    def run_agent(self):
        self.allocate()
        messages = [
            {"role": "system", "content": instruction(self.role)},
            {"role": "user", "content": json.dumps(self.run["request"])},
        ]
        self.state.update(messages=messages, steps=0, experiments=[])
        self.save()
        seen_ids = set()
        while self.state["steps"] < self.run["request"]["max_steps"]:
            if len(json.dumps(messages)) > 220000:
                raise EndRun("completed", error="model_context_budget_exceeded")
            message = self.call(lambda: self.model.complete(messages, list(self.tools.values())))
            calls = message.get("tool_calls") or []
            if not calls:
                raise EndRun(
                    "completed",
                    result={"summary": str(message.get("content") or ""), "published": False},
                )
            if not isinstance(calls, list) or len(calls) > 8:
                raise GateError("invalid_model_tool_calls")
            messages.append(message)
            self.save()
            for call in calls:
                self.check()
                if self.state["steps"] >= self.run["request"]["max_steps"]:
                    raise EndRun("completed", error="step_budget_exceeded")
                self.state["steps"] += 1
                self.save()
                call_id = call.get("id")
                if not isinstance(call_id, str) or not call_id or call_id in seen_ids:
                    raise GateError("invalid_or_reused_model_tool_id")
                seen_ids.add(call_id)
                function = call.get("function", {})
                name = function.get("name")
                try:
                    arguments = function.get("arguments", "")
                    if not isinstance(arguments, str) or len(arguments) > 140000:
                        raise GateError("invalid_model_tool_arguments")
                    args = json.loads(arguments)
                    if not isinstance(args, dict):
                        raise GateError("tool_arguments_must_be_an_object")
                    self.event("tool_call", {"name": name, "call_id": call_id})
                    result = self.tool(name, args)
                except (GateError, ServiceError, ValueError) as exc:
                    result = {
                        "error": exc.code if isinstance(exc, ServiceError) else str(exc)[:2000]
                    }
                rendered = json.dumps(result, allow_nan=False)
                if len(rendered) > 70000:
                    rendered = json.dumps({"truncated": True, "source_excerpt": rendered[:65000]})
                messages.append({"role": "tool", "tool_call_id": call_id, "content": rendered})
                self.save()
        raise EndRun("completed", error="step_budget_exceeded")


def run_once(store, services, model, web, stop=None, *, claim_queued=True, model_factory=None):
    claim = store.claim(include_queued=claim_queued)
    if not claim:
        return False
    run, token, interrupted = claim
    engine = Engine(store, run, token, services, model, web, stop)
    status, result, error = run["status"], run["result"], run["error"]
    owned_model = None
    try:
        if run["status"] in TERMINAL:
            pass  # Reconcile cleanup only, never restart a finished agent.
        elif interrupted:
            status, error = "needs_attention", "interrupted_run_not_replayed"
            if run["state"].get("publication", {}).get("status") == "dispatching":
                error = "publication_outcome_unknown_do_not_resubmit"
        else:
            if model_factory:
                owned_model = engine.model = model_factory(run["request"])
                engine.state["runtime"] = {
                    "harness": run["request"].get("harness", "builtin"),
                    "model": run["request"].get("model") or getattr(owned_model, "model", None),
                }
                if hasattr(owned_model, "cleanup_reference"):
                    engine.state["runtime_cleanup"] = owned_model.cleanup_reference()
                    engine.state["runtime_cleanup_required"] = True
                    store.update(run["id"], token, cleanup_required=True)
                engine.save()
            engine.run_agent()
    except EndRun as exc:
        status, result, error = exc.status, exc.result, exc.error
    except RunCancelled:
        status, error = "cancelled", "cancelled_by_user"
    except LostLease:
        if owned_model:
            # The durable reference lets the new owner reconcile an interrupted teardown.
            def close_lost_model():
                try:
                    owned_model.close()
                except Exception:
                    pass

            threading.Thread(target=close_lost_model, daemon=True).start()
        return True  # The next owner will reconcile the remote sandbox.
    except ServiceError as exc:
        status, error = "failed", exc.code
    except Exception:
        status, error = "failed", "agent_runtime_error"
    # Cancellation or interruption during a publication request leaves an uncertain outcome.
    if engine.state.get("publication", {}).get("status") == "dispatching":
        status, error = "needs_attention", "publication_outcome_unknown_do_not_resubmit"
    if owned_model or engine.state.get("runtime_cleanup"):
        try:
            engine.close_runtime(owned_model)
        except LostLease:
            return True
        except Exception:
            status, error = "needs_attention", "harness_runtime_cleanup_required"
            engine.state["runtime_cleanup_required"] = True
            try:
                engine.save()
            except LostLease:
                pass
    try:
        engine.cleanup()
        store.finish(run["id"], token, status, error, result)
    except LostLease:
        pass
    return True
