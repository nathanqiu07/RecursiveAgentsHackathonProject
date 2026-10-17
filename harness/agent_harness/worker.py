"""Start any number of workers; a run's lease gives it one orchestrator and sandbox."""

import argparse
import signal
import threading
from contextlib import ExitStack
from uuid import uuid4

from .clients import ServiceError
from .config import database_path, services_from_env, web_from_settings
from .engine import run_once
from .profiles import harness_catalog, model_for_run
from .settings import SettingsError
from .store import RunStore


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default=database_path())
    parser.add_argument(
        "--once", action="store_true", help="Process one run or cleanup reconciliation"
    )
    args = parser.parse_args(argv)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_args: stop.set())
    signal.signal(signal.SIGTERM, lambda *_args: stop.set())
    store = RunStore(args.db_path)
    store.worker_id = str(uuid4())
    try:
        while not stop.is_set():
            store.touch_worker()
            worked = False
            try:
                # Reopen connections for each run, so dashboard edits require no restart.
                with ExitStack() as clients:
                    configured = any(
                        runtime["available"]
                        for runtime in harness_catalog(args.db_path)["harnesses"]
                    )
                    services = clients.enter_context(services_from_env())
                    web = clients.enter_context(web_from_settings(args.db_path))
                    worked = run_once(
                        store,
                        services,
                        None,
                        web,
                        stop,
                        claim_queued=configured,
                        model_factory=lambda request: model_for_run(args.db_path, request),
                    )
            except (ValueError, ServiceError, SettingsError):
                # Invalid or absent settings leave new runs queued; no secrets in logs.
                pass
            if args.once:
                break
            if not worked:
                stop.wait(1)
    finally:
        store.remove_worker()
        store.close()


if __name__ == "__main__":
    main()
