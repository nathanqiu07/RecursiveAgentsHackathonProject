"""Start any number of workers; a run's lease gives it one orchestrator and sandbox."""

import argparse
import signal
import threading
from uuid import uuid4

from .config import database_path, model_from_env, services_from_env, web_from_env
from .engine import run_once
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
    try:
        model = model_from_env()
    except ValueError as exc:
        parser.error(str(exc))
    store = RunStore(args.db_path)
    store.worker_id = str(uuid4())
    try:
        with model, services_from_env() as services, web_from_env() as web:
            while not stop.is_set():
                store.touch_worker()
                worked = run_once(store, services, model, web, stop)
                if args.once:
                    break
                if not worked:
                    stop.wait(1)
    finally:
        store.remove_worker()
        store.close()


if __name__ == "__main__":
    main()
