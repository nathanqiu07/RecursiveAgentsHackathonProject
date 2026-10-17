"""Read-only dependency checks; never make paid inference/search calls on polling."""

from concurrent.futures import ThreadPoolExecutor

import httpx

from .clients import ServiceError
from .config import model_configuration, services_from_env
from .profiles import runtime_catalog
from .settings import SettingsError, SettingsStore
from .store import RunStore


def eligible_pool(pool):
    def number(name, minimum):
        value = pool.get(name)
        return type(value) in (int, float) and value >= minimum

    return (
        isinstance(pool, dict)
        and pool.get("enabled") is not False
        and isinstance(pool.get("node_id"), str)
        and bool(pool["node_id"])
        and number("cpus", 0.001)
        and number("memory_mb", 64)
        and number("scratch_mb", 16)
        and number("max_timeout_seconds", 30)
    )


def dependency_check(kind):
    # Each probe gets its own credential-owning connection, a short socket timeout,
    # and a bounded body. Reads have no publication/allocation side effects.
    import time

    options = {
        "extensions": {"timeout": httpx.Timeout(3).as_dict()},
        "maximum": 524288,
        "deadline": time.monotonic() + 5,
    }
    try:
        with services_from_env() as services:
            if kind == "resources":
                value = services._manager("GET", "/resources", **options)
                pools = value.get("resources")
                if not isinstance(pools, list):
                    raise ServiceError("invalid_response")
                ok = any(eligible_pool(pool) for pool in pools)
                return ok, (
                    "An eligible sandbox pool is available."
                    if ok
                    else "ResourceManager is reachable; enable a registered sandbox pool."
                )
            if kind == "library":
                value = services._library("GET", "/papers?limit=1&offset=0", **options)
                ok = isinstance(value.get("papers"), list) and type(value.get("total")) is int
            else:
                value = services._benchmark("GET", "/benchmarks", **options)
                ok = isinstance(value.get("benchmarks"), list)
            if not ok:
                raise ServiceError("invalid_response")
            return True, "Service is reachable and authenticated."
    except (ServiceError, ValueError):
        return False, "Service is unavailable or authentication failed. Check its server settings."


def build_readiness(db_path):
    checks = []

    def add(identifier, label, ok, message, required=True):
        checks.append(
            {
                "id": identifier,
                "label": label,
                "ok": bool(ok),
                "required": required,
                "message": message,
            }
        )

    try:
        with SettingsStore(db_path) as settings:
            providers = settings.get_public()["providers"]
        configuration = model_configuration(db_path)
        crusoe = providers["crusoe"]
        is_crusoe = configuration and configuration["base_url"] == (
            "https://api.inference.crusoecloud.com/v1"
        )
        checked = bool(crusoe.get("check") and crusoe["check"].get("ok"))
        codex = next(item for item in runtime_catalog() if item["id"] == "codex")
        model_ready = codex["available"] or (bool(configuration) and (not is_crusoe or checked))
        add(
            "model",
            "Model connection",
            model_ready,
            "Codex is signed in locally. Model access is verified when a turn completes."
            if codex["available"]
            else "Crusoe tool calling and JSON responses passed the connection test."
            if is_crusoe and checked
            else "Configured through the environment; provider compatibility is not checked here."
            if configuration and not is_crusoe
            else "Save a Crusoe API key and model, then run its connection test.",
        )
        you = providers["you"]
        add(
            "search",
            "You.com search",
            True,
            "You.com MCP uses the saved server key; live tools are checked when research starts."
            if you.get("configured")
            else (
                "You.com MCP free profile is available without a key (100 searches/day); "
                "paper reads use public HTTPS. Live access is checked when research starts."
            ),
            False,
        )
    except (ValueError, ServiceError, SettingsError):
        add("model", "Model connection", False, "Provider settings could not be loaded.")
        add("search", "You.com search", False, "Provider settings could not be loaded.", False)

    store = RunStore(db_path)
    try:
        online = store.status()["workers_online"]
    finally:
        store.close()
    add(
        "worker",
        "Agent worker",
        online > 0,
        f"{online} worker(s) online." if online else "Start python -m agent_harness.worker.",
    )
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(dependency_check, ("resources", "library", "benchmark")))
    for identifier, label, (ok, message) in zip(
        ("resources", "library", "benchmark"),
        ("Sandbox capacity", "Paper Library", "Benchmark API"),
        results,
    ):
        add(identifier, label, ok, message)
    return {"ready": all(item["ok"] for item in checks if item["required"]), "checks": checks}
