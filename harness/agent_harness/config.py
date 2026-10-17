import os
from pathlib import Path

from .clients import ChatModel, Services, WebResearch


def database_path():
    return os.environ.get(
        "HARNESS_DB_PATH", str(Path(__file__).resolve().parent.parent / "data" / "harness.sqlite3")
    )


def services_from_env():
    return Services(
        library_url=os.environ.get("HARNESS_LIBRARY_URL", "http://127.0.0.1:8001"),
        benchmark_url=os.environ.get("HARNESS_BENCHMARK_URL", "http://127.0.0.1:3001"),
        manager_url=os.environ.get("HARNESS_RESOURCE_MANAGER_URL", "http://127.0.0.1:8002"),
        library_key=os.environ.get("HARNESS_LIBRARY_API_KEY") or None,
        benchmark_key=os.environ.get("HARNESS_BENCHMARK_API_KEY") or None,
        manager_key=os.environ.get("HARNESS_RESOURCE_MANAGER_API_KEY") or None,
    )


def model_from_env():
    base_url, model = os.environ.get("HARNESS_MODEL_BASE_URL"), os.environ.get("HARNESS_MODEL")
    if not base_url or not model:
        raise ValueError("Configure HARNESS_MODEL_BASE_URL and HARNESS_MODEL")
    return ChatModel(base_url, model, os.environ.get("HARNESS_MODEL_API_KEY") or None)


def web_from_env():
    return WebResearch(
        os.environ.get("HARNESS_SEARCH_URL") or None,
        os.environ.get("HARNESS_SEARCH_API_KEY") or None,
    )


def model_configuration(db_path):
    """Snapshot credentials between runs; an active run keeps its own clients."""
    from .settings import SettingsStore

    with SettingsStore(db_path) as settings:
        crusoe = settings.get_private("crusoe")
    if crusoe.get("enabled") and crusoe.get("api_key") and crusoe.get("model"):
        return {
            "base_url": "https://api.inference.crusoecloud.com/v1",
            "model": crusoe["model"],
            "api_key": crusoe["api_key"],
        }
    # A saved disconnect also overrides any Crusoe environment credentials.
    base = os.environ.get("HARNESS_MODEL_BASE_URL", "").rstrip("/")
    model = os.environ.get("HARNESS_MODEL")
    if base and model and base != "https://api.inference.crusoecloud.com/v1":
        return {
            "base_url": base,
            "model": model,
            "api_key": os.environ.get("HARNESS_MODEL_API_KEY") or None,
        }
    return None


def model_from_settings(db_path):
    configuration = model_configuration(db_path)
    return ChatModel(**configuration) if configuration else None


def web_from_settings(db_path):
    from .settings import SettingsStore

    with SettingsStore(db_path) as settings:
        you = settings.get_private("you")
    if you.get("enabled") and you.get("api_key"):
        return WebResearch(search_key=you["api_key"], provider="you")
    return web_from_env()
