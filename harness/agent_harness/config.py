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
