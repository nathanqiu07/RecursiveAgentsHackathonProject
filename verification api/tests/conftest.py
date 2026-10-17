import pytest
from fastapi.testclient import TestClient

from verification_api.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("VERIFIER_DB_PATH", str(tmp_path / "jobs.sqlite3"))
    monkeypatch.delenv("VERIFIER_API_KEY", raising=False)
    with TestClient(app) as client:
        yield client
