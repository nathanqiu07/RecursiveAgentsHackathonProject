import json
import os
import stat

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from agent_harness import main
from agent_harness.clients import ServiceError
from agent_harness.settings import CRUSOE_BASE_URL, SettingsError, SettingsStore

KEY = "dummy-crusoe-credential-for-unit-test"
YOU_KEY = "dummy-you-credential-for-unit-test"
MODEL = "vendor/test-model"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in (
        "HARNESS_SETTINGS_KEY",
        "HARNESS_MODEL_BASE_URL",
        "HARNESS_MODEL_API_KEY",
        "HARNESS_MODEL",
        "CRUSOE_INFERENCE_API_KEY",
        "CRUSOE_MODEL",
        "HARNESS_YOU_API_KEY",
        "YDC_API_KEY",
        "HARNESS_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = tmp_path / "harness.sqlite3"
    monkeypatch.setenv("HARNESS_DB_PATH", str(path))
    return path


@pytest.fixture
def client(db_path):
    with TestClient(main.app) as connection:
        yield connection


def test_credentials_are_encrypted_on_disk_and_never_returned(db_path):
    with SettingsStore(db_path) as store:
        public = store.save("crusoe", KEY, MODEL)
        assert store.get_private("crusoe")["api_key"] == KEY
        assert public["providers"]["crusoe"]["configured"]
        assert public["revision"] == 1
        assert KEY not in json.dumps(public)
        encrypted = store.db.execute(
            "SELECT encrypted_key FROM agent_provider_settings WHERE provider = 'crusoe'"
        ).fetchone()[0]
        assert KEY not in encrypted
        for name in (str(db_path), str(db_path) + "-wal"):
            if os.path.exists(name):
                assert KEY.encode() not in open(name, "rb").read()
    key_file = str(db_path) + ".settings.key"
    assert stat.S_IMODE(os.stat(key_file).st_mode) == 0o600
    with SettingsStore(db_path) as reopened:
        assert reopened.get_private("crusoe")["api_key"] == KEY


def test_blank_key_retains_secret_and_model_edits_invalidate_checks(db_path):
    with SettingsStore(db_path, now=lambda: 123) as store:
        store.save("crusoe", KEY, MODEL)
        assert store.record_check("crusoe", True, "Connection verified")
        assert store.get_public()["providers"]["crusoe"]["check"] == {
            "ok": True,
            "message": "Connection verified",
            "checked_at": 123,
        }
        public = store.save("crusoe", "", "vendor/new-model")
        assert store.get_private("crusoe")["api_key"] == KEY
        assert public["providers"]["crusoe"]["model"] == "vendor/new-model"
        assert public["providers"]["crusoe"]["check"] is None
        assert public["revision"] == 2


def test_environment_fallback_and_disconnect_override(monkeypatch, db_path):
    monkeypatch.setenv("HARNESS_MODEL_BASE_URL", CRUSOE_BASE_URL + "/")
    monkeypatch.setenv("HARNESS_MODEL_API_KEY", KEY)
    monkeypatch.setenv("HARNESS_MODEL", MODEL)
    monkeypatch.setenv("YDC_API_KEY", YOU_KEY)
    with SettingsStore(db_path) as store:
        assert store.get_public()["providers"]["crusoe"]["source"] == "environment"
        assert store.get_public()["providers"]["you"]["configured"]
        store.remove("crusoe")
        assert store.get_private("crusoe") == {
            "enabled": False,
            "api_key": None,
            "model": None,
            "source": "saved",
        }
        assert not store.get_public()["providers"]["crusoe"]["configured"]
        store.save("you", "")
        assert store.get_private("you")["api_key"] == YOU_KEY
        assert store.get_public()["providers"]["you"]["source"] == "saved"


def test_generic_environment_model_is_not_labeled_crusoe(monkeypatch, db_path):
    monkeypatch.setenv("HARNESS_MODEL_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("HARNESS_MODEL_API_KEY", KEY)
    monkeypatch.setenv("HARNESS_MODEL", MODEL)
    with SettingsStore(db_path) as store:
        assert store.get_private("crusoe")["source"] == "none"
        assert not store.get_public()["providers"]["crusoe"]["configured"]


def test_environment_changes_and_racing_save_do_not_reuse_a_stale_check(monkeypatch, db_path):
    monkeypatch.setenv("YDC_API_KEY", YOU_KEY)
    with SettingsStore(db_path) as store:
        snapshot = store.get_private("you")
        store.record_check("you", True, "Search verified", expected=snapshot)
        monkeypatch.setenv("YDC_API_KEY", "new-dummy-key")
        assert store.get_public()["providers"]["you"]["check"] is None
        assert not store.record_check("you", True, "Stale test", expected=snapshot)
        store.save("you", "third-dummy-key")
        assert not store.record_check("you", True, "Stale test", expected=snapshot)
        assert store.get_public()["providers"]["you"]["check"] is None


def test_wrong_master_key_has_a_fixed_safe_error(monkeypatch, db_path):
    with SettingsStore(db_path) as store:
        store.save("crusoe", KEY, MODEL)
    monkeypatch.setenv("HARNESS_SETTINGS_KEY", Fernet.generate_key().decode())
    with SettingsStore(db_path) as store:
        with pytest.raises(SettingsError, match="could not be decrypted") as error:
            store.get_public()
        assert KEY not in str(error.value)


def test_environment_master_key_does_not_create_a_key_file(monkeypatch, db_path):
    monkeypatch.setenv("HARNESS_SETTINGS_KEY", Fernet.generate_key().decode())
    with SettingsStore(db_path) as store:
        store.save("you", YOU_KEY)
    assert not os.path.exists(str(db_path) + ".settings.key")


def test_provider_validation_and_redacted_check_messages(db_path):
    with SettingsStore(db_path) as store:
        with pytest.raises(SettingsError, match="Unknown provider"):
            store.get_private("unknown")
        with pytest.raises(SettingsError, match="API key must"):
            store.save("you", "do-not-reflect\ncredential")
        with pytest.raises(SettingsError, match="Model ID must"):
            store.save("crusoe", KEY, "bad model")
        with pytest.raises(SettingsError, match="Model ID must"):
            store.save("crusoe", KEY, KEY)
        store.save("you", YOU_KEY)
        store.record_check("you", False, "Error for " + YOU_KEY)
        assert YOU_KEY not in json.dumps(store.get_public())
        assert "[redacted]" in store.get_public()["providers"]["you"]["check"]["message"]


def test_provider_api_does_not_echo_keys_and_preserves_blank_edits(client):
    response = client.post("/settings/providers/crusoe", json={"api_key": KEY, "model": MODEL})
    assert response.status_code == 200
    assert KEY not in response.text
    assert response.headers["cache-control"] == "no-store"
    response = client.post("/settings/providers/crusoe", json={"api_key": "", "model": "other/id"})
    assert response.json()["providers"]["crusoe"]["api_key_set"]
    assert client.get("/settings/providers").json()["providers"]["crusoe"]["model"] == "other/id"
    response = client.post("/settings/providers/crusoe/remove")
    assert not response.json()["providers"]["crusoe"]["enabled"]


@pytest.mark.parametrize(
    "body",
    [
        {"api_key": "x" * 4097},
        {"api_key": {"secret": KEY}},
        {"other": KEY},
        {"api_key": KEY + "\n"},
        {"model": "invalid model " + KEY},
    ],
)
def test_invalid_settings_body_does_not_echo_secret(client, body):
    response = client.post("/settings/providers/crusoe", json=body)
    assert response.status_code in {400, 422}
    assert KEY not in response.text


def test_provider_routes_reject_unknown_providers_and_you_model(client):
    assert client.post("/settings/providers/unknown", json={"api_key": KEY}).status_code == 404
    assert client.post("/settings/providers/you", json={"model": MODEL}).status_code == 400


def test_models_and_checks_use_only_saved_private_config(client, monkeypatch):
    assert client.get("/settings/providers/crusoe/models").status_code == 409
    client.post("/settings/providers/crusoe", json={"api_key": KEY, "model": MODEL})
    calls = []

    def models(key):
        calls.append(("models", key))
        return [{"id": MODEL, "supported_parameters": ["tools", "response_format"]}]

    def check(provider, key, model=None):
        calls.append((provider, key, model))
        return {"ok": True, "message": "Model tools and structured output verified"}

    monkeypatch.setattr(main, "provider_models", models)
    monkeypatch.setattr(main, "check_provider", check)
    assert client.get("/settings/providers/crusoe/models").json()["models"][0]["id"] == MODEL
    response = client.post("/settings/providers/crusoe/test")
    assert response.status_code == 200
    assert response.json()["providers"]["crusoe"]["check"]["ok"]
    assert KEY not in response.text
    assert calls == [("models", KEY), ("crusoe", KEY, MODEL)]


def test_check_failures_are_fixed_safe_and_disabled_provider_cannot_be_tested(client, monkeypatch):
    assert client.post("/settings/providers/you/test").status_code == 409
    client.post("/settings/providers/you", json={"api_key": YOU_KEY})

    def failed(*_args):
        raise ServiceError("upstream_failure_" + YOU_KEY)

    monkeypatch.setattr(main, "check_provider", failed)
    response = client.post("/settings/providers/you/test")
    assert response.status_code == 200
    assert not response.json()["providers"]["you"]["check"]["ok"]
    assert YOU_KEY not in response.text


def test_check_does_not_mark_new_credentials_as_verified(client, monkeypatch, db_path):
    client.post("/settings/providers/you", json={"api_key": YOU_KEY})

    def racing(*_args):
        with SettingsStore(db_path) as other:
            other.save("you", "replacement-dummy-key")
        return {"ok": True, "message": "Old credentials verified"}

    monkeypatch.setattr(main, "check_provider", racing)
    response = client.post("/settings/providers/you/test")
    assert response.status_code == 409
    assert client.get("/settings/providers").json()["providers"]["you"]["check"] is None


def test_settings_auth_covers_all_reads_and_writes(monkeypatch, db_path):
    monkeypatch.setenv("HARNESS_API_KEY", "dummy-harness-key")
    with TestClient(main.app) as client:
        for method, path, body in (
            ("GET", "/settings/providers", None),
            ("GET", "/settings/providers/crusoe/models", None),
            ("GET", "/settings/readiness", None),
            ("POST", "/settings/providers/you", {"api_key": YOU_KEY}),
            ("POST", "/settings/providers/you/test", None),
            ("POST", "/settings/providers/you/remove", None),
        ):
            assert client.request(method, path, json=body).status_code == 401
        response = client.get(
            "/settings/providers", headers={"Authorization": "Bearer dummy-harness-key"}
        )
        assert response.status_code == 200


def test_browser_cannot_directly_mutate_harness_settings(client):
    client.post("/settings/providers/you", json={"api_key": YOU_KEY})
    response = client.post(
        "/settings/providers/you/remove", headers={"Origin": "https://attacker.test"}
    )
    assert response.status_code == 403
    assert client.get("/settings/providers").json()["providers"]["you"]["configured"]
