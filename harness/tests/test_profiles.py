import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from agent_harness import main, profiles
from agent_harness.clients import ServiceError
from agent_harness.config import model_configuration, web_from_settings
from agent_harness.profiles import ProfileRequest, ProfileStore, model_for_run
from agent_harness.settings import SettingsError, SettingsStore
from agent_harness.store import RunStore

SECRET = "dummy-profile-secret-never-live"


def profile(**changes):
    return ProfileRequest.model_validate(
        {
            "name": "Claude experiment",
            "harness": "claude-code",
            "model": "example-model",
            "base_url": "https://example.test/v1",
            "api_key": SECRET,
            "auth_mode": "api-key",
            **changes,
        }
    )


def test_named_profiles_encrypt_keys_and_keep_only_public_metadata_after_restart(tmp_path):
    path = tmp_path / "db.sqlite3"
    with ProfileStore(path) as store:
        created = store.save_profile(profile())
        assert created["api_key_set"]
        assert SECRET not in json.dumps(created)
    assert SECRET.encode() not in path.read_bytes()
    with ProfileStore(path) as store:
        assert store.private_profile(created["id"])["api_key"] == SECRET
        store.save_profile(profile(id=created["id"], api_key="", model="another-model"))
        assert store.private_profile(created["id"])["api_key"] == SECRET
        store.remove_profile(created["id"])
        deleted = store.private_profile(created["id"])
        assert not deleted["enabled"] and not deleted["api_key_set"]
        assert deleted["api_key"] is None


@pytest.mark.parametrize(
    "changes",
    [
        {"base_url": "https://user:secret@example.test/v1"},
        {"base_url": "http://remote.example.test"},
        {"base_url": "https://example.test?key=secret"},
        {"harness": "codex", "auth_mode": "api-key"},
        {"harness": "other"},
        {"model": "$(touch /tmp/untrusted)"},
        {"api_key": "bad\nheader"},
        {"auth_mode": "environment", "api_key": None},
        {"auth_mode": "environment", "base_url": None},
    ],
)
def test_profiles_reject_unsupported_runtime_and_unsafe_credentials(changes):
    with pytest.raises(ValidationError):
        profile(**changes)


def test_codex_profile_uses_sign_in_without_a_portable_token(tmp_path):
    with ProfileStore(tmp_path / "db") as store:
        value = store.save_profile(
            profile(
                harness="codex",
                auth_mode="codex-oauth",
                api_key=None,
                base_url=None,
            )
        )
        assert not value["api_key_set"]
        assert store.private_profile(value["id"])["api_key"] is None


def test_switching_provider_requires_new_credentials_without_reusing_stored_key(tmp_path):
    with ProfileStore(tmp_path / "db") as store:
        original = store.save_profile(
            profile(
                harness="nemoclaw",
                base_url="https://api.inference.crusoecloud.com/v1",
            )
        )
        with pytest.raises(SettingsError, match="new API key"):
            store.save_profile(
                profile(
                    id=original["id"],
                    harness="nemoclaw",
                    api_key="",
                    base_url="https://api.openai.com/v1",
                )
            )
        # Rejected edits retain the original endpoint and credential atomically.
        unchanged = store.private_profile(original["id"])
        assert unchanged["base_url"] == "https://api.inference.crusoecloud.com/v1"
        assert unchanged["api_key"] == SECRET
        updated = store.save_profile(
            profile(
                id=original["id"],
                harness="nemoclaw",
                api_key="dummy-openai-key",
                base_url="https://api.openai.com/v1",
            )
        )
        assert updated["base_url"] == "https://api.openai.com/v1"
        assert store.private_profile(original["id"])["api_key"] == "dummy-openai-key"


def test_switching_harness_cannot_reuse_key_for_a_different_default_endpoint(tmp_path):
    with ProfileStore(tmp_path / "db") as store:
        original = store.save_profile(profile(harness="nemoclaw", base_url=None))
        with pytest.raises(SettingsError, match="new API key"):
            store.save_profile(
                profile(id=original["id"], harness="claude-code", base_url=None, api_key="")
            )
        unchanged = store.private_profile(original["id"])
        assert unchanged["harness"] == "nemoclaw"
        assert unchanged["base_url"] is None
        assert unchanged["api_key"] == SECRET


@pytest.mark.parametrize("api_key", [None, "", "   "])
def test_api_key_profile_cannot_fall_back_to_worker_credentials(tmp_path, api_key):
    with ProfileStore(tmp_path / "db") as store:
        with pytest.raises(SettingsError, match="Enter an API key"):
            store.save_profile(profile(harness="nemoclaw", api_key=api_key))
        assert not store.list_profiles()


def test_legacy_keyless_profile_cannot_fall_back_to_worker_credentials(tmp_path, monkeypatch):
    path = tmp_path / "db"
    with ProfileStore(path) as store:
        created = store.save_profile(profile(harness="nemoclaw"))
        store.db.execute("UPDATE agent_model_profiles SET encrypted_key=NULL")
        store.db.commit()
    monkeypatch.setenv("HARNESS_NEMOCLAW_API_KEY", "dummy-worker-key")
    with pytest.raises(ServiceError, match="model_profile_unavailable"):
        model_for_run(path, {"harness": "nemoclaw", "model_profile_id": created["id"]})


def test_legacy_environment_profile_cannot_override_worker_endpoint(tmp_path, monkeypatch):
    from agent_harness import cli_models

    path = tmp_path / "db"
    with ProfileStore(path) as store:
        created = store.save_profile(
            profile(harness="nemoclaw", auth_mode="environment", base_url=None, api_key=None)
        )
        # Older versions allowed endpoint overrides on environment-authenticated profiles.
        legacy = {**created, "base_url": "https://custom.example/v1"}
        store.db.execute("UPDATE agent_model_profiles SET public=?", (json.dumps(legacy),))
        store.db.commit()
    monkeypatch.setattr(cli_models, "NemoClawModel", lambda **kwargs: kwargs)
    configuration = model_for_run(path, {"harness": "nemoclaw", "model_profile_id": created["id"]})
    assert configuration == {"model": "example-model", "base_url": None, "api_key": None}


def test_profile_must_match_run_harness_and_still_be_enabled(tmp_path):
    path = tmp_path / "db"
    with ProfileStore(path) as store:
        value = store.save_profile(profile())
    with pytest.raises(ServiceError, match="model_profile_unavailable"):
        model_for_run(path, {"harness": "codex", "model_profile_id": value["id"]})
    with ProfileStore(path) as store:
        store.remove_profile(value["id"])
    with pytest.raises(ServiceError, match="model_profile_unavailable"):
        model_for_run(path, {"harness": "claude-code", "model_profile_id": value["id"]})


def test_saved_provider_configuration_is_loaded_fresh_without_worker_restart(tmp_path, monkeypatch):
    path = tmp_path / "db"
    monkeypatch.delenv("HARNESS_MODEL_BASE_URL", raising=False)
    with SettingsStore(path) as settings:
        settings.save("crusoe", SECRET, "model-a")
    assert model_configuration(path)["model"] == "model-a"
    with SettingsStore(path) as settings:
        settings.save("crusoe", model="model-b")
    assert model_configuration(path)["model"] == "model-b"
    with SettingsStore(path) as settings:
        settings.save("you", SECRET)
    with web_from_settings(path) as web:
        assert web.provider == "you"
    with SettingsStore(path) as settings:
        settings.remove("crusoe")
    assert model_configuration(path) is None


def test_keyed_model_profile_works_without_a_default_provider(tmp_path, monkeypatch):
    path = tmp_path / "db"
    monkeypatch.setattr(profiles, "model_configuration", lambda _path: None)
    monkeypatch.setattr(profiles, "runtime_catalog", lambda: [])
    with ProfileStore(path) as store:
        created = store.save_profile(profile(harness="builtin"))
    catalog = profiles.harness_catalog(path)
    assert catalog["harnesses"][0]["available"]
    assert not catalog["harnesses"][0]["models"]
    with model_for_run(
        path,
        {
            "harness": "builtin",
            "model_profile_id": created["id"],
            "model": "example-model",
        },
    ) as model:
        assert model.model == "example-model"
        assert model.api_key == SECRET
        assert model.base_url == "https://example.test/v1"


def test_settings_profile_routes_require_auth_and_never_return_keys(tmp_path, monkeypatch):
    path = tmp_path / "db"
    monkeypatch.setenv("HARNESS_DB_PATH", str(path))
    monkeypatch.setenv("HARNESS_API_KEY", "dummy-service-auth")
    monkeypatch.setattr(profiles, "runtime_catalog", lambda: [])
    headers = {"Authorization": "Bearer dummy-service-auth"}
    with TestClient(main.app) as client:
        assert client.get("/settings/harnesses").status_code == 401
        data = profile().model_dump(mode="json")
        data["api_key"] = SECRET
        result = client.post("/settings/harnesses/profiles", json=data, headers=headers)
        assert result.status_code == 200
        assert result.headers["cache-control"] == "no-store"
        assert SECRET not in result.text
        identifier = result.json()["profiles"][0]["id"]
        removed = client.post(f"/settings/harnesses/profiles/{identifier}/remove", headers=headers)
        assert removed.status_code == 200
        assert not removed.json()["profiles"][0]["enabled"]


def test_profile_public_fields_cannot_reflect_its_key(tmp_path):
    with ProfileStore(tmp_path / "db") as store:
        with pytest.raises(SettingsError):
            store.save_profile(profile(name=SECRET))
        with pytest.raises(SettingsError):
            store.private_profile(uuid4())


def test_waiting_workers_can_reconcile_without_claiming_new_runs(tmp_path):
    store = RunStore(tmp_path / "db")
    try:
        assert store.claim(include_queued=False) is None
    finally:
        store.close()
