"""Named model choices for each harness, with backend-only encrypted credentials."""

import importlib
import json
import os
import threading
import time
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from .clients import ChatModel, ServiceError
from .config import model_configuration
from .settings import SettingsError, SettingsStore

HARNESSES = ("builtin", "codex", "claude-code", "nemoclaw")


class ProfileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: UUID | None = None
    name: str = Field(min_length=1, max_length=100)
    harness: str
    model: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9._/:-]+$")
    base_url: str | None = Field(default=None, max_length=2048)
    api_key: SecretStr | None = None
    auth_mode: str

    @field_validator("harness")
    @classmethod
    def harness_known(cls, value):
        if value not in HARNESSES:
            raise ValueError("Unknown harness")
        return value

    @field_validator("api_key")
    @classmethod
    def valid_key(cls, value):
        if value is not None:
            raw = value.get_secret_value()
            if len(raw) > 4096 or any(not 32 <= ord(char) <= 126 for char in raw):
                raise ValueError("Invalid API key")
        return value

    @field_validator("base_url")
    @classmethod
    def valid_endpoint(cls, value):
        if not value:
            return None
        try:
            url = urlsplit(value)
            _ = url.port
            if (
                url.scheme not in {"http", "https"}
                or not url.hostname
                or (url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1"})
                or url.username
                or url.password
                or url.query
                or url.fragment
                or "\\" in value
                or any(ord(char) <= 32 for char in value)
            ):
                raise ValueError
        except ValueError:
            raise ValueError(
                "Use an HTTPS API base URL without credentials or query parameters"
            ) from None
        return value.rstrip("/")

    @model_validator(mode="after")
    def authentication(self):
        expected = {"codex-oauth"} if self.harness == "codex" else {"api-key", "environment"}
        if self.auth_mode not in expected:
            raise ValueError("Choose an authentication method supported by this harness")
        if self.harness == "codex" and (self.base_url or self.api_key):
            raise ValueError("Codex OAuth uses the local Codex sign-in and its own endpoint")
        if self.auth_mode == "environment" and (self.base_url or self.api_key):
            raise ValueError(
                "Backend environment authentication uses the worker's endpoint and key"
            )
        return self


class ProfileStore(SettingsStore):
    def __init__(self, db_path):
        super().__init__(db_path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS agent_model_profiles (
            id TEXT PRIMARY KEY, public TEXT NOT NULL, encrypted_key TEXT
        )""")

    def list_profiles(self):
        return [
            json.loads(row[0])
            for row in self.db.execute("SELECT public FROM agent_model_profiles ORDER BY rowid")
        ]

    def private_profile(self, identifier):
        row = self.db.execute(
            "SELECT public,encrypted_key FROM agent_model_profiles WHERE id=?", (str(identifier),)
        ).fetchone()
        if row is None:
            raise SettingsError("Model profile was not found")
        value = json.loads(row["public"])
        value["api_key"] = None
        if row["encrypted_key"]:
            try:
                value["api_key"] = self._fernet().decrypt(row["encrypted_key"].encode()).decode()
            except Exception:
                raise SettingsError("Saved profile credentials could not be decrypted") from None
        return value

    def save_profile(self, body):
        identifier = str(body.id or uuid4())
        with self._transaction():
            previous = self.private_profile(identifier) if body.id else None
            key = body.api_key.get_secret_value().strip() if body.api_key else None
            if (
                body.auth_mode == "api-key"
                and previous
                and previous["api_key"]
                and not key
                and (previous["base_url"] != body.base_url or previous["harness"] != body.harness)
            ):
                raise SettingsError(
                    "Enter a new API key when changing the profile endpoint or harness"
                )
            key = key or (previous and previous["api_key"])
            if body.auth_mode == "api-key" and not key:
                raise SettingsError("Enter an API key for this model profile")
            if body.auth_mode != "api-key":
                key = None
            fields = (body.name, body.model, body.base_url or "")
            if key and any(key in value for value in fields):
                raise SettingsError("Public profile fields must not contain credentials")
            value = {
                "id": identifier,
                "name": body.name.strip(),
                "harness": body.harness,
                "model": body.model,
                "base_url": body.base_url,
                "auth_mode": body.auth_mode,
                "api_key_set": bool(key),
                "enabled": True,
            }
            encrypted = self._fernet().encrypt(key.encode()).decode() if key else None
            self.db.execute(
                "INSERT INTO agent_model_profiles VALUES (?,?,?) ON CONFLICT(id) DO UPDATE "
                "SET public=excluded.public,encrypted_key=excluded.encrypted_key",
                (identifier, json.dumps(value), encrypted),
            )
        return value

    def remove_profile(self, identifier):
        with self._transaction():
            value = self.private_profile(identifier)
            value.pop("api_key", None)
            value.update(enabled=False, api_key_set=False)
            self.db.execute(
                "UPDATE agent_model_profiles SET public=?,encrypted_key=NULL WHERE id=?",
                (json.dumps(value), str(identifier)),
            )


_catalog_lock = threading.Lock()
_catalog_cache = None


def runtime_catalog(refresh=False):
    global _catalog_cache
    with _catalog_lock:
        if not refresh and _catalog_cache and time.monotonic() - _catalog_cache[0] < 30:
            return _catalog_cache[1]
        entries = []
        for identifier, label, module, status_function in (
            ("codex", "Codex", "codex_model", "codex_status"),
            ("claude-code", "Claude Code", "cli_models", "claude_code_status"),
            ("nemoclaw", "NVIDIA NemoClaw", "cli_models", "nemoclaw_status"),
        ):
            try:
                adapter = importlib.import_module(f".{module}", __package__)
                status = getattr(adapter, status_function)()
                models = (
                    adapter.codex_models() if identifier == "codex" and status["available"] else []
                )
                entry = {
                    "id": identifier,
                    "label": label,
                    "available": bool(status["available"]),
                    "message": status["message"],
                    "models": models,
                }
            except Exception:
                entry = {
                    "id": identifier,
                    "label": label,
                    "available": False,
                    "models": [],
                    "message": "Runtime check failed. Verify its installation and sign-in.",
                }
            entries.append(entry)
        _catalog_cache = time.monotonic(), entries
        return entries


def harness_catalog(db_path):
    configuration = model_configuration(db_path)
    with ProfileStore(db_path) as store:
        profiles = store.list_profiles()
    keyed_profiles = any(
        profile["enabled"] and profile["harness"] == "builtin" and profile["api_key_set"]
        for profile in profiles
    )
    builtin = {
        "id": "builtin",
        "label": "Research harness",
        "available": bool(configuration) or keyed_profiles,
        "message": "Uses the saved Crusoe or environment model connection."
        if configuration
        else "Select a saved model profile, or configure a default Crusoe connection.",
        "models": [{"id": configuration["model"]}] if configuration else [],
    }
    return {
        "default_harness": "codex",
        "harnesses": [builtin, *runtime_catalog()],
        "profiles": profiles,
    }


def model_for_run(db_path, request):
    harness = request.get("harness", "builtin")
    identifier = request.get("model_profile_id")
    profile = None
    if identifier:
        with ProfileStore(db_path) as store:
            profile = store.private_profile(identifier)
        if not profile["enabled"] or profile["harness"] != harness:
            raise ServiceError("model_profile_unavailable")
        if profile["auth_mode"] == "api-key" and not profile["api_key"]:
            raise ServiceError("model_profile_unavailable")
    model = request.get("model") or (profile["model"] if profile else None)
    if harness == "builtin":
        if profile and profile["auth_mode"] == "api-key" and profile["api_key"]:
            configuration = {
                "base_url": profile["base_url"]
                or os.environ.get("HARNESS_MODEL_BASE_URL")
                or "https://api.inference.crusoecloud.com/v1",
                "model": model,
                "api_key": profile["api_key"],
            }
        else:
            configuration = model_configuration(db_path)
        if not configuration:
            raise ServiceError("model_not_configured")
        if profile and profile["auth_mode"] == "api-key":
            configuration.update(
                base_url=profile["base_url"] or configuration["base_url"],
                api_key=profile["api_key"],
            )
        if model:
            configuration["model"] = model
        return ChatModel(**configuration)
    if harness == "codex":
        from .codex_model import CodexModel

        return CodexModel(model=model)
    from .cli_models import ClaudeCodeModel, NemoClawModel

    if not profile:
        raise ServiceError("model_profile_required")
    adapter = ClaudeCodeModel if harness == "claude-code" else NemoClawModel
    return adapter(
        model=model,
        base_url=profile["base_url"] if profile["auth_mode"] == "api-key" else None,
        api_key=profile["api_key"] if profile["auth_mode"] == "api-key" else None,
    )
