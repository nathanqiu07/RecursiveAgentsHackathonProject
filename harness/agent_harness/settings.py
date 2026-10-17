"""Backend-only provider credentials, encrypted at rest and never returned by the API."""

import hashlib
import json
import os
import sqlite3
import stat
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

CRUSOE_BASE_URL = "https://api.inference.crusoecloud.com/v1"
PROVIDERS = {"crusoe", "you"}


class SettingsError(Exception):
    """Safe, fixed text suitable for a public error response."""


class SettingsStore:
    def __init__(self, db_path, now=time.time):
        self.path = str(db_path)
        self.now = now
        self._cipher = None
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS agent_provider_settings (
                provider TEXT PRIMARY KEY, enabled INTEGER NOT NULL,
                encrypted_key TEXT, model TEXT
            );
            CREATE TABLE IF NOT EXISTS agent_provider_checks (
                provider TEXT PRIMARY KEY, fingerprint TEXT NOT NULL,
                ok INTEGER NOT NULL, message TEXT NOT NULL, checked_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS agent_settings_revision (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1), revision INTEGER NOT NULL
            );
            INSERT OR IGNORE INTO agent_settings_revision VALUES (1, 0);
        """)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def close(self):
        self.db.close()

    @contextmanager
    def _transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    @staticmethod
    def _provider(provider):
        if provider not in PROVIDERS:
            raise SettingsError("Unknown provider")

    def _fernet(self):
        if self._cipher is not None:
            return self._cipher
        configured = os.environ.get("HARNESS_SETTINGS_KEY")
        try:
            if configured:
                key = configured.encode("ascii")
            elif self.path == ":memory:":
                key = Fernet.generate_key()
            else:
                target = Path(self.path + ".settings.key")
                if not target.exists():
                    # Link a fully written private file into place atomically. Another process
                    # can win creation without readers seeing an incomplete master key.
                    fd, temporary = tempfile.mkstemp(prefix=".settings-key-", dir=target.parent)
                    try:
                        with os.fdopen(fd, "wb") as output:
                            output.write(Fernet.generate_key())
                            output.flush()
                            os.fsync(output.fileno())
                        try:
                            os.link(temporary, target)
                        except FileExistsError:
                            pass
                    finally:
                        os.unlink(temporary)
                fd = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                try:
                    info = os.fstat(fd)
                    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                        raise SettingsError("Provider encryption key file is not private")
                    os.fchmod(fd, 0o600)
                    key = os.read(fd, 129)
                finally:
                    os.close(fd)
            self._cipher = Fernet(key)
        except (ValueError, UnicodeError, OSError):
            raise SettingsError("Provider encryption key is unavailable or invalid") from None
        return self._cipher

    @staticmethod
    def _environment(provider):
        if provider == "you":
            key = os.environ.get("HARNESS_YOU_API_KEY") or os.environ.get("YDC_API_KEY")
            return {
                "enabled": bool(key),
                "api_key": key,
                "model": None,
                "source": "environment" if key else "none",
            }
        base = os.environ.get("HARNESS_MODEL_BASE_URL", "").rstrip("/")
        key = os.environ.get("CRUSOE_INFERENCE_API_KEY")
        if not key and base == CRUSOE_BASE_URL:
            key = os.environ.get("HARNESS_MODEL_API_KEY")
        model = os.environ.get("HARNESS_MODEL") or os.environ.get("CRUSOE_MODEL")
        recognized = bool(key) or base == CRUSOE_BASE_URL
        return {
            "enabled": recognized,
            "api_key": key,
            "model": model if recognized else None,
            "source": "environment" if recognized else "none",
        }

    def get_private(self, provider):
        self._provider(provider)
        row = self.db.execute(
            "SELECT * FROM agent_provider_settings WHERE provider = ?", (provider,)
        ).fetchone()
        if row is None:
            return self._environment(provider)
        key = None
        if row["encrypted_key"]:
            try:
                key = self._fernet().decrypt(row["encrypted_key"].encode("ascii")).decode("utf-8")
            except (InvalidToken, ValueError, UnicodeError):
                raise SettingsError("Saved provider credentials could not be decrypted") from None
        return {
            "enabled": bool(row["enabled"]),
            "api_key": key,
            "model": row["model"],
            "source": "saved",
        }

    @staticmethod
    def _fingerprint(private):
        return hashlib.sha256(json.dumps(private, sort_keys=True).encode()).hexdigest()

    def get_public(self):
        providers = {}
        for name in ("crusoe", "you"):
            private = self.get_private(name)
            check = self.db.execute(
                "SELECT * FROM agent_provider_checks WHERE provider = ? AND fingerprint = ?",
                (name, self._fingerprint(private)),
            ).fetchone()
            configured = bool(private["enabled"] and private["api_key"])
            if name == "crusoe":
                configured = configured and bool(private["model"])
            providers[name] = {
                "configured": configured,
                "api_key_set": bool(private["api_key"]),
                "source": private["source"],
                "enabled": private["enabled"],
                "check": None
                if check is None
                else {
                    "ok": bool(check["ok"]),
                    "message": check["message"],
                    "checked_at": check["checked_at"],
                },
            }
            if name == "crusoe":
                providers[name]["model"] = private["model"]
        revision = self.db.execute(
            "SELECT revision FROM agent_settings_revision WHERE singleton = 1"
        ).fetchone()[0]
        return {"providers": providers, "revision": revision}

    def _changed(self, provider):
        self.db.execute("DELETE FROM agent_provider_checks WHERE provider = ?", (provider,))
        self.db.execute("UPDATE agent_settings_revision SET revision = revision + 1")

    def save(self, provider, api_key=None, model=None):
        self._provider(provider)
        if api_key is not None and (
            not isinstance(api_key, str)
            or len(api_key) > 4096
            or any(ord(char) < 32 or ord(char) > 126 for char in api_key)
        ):
            raise SettingsError("API key must contain at most 4096 printable characters")
        if model is not None and (
            not isinstance(model, str)
            or len(model) > 256
            or any(not (char.isascii() and (char.isalnum() or char in "-_./:")) for char in model)
        ):
            raise SettingsError("Model ID must contain at most 256 letters, digits or - _ . / :")
        with self._transaction():
            previous = self.get_private(provider)
            key = api_key.strip() if api_key and api_key.strip() else previous["api_key"]
            selected = model.strip() if model is not None else previous["model"]
            if key and selected and key in selected:
                raise SettingsError("Model ID must not contain an API key")
            encrypted = self._fernet().encrypt(key.encode()).decode() if key else None
            self.db.execute(
                """
                INSERT INTO agent_provider_settings (provider, enabled, encrypted_key, model)
                VALUES (?, 1, ?, ?)
                ON CONFLICT(provider) DO UPDATE SET enabled = 1,
                    encrypted_key = excluded.encrypted_key, model = excluded.model
            """,
                (provider, encrypted, selected if provider == "crusoe" else None),
            )
            self._changed(provider)
        return self.get_public()

    def remove(self, provider):
        self._provider(provider)
        with self._transaction():
            self.db.execute(
                """
                INSERT INTO agent_provider_settings (provider, enabled, encrypted_key, model)
                VALUES (?, 0, NULL, NULL)
                ON CONFLICT(provider) DO UPDATE SET enabled = 0, encrypted_key = NULL, model = NULL
            """,
                (provider,),
            )
            self._changed(provider)
        return self.get_public()

    def record_check(self, provider, ok, message, expected=None):
        self._provider(provider)
        # Pass only fixed application messages here, never raw upstream text.
        with self._transaction():
            private = self.get_private(provider)
            if expected is not None and private != expected:
                return False
            message = str(message)
            if private["api_key"]:
                message = message.replace(private["api_key"], "[redacted]")
            message = message[:240]
            self.db.execute(
                """
                INSERT INTO agent_provider_checks (provider, fingerprint, ok, message, checked_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(provider) DO UPDATE SET fingerprint = excluded.fingerprint,
                    ok = excluded.ok, message = excluded.message, checked_at = excluded.checked_at
            """,
                (provider, self._fingerprint(private), bool(ok), message, self.now()),
            )
        return True
