"""Where the alert state lives between runs.

GitHub Actions runners are thrown away after every run, so a local file
(.alert_state.json) is forgotten and the same alerts get sent again. The state
therefore lives in a small Postgres table (e.g. the free Neon database you
already use through DATABASE_URL). A local JSON file remains as a fallback for
running on your own machine.

    store = get_store()          # Postgres if DATABASE_URL is set, else file
    state = store.load()         # {} when nothing stored yet
    ...                          # change `state` (plain JSON-compatible dict)
    store.save(state)

Postgres details
  * one row: table `alert_state` (id, state JSONB, updated_at), created on demand
  * save() re-reads the row under a row lock and applies ONLY the keys this
    run changed (per symbol, and per symbol inside "_levels"), so two runs
    that overlap (cron + push, manual dispatch) don't wipe each other's
    entries -- unlike a blind "write the whole file back".
  * the connection string is never logged.

State layout (unchanged): {"SPY": "<bar>|<Up/Down>", "_levels": {"SPY": {...}}}
"""
from __future__ import annotations

import copy
import json
import logging
import os
import re
import tempfile

import config

log = logging.getLogger(__name__)
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# --------------------------------------------------------------------------
# Pure merge helper (unit-testable, no database)
# --------------------------------------------------------------------------
def merge_changes(base: dict, new: dict, current: dict) -> dict:
    """Apply what changed between `base` (state as loaded) and `new` (state at
    the end of this run) onto `current` (what the store holds right now).
    Top-level keys are compared per key; "_levels" is compared per symbol.
    Keys this run did not touch are left exactly as `current` has them."""
    merged = copy.deepcopy(current)
    for key, value in new.items():
        if key == "_levels":
            base_lv = base.get("_levels", {})
            for sym, sym_state in value.items():
                if sym_state != base_lv.get(sym):
                    merged.setdefault("_levels", {})[sym] = copy.deepcopy(sym_state)
        elif value != base.get(key):
            merged[key] = copy.deepcopy(value)
    return merged


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------
class FileStore:
    """Local JSON file. Fine on your own machine; NOT persistent on Actions."""

    name = "file"

    def __init__(self, path: str | None = None):
        self.path = path or config.STATE_FILE

    def load(self) -> dict:
        try:
            with open(self.path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save(self, state: dict) -> None:
        folder = os.path.dirname(os.path.abspath(self.path))
        fd, tmp = tempfile.mkstemp(dir=folder, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(state, f)
            os.replace(tmp, self.path)               # atomic: no half-written file
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise


class PostgresStore:
    """JSON state in one Postgres row (Neon, Supabase, RDS, local ...)."""

    name = "postgres"

    def __init__(self, url: str, key: str | None = None, table: str | None = None,
                 connect_timeout: int = 10):
        self.url = url
        self.key = key or config.STATE_KEY
        self.table = table or config.STATE_TABLE
        if not _IDENT.match(self.table):
            raise ValueError(f"invalid table name: {self.table!r}")
        self.connect_timeout = connect_timeout
        self._base: dict = {}                         # state as loaded; used to find our changes

    def _connect(self):
        import psycopg2                                # lazy: only needed for this backend
        return psycopg2.connect(self.url, connect_timeout=self.connect_timeout)

    def _ensure_table(self, cur) -> None:
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS {self.table} ("
            "id TEXT PRIMARY KEY, "
            "state JSONB NOT NULL, "
            "updated_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )

    def load(self) -> dict:
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:           # `with conn` = one transaction
                self._ensure_table(cur)
                cur.execute(f"SELECT state FROM {self.table} WHERE id = %s", (self.key,))
                row = cur.fetchone()
        finally:
            conn.close()
        state = row[0] if row else {}
        self._base = copy.deepcopy(state)
        return state

    def save(self, state: dict) -> None:
        from psycopg2.extras import Json
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                self._ensure_table(cur)
                cur.execute(
                    f"SELECT state FROM {self.table} WHERE id = %s FOR UPDATE", (self.key,)
                )
                row = cur.fetchone()
                merged = merge_changes(self._base, state, row[0] if row else {})
                cur.execute(
                    f"INSERT INTO {self.table} (id, state, updated_at) VALUES (%s, %s, now()) "
                    "ON CONFLICT (id) DO UPDATE SET state = EXCLUDED.state, updated_at = now()",
                    (self.key, Json(merged)),
                )
        finally:
            conn.close()
        self._base = copy.deepcopy(merged)


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------
def get_store(backend: str | None = None):
    """config.STATE_BACKEND: "auto" (Postgres if DATABASE_URL is set, else
    file), "postgres" (requires DATABASE_URL) or "file"."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    backend = (backend or config.STATE_BACKEND).lower()
    url = os.getenv("DATABASE_URL")

    if backend == "postgres" or (backend == "auto" and url):
        if not url:
            raise RuntimeError("STATE_BACKEND is 'postgres' but DATABASE_URL is not set.")
        return PostgresStore(url)

    if backend not in ("auto", "file"):
        raise ValueError(f"unknown STATE_BACKEND: {backend!r}")
    if os.getenv("GITHUB_ACTIONS") == "true":
        log.warning("Using the local state FILE on GitHub Actions: it is discarded after every "
                    "run, so alerts can repeat. Set DATABASE_URL (Postgres) to persist state.")
    return FileStore()
