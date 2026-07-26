import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import streamlit as st


def _owner_hash(workspace_id: str, pin: str) -> str:
    raw = f"{workspace_id.strip().lower()}::{pin.strip()}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class SQLiteStorage:
    """Free fallback. Survives browser refreshes, but not guaranteed across redeploys."""

    def __init__(self, path: str = "/tmp/mathbridge_sessions.db"):
        self.path = path
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.path)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    owner_hash TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    title TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def save(self, workspace_id: str, pin: str, mode: str, title: str, payload: dict, session_id: str | None = None) -> str:
        session_id = session_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        owner = _owner_hash(workspace_id, pin)
        with self._connect() as conn:
            existing = conn.execute("SELECT created_at FROM sessions WHERE id=? AND owner_hash=?", (session_id, owner)).fetchone()
            created = existing[0] if existing else now
            conn.execute(
                """INSERT OR REPLACE INTO sessions
                (id, owner_hash, mode, title, payload, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (session_id, owner, mode, title, json.dumps(payload, ensure_ascii=False), created, now),
            )
        return session_id

    def list(self, workspace_id: str, pin: str, mode: str | None = None) -> list[dict]:
        owner = _owner_hash(workspace_id, pin)
        query = "SELECT id, mode, title, created_at, updated_at FROM sessions WHERE owner_hash=?"
        params: list[Any] = [owner]
        if mode:
            query += " AND mode=?"
            params.append(mode)
        query += " ORDER BY updated_at DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(zip(["id", "mode", "title", "created_at", "updated_at"], row)) for row in rows]

    def load(self, workspace_id: str, pin: str, session_id: str) -> dict | None:
        owner = _owner_hash(workspace_id, pin)
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM sessions WHERE id=? AND owner_hash=?", (session_id, owner)).fetchone()
        return json.loads(row[0]) if row else None

    def delete(self, workspace_id: str, pin: str, session_id: str) -> None:
        owner = _owner_hash(workspace_id, pin)
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE id=? AND owner_hash=?", (session_id, owner))


class SupabaseStorage:
    """Durable free-tier storage when Supabase secrets are configured."""

    def __init__(self, url: str, service_key: str):
        self.base = f"{url.rstrip('/')}/rest/v1/mathbridge_sessions"
        self.headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Content-Type": "application/json",
        }

    def save(self, workspace_id: str, pin: str, mode: str, title: str, payload: dict, session_id: str | None = None) -> str:
        session_id = session_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        body = {
            "id": session_id,
            "owner_hash": _owner_hash(workspace_id, pin),
            "mode": mode,
            "title": title,
            "payload": payload,
            "updated_at": now,
        }
        headers = {**self.headers, "Prefer": "resolution=merge-duplicates,return=representation"}
        response = requests.post(self.base, headers=headers, json=body, timeout=30)
        response.raise_for_status()
        return session_id

    def list(self, workspace_id: str, pin: str, mode: str | None = None) -> list[dict]:
        params = {
            "owner_hash": f"eq.{_owner_hash(workspace_id, pin)}",
            "select": "id,mode,title,created_at,updated_at",
            "order": "updated_at.desc",
        }
        if mode:
            params["mode"] = f"eq.{mode}"
        response = requests.get(self.base, headers=self.headers, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def load(self, workspace_id: str, pin: str, session_id: str) -> dict | None:
        params = {
            "id": f"eq.{session_id}",
            "owner_hash": f"eq.{_owner_hash(workspace_id, pin)}",
            "select": "payload",
            "limit": "1",
        }
        response = requests.get(self.base, headers=self.headers, params=params, timeout=30)
        response.raise_for_status()
        rows = response.json()
        return rows[0]["payload"] if rows else None

    def delete(self, workspace_id: str, pin: str, session_id: str) -> None:
        params = {"id": f"eq.{session_id}", "owner_hash": f"eq.{_owner_hash(workspace_id, pin)}"}
        response = requests.delete(self.base, headers=self.headers, params=params, timeout=30)
        response.raise_for_status()


@st.cache_resource
def get_storage():
    url = st.secrets.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if url and key:
        return SupabaseStorage(url, key)
    return SQLiteStorage()


def workspace_ready() -> bool:
    return bool(st.session_state.get("workspace_connected") and st.session_state.get("workspace_id") and st.session_state.get("workspace_pin"))
