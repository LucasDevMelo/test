from __future__ import annotations

import os
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from google.cloud import firestore  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    firestore = None


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionStore(ABC):
    @abstractmethod
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def create_session(self, user_id: Optional[str] = None) -> str:
        raise NotImplementedError

    @abstractmethod
    async def append_message(
        self, session_id: str, role: str, text: str
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def list_messages(self, session_id: str, limit: int) -> List[Dict[str, Any]]:
        raise NotImplementedError


class MemorySessionStore(SessionStore):
    def __init__(self, *, history_limit: int = 12) -> None:
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._history_limit = history_limit

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._sessions.get(session_id)

    async def create_session(self, user_id: Optional[str] = None) -> str:
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": _utcnow(),
            "updated_at": _utcnow(),
            "messages": [],
        }
        return session_id

    async def append_message(self, session_id: str, role: str, text: str) -> None:
        session = self._sessions.get(session_id)
        if not session:
            # Auto-create session on first message (handles client-generated IDs)
            session = {
                "session_id": session_id,
                "user_id": None,
                "created_at": _utcnow(),
                "updated_at": _utcnow(),
                "messages": [],
            }
            self._sessions[session_id] = session
        messages = session.get("messages", [])
        messages.append({"role": role, "text": text, "ts": _utcnow()})
        if len(messages) > self._history_limit:
            messages = messages[-self._history_limit :]
        session["messages"] = messages
        session["updated_at"] = _utcnow()
        session["last_message"] = text

    async def list_messages(self, session_id: str, limit: int) -> List[Dict[str, Any]]:
        session = self._sessions.get(session_id)
        if not session:
            return []
        messages = session.get("messages", [])
        return messages[-limit:]


class FirestoreSessionStore(SessionStore):
    def __init__(
        self,
        *,
        project_id: Optional[str] = None,
        collection: str = "sessions",
        history_limit: int = 12,
    ) -> None:
        if firestore is None:
            raise RuntimeError("google-cloud-firestore not installed.")
        self._client = firestore.AsyncClient(project=project_id)
        self._collection = collection
        self._history_limit = history_limit

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        doc = await self._client.collection(self._collection).document(session_id).get()
        if not doc.exists:
            return None
        payload = doc.to_dict() or {}
        payload["session_id"] = session_id
        return payload

    async def create_session(self, user_id: Optional[str] = None) -> str:
        session_id = str(uuid.uuid4())
        payload = {
            "user_id": user_id,
            "created_at": _utcnow(),
            "updated_at": _utcnow(),
            "messages": [],
        }
        await self._client.collection(self._collection).document(session_id).set(payload)
        return session_id

    async def append_message(self, session_id: str, role: str, text: str) -> None:
        doc_ref = self._client.collection(self._collection).document(session_id)
        doc = await doc_ref.get()
        if not doc.exists:
            # Auto-create session on first message (handles client-generated IDs)
            payload = {
                "user_id": None,
                "created_at": _utcnow(),
                "updated_at": _utcnow(),
                "messages": [],
            }
        else:
            payload = doc.to_dict() or {}
        messages = payload.get("messages", [])
        messages.append({"role": role, "text": text, "ts": _utcnow()})
        if len(messages) > self._history_limit:
            messages = messages[-self._history_limit :]
        payload["messages"] = messages
        payload["updated_at"] = _utcnow()
        payload["last_message"] = text
        await doc_ref.set(payload)

    async def list_messages(self, session_id: str, limit: int) -> List[Dict[str, Any]]:
        doc = await self._client.collection(self._collection).document(session_id).get()
        if not doc.exists:
            return []
        payload = doc.to_dict() or {}
        messages = payload.get("messages", [])
        return messages[-limit:]


_SESSION_STORE: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    global _SESSION_STORE
    if _SESSION_STORE is not None:
        return _SESSION_STORE

    history_limit = int(os.environ.get("PROFECTUS_SESSION_HISTORY_LIMIT", "12"))
    store_kind = os.environ.get("PROFECTUS_SESSION_STORE", "").lower().strip()
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")

    if store_kind == "memory":
        _SESSION_STORE = MemorySessionStore(history_limit=history_limit)
        return _SESSION_STORE

    if store_kind == "firestore" or project_id or os.environ.get("FIRESTORE_EMULATOR_HOST"):
        if firestore is None:
            _SESSION_STORE = MemorySessionStore(history_limit=history_limit)
        else:
            _SESSION_STORE = FirestoreSessionStore(
                project_id=project_id, history_limit=history_limit
            )
        return _SESSION_STORE

    _SESSION_STORE = MemorySessionStore(history_limit=history_limit)
    return _SESSION_STORE
