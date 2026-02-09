from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

try:
    from google.cloud import firestore  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    firestore = None


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class EscalationStore:
    async def create_escalation(self, payload: Dict[str, Any]) -> str:
        raise NotImplementedError


class MemoryEscalationStore(EscalationStore):
    def __init__(self) -> None:
        self._items: Dict[str, Dict[str, Any]] = {}

    async def create_escalation(self, payload: Dict[str, Any]) -> str:
        escalation_id = str(uuid.uuid4())
        stored = dict(payload)
        stored.setdefault("created_at", _utcnow())
        stored["escalation_id"] = escalation_id
        self._items[escalation_id] = stored
        return escalation_id


class FirestoreEscalationStore(EscalationStore):
    def __init__(
        self,
        *,
        project_id: Optional[str] = None,
        collection: str = "escalations",
    ) -> None:
        if firestore is None:
            raise RuntimeError("google-cloud-firestore not installed.")
        self._client = firestore.AsyncClient(project=project_id)
        self._collection = collection

    async def create_escalation(self, payload: Dict[str, Any]) -> str:
        escalation_id = str(uuid.uuid4())
        doc_ref = self._client.collection(self._collection).document(escalation_id)
        stored = dict(payload)
        stored.setdefault("created_at", _utcnow())
        stored["escalation_id"] = escalation_id
        await doc_ref.set(stored)
        return escalation_id


_ESCALATION_STORE: Optional[EscalationStore] = None


def get_escalation_store() -> EscalationStore:
    global _ESCALATION_STORE
    if _ESCALATION_STORE is not None:
        return _ESCALATION_STORE

    store_kind = os.environ.get("PROFECTUS_SESSION_STORE", "").lower().strip()
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")

    if store_kind == "memory":
        _ESCALATION_STORE = MemoryEscalationStore()
        return _ESCALATION_STORE

    if store_kind == "firestore" or project_id or os.environ.get("FIRESTORE_EMULATOR_HOST"):
        if firestore is None:
            _ESCALATION_STORE = MemoryEscalationStore()
        else:
            _ESCALATION_STORE = FirestoreEscalationStore(project_id=project_id)
        return _ESCALATION_STORE

    _ESCALATION_STORE = MemoryEscalationStore()
    return _ESCALATION_STORE
