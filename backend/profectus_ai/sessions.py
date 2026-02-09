from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol


@dataclass
class SessionState:
    session_id: str
    user_id: Optional[str] = None
    history: List[str] = field(default_factory=list)
    last_evidence_ids: List[str] = field(default_factory=list)
    active_tools: List[str] = field(default_factory=list)


class SessionService(Protocol):
    def get(self, session_id: str) -> Optional[SessionState]:
        ...

    def save(self, session: SessionState) -> None:
        ...


class InMemorySessionService:
    def __init__(self) -> None:
        self._sessions: Dict[str, SessionState] = {}

    def get(self, session_id: str) -> Optional[SessionState]:
        return self._sessions.get(session_id)

    def save(self, session: SessionState) -> None:
        self._sessions[session.session_id] = session
