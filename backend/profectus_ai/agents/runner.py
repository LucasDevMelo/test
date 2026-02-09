from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

from profectus_ai.agents.policy import EscalationDecision, detect_high_risk, detect_human_request
from profectus_ai.agents.verification import verify_evidence
from profectus_ai.evidence.context import assemble_evidence_blocks
from profectus_ai.evidence.fusion import (
    compute_source_weights,
    fuse_candidates,
    infer_preferred_source,
    merge_rag_candidates,
    select_top_rag_candidates,
)
from profectus_ai.models import (
    AgentResult,
    Escalation,
    EvidenceSpan,
    IndexInfoRequest,
    IndexInfoResponse,
    RagInfoRequest,
    RagInfoResponse,
    ReadSpansRequest,
    ReadSpansResponse,
    ToolCallLog,
)
from profectus_ai.query_normalization import normalize_query
from profectus_ai.sessions import InMemorySessionService, SessionService, SessionState
from profectus_ai.tools.indexinfo import indexinfo
from profectus_ai.tools.raginfo import raginfo
from profectus_ai.tools.read_spans import read_spans

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    top_k: int = 5
    max_iterations: int = 2


class AgenticRunner:
    def __init__(
        self,
        *,
        config: Optional[AgentConfig] = None,
        session_service: Optional[SessionService] = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.session_service = session_service or InMemorySessionService()

    def run(self, query: str, *, session_id: Optional[str] = None, user_id: Optional[str] = None) -> AgentResult:
        session = self._load_session(session_id=session_id, user_id=user_id)
        session.history.append(query)
        session.active_tools = ["indexinfo", "raginfo", "read_spans"]

        tool_logs: List[ToolCallLog] = []
        top_k = self.config.top_k
        evidence_spans = []
        evidence_blocks = []

        for iteration in range(self.config.max_iterations):
            normalized_query = normalize_query(query)
            index_response, index_log = self._call_indexinfo(normalized_query, top_k)
            tool_logs.append(index_log)

            candidates_by_source = {}
            for source in ("Help Center", "YouTube"):
                rag_response, rag_log = self._call_raginfo(normalized_query, top_k, sources=[source])
                rag_log.name = f"raginfo_{source.lower().replace(' ', '_')}"
                tool_logs.append(rag_log)
                candidates_by_source[source] = rag_response.candidates

            preferred_source = infer_preferred_source(query)
            source_weights = compute_source_weights(
                candidates_by_source,
                preferred_source=preferred_source,
            )
            logger.info(
                "raginfo counts by source: %s (preferred=%s)",
                {k: len(v) for k, v in candidates_by_source.items()},
                preferred_source,
            )
            logger.info("raginfo source weights: %s", source_weights)
            merged_rag = merge_rag_candidates(candidates_by_source, source_weights)
            tool_logs.append(
                ToolCallLog(
                    name="raginfo_merge",
                    request={"preferred_source": preferred_source},
                    response={
                        "source_weights": source_weights,
                        "counts": {k: len(v) for k, v in candidates_by_source.items()},
                    },
                    error=None,
                    duration_ms=None,
                )
            )

            fused = fuse_candidates(index_response.items, merged_rag, top_k=top_k)
            top_rag = select_top_rag_candidates(merged_rag, top_k=top_k)

            spans_response, spans_log = self._call_read_spans([c.chunk_id for c in top_rag])
            tool_logs.append(spans_log)

            evidence_spans = spans_response.spans
            evidence_blocks = assemble_evidence_blocks(evidence_spans)
            session.last_evidence_ids = [span.span_id for span in evidence_spans]

            if evidence_spans:
                break
            top_k = min(top_k * 2, 20)

        decision = self._escalation_decision(query, evidence_spans)
        escalation = None
        if decision.should_escalate:
            escalation = Escalation(
                reasons=decision.reasons,
                detail=decision.detail,
                evidence_ids=[span.span_id for span in evidence_spans],
                evidence_count=len(evidence_spans),
            )

        status = "escalated" if escalation else "ok"
        result = AgentResult(
            status=status,
            session_id=session.session_id,
            evidence=evidence_spans,
            evidence_blocks=evidence_blocks,
            tool_logs=tool_logs,
            escalation=escalation,
        )
        self.session_service.save(session)
        return result

    def _load_session(self, *, session_id: Optional[str], user_id: Optional[str]) -> SessionState:
        if session_id:
            existing = self.session_service.get(session_id)
            if existing:
                return existing
        new_id = session_id or f"session_{int(time.time() * 1000)}"
        return SessionState(session_id=new_id, user_id=user_id)

    def _escalation_decision(self, query: str, spans: List[EvidenceSpan]) -> EscalationDecision:
        reasons: List[str] = []
        detail = None

        high_risk = detect_high_risk(query)
        if high_risk:
            reasons.append("high_risk")
            detail = high_risk

        if detect_human_request(query):
            reasons.append("user_request")

        verification = verify_evidence(spans)
        if not verification.ok:
            reasons.append(verification.reason or "verification_failed")

        return EscalationDecision(should_escalate=bool(reasons), reasons=reasons, detail=detail)

    def _call_indexinfo(self, query: str, top_k: int):
        start = time.perf_counter()
        request = IndexInfoRequest(query=query, limit=top_k)
        error = None
        response = {}
        result = IndexInfoResponse(total=0, offset=0, limit=top_k, items=[])
        try:
            result = indexinfo(request)
            response = result.model_dump()
        except Exception as exc:
            error = str(exc)
            response = {"error": error}
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
        log = ToolCallLog(
            name="indexinfo",
            request=request.model_dump(),
            response=response,
            error=error,
            duration_ms=duration_ms,
        )
        return result, log

    def _call_raginfo(self, query: str, top_k: int, *, sources: Optional[List[str]] = None):
        start = time.perf_counter()
        request = RagInfoRequest(query=query, top_k=top_k, sources=sources)
        error = None
        response = {}
        result = RagInfoResponse(candidates=[])
        try:
            result = raginfo(request)
            response = result.model_dump()
        except Exception as exc:
            error = str(exc)
            response = {"error": error}
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
        log = ToolCallLog(
            name="raginfo",
            request=request.model_dump(),
            response=response,
            error=error,
            duration_ms=duration_ms,
        )
        return result, log

    def _call_read_spans(self, chunk_ids: List[str]):
        start = time.perf_counter()
        request = ReadSpansRequest(chunk_ids=chunk_ids)
        error = None
        response = {}
        result = ReadSpansResponse(spans=[])
        try:
            result = read_spans(request)
            response = result.model_dump()
        except Exception as exc:
            error = str(exc)
            response = {"error": error}
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
        log = ToolCallLog(
            name="read_spans",
            request=request.model_dump(),
            response=response,
            error=error,
            duration_ms=duration_ms,
        )
        return result, log
