from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict


SourceType = Literal["Help Center", "YouTube"]


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: SourceType
    url: str
    anchor: Optional[str] = None
    timestamp_start: Optional[float] = None
    timestamp_end: Optional[float] = None
    reference: Optional[str] = None


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    source: SourceType
    title: str
    doc_url: str
    summary: str
    hierarchy: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    updated_at: Optional[str] = None
    source_id: Optional[str] = None
    provenance: Provenance


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    doc_id: str
    source: SourceType
    chunk_index: int
    text: str
    start: Optional[int] = None
    end: Optional[int] = None
    time_start: Optional[float] = None
    time_end: Optional[float] = None
    provenance: Provenance


class IndexEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    source: SourceType
    title: str
    summary: str
    hierarchy: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    updated_at: Optional[str] = None
    source_id: Optional[str] = None
    doc_url: str
    provenance: Provenance


class EvidenceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    span_id: str
    doc_id: str
    chunk_id: str
    text: str
    start: Optional[int] = None
    end: Optional[int] = None
    time_start: Optional[float] = None
    time_end: Optional[float] = None
    provenance: Provenance


class IndexOverviewMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    build_timestamp: str
    source_counts: Dict[str, int]
    checksum: str
    total_docs: int


class IndexInfoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: Optional[str] = None
    sources: Optional[List[SourceType]] = None
    tags: Optional[List[str]] = None
    hierarchy_prefix: Optional[List[str]] = None
    updated_after: Optional[str] = None
    detail: Literal["full", "summary", "title"] = "summary"
    summary_max_chars: int = 280
    limit: int = 10
    offset: int = 0


class IndexInfoResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: int
    offset: int
    limit: int
    items: List[IndexEntry]


class RagInfoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    top_k: int = 8
    sources: Optional[List[SourceType]] = None
    tags: Optional[List[str]] = None
    hierarchy_prefix: Optional[List[str]] = None


class RagCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    doc_id: str
    snippet: str
    rank: int
    provenance: Provenance
    score: Optional[float] = None
    tags: List[str] = Field(default_factory=list)
    hierarchy: List[str] = Field(default_factory=list)


class RagInfoResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: List[RagCandidate]


class ReadSpansRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_ids: List[str]
    start: Optional[int] = None
    end: Optional[int] = None
    time_start: Optional[float] = None
    time_end: Optional[float] = None


class ReadSpansResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spans: List[EvidenceSpan]


class ToolCallLog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    request: Dict[str, object]
    response: Dict[str, object]
    error: Optional[str] = None
    duration_ms: Optional[float] = None


class Escalation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasons: List[str]
    detail: Optional[str] = None
    evidence_ids: List[str] = Field(default_factory=list)
    evidence_count: int = 0


class AgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    session_id: Optional[str] = None
    evidence: List[EvidenceSpan]
    evidence_blocks: List[Dict[str, object]]
    tool_logs: List[ToolCallLog]
    escalation: Optional[Escalation] = None
