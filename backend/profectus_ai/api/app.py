from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dotenv import load_dotenv

from profectus_ai.adk_runner import run_query
from profectus_ai.adk.classifier import classify_query
from profectus_ai.services.escalation_store import get_escalation_store
from profectus_ai.services.logging import log_event
from profectus_ai.services.session_store import get_session_store


REPO_ROOT = Path(__file__).resolve().parents[3]
for env_path in (REPO_ROOT / ".env", REPO_ROOT / "backend" / ".env"):
    if env_path.exists():
        load_dotenv(env_path, override=False)
        break
WEB_DIR = REPO_ROOT / "frontend"

SESSION_HISTORY_LIMIT = int(os.environ.get("PROFECTUS_SESSION_HISTORY_LIMIT", "12"))
HISTORY_MAX_CHARS = int(os.environ.get("PROFECTUS_HISTORY_MAX_CHARS", "4000"))

_LOG_LEVEL = os.environ.get("PROFECTUS_LOG_LEVEL", "INFO").upper()
root_logger = logging.getLogger()
root_logger.setLevel(_LOG_LEVEL)
if not root_logger.handlers:
    logging.basicConfig(
        level=_LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
logger = logging.getLogger(__name__)


def _describe_path(path: Path) -> str:
    try:
        if path.exists():
            size = path.stat().st_size
            return f"found ({size} bytes)"
    except Exception as exc:
        return f"error ({exc})"
    return "missing"


def _resolve_embedding_model_path(model_name: str) -> Optional[str]:
    env_path = os.environ.get("PROFECTUS_EMBEDDING_MODEL_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return str(path)
        return None

    cache_root = Path.home() / ".cache" / "torch" / "sentence_transformers"
    normalized = model_name.replace("/", "_")
    candidates = [
        cache_root / model_name,
        cache_root / normalized,
        cache_root / f"sentence-transformers_{normalized}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    model_variants = [model_name]
    if "/" not in model_name:
        model_variants.append(f"sentence-transformers/{model_name}")

    for variant in model_variants:
        hub_root = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / f"models--{variant.replace('/', '--')}"
            / "snapshots"
        )
        if hub_root.exists():
            snapshots = sorted([p for p in hub_root.iterdir() if p.is_dir()])
            if snapshots:
                return str(snapshots[0])

    return None


class TimingTracker:
    """Tracks timing breakdown for a request."""
    def __init__(self) -> None:
        self.start_time = time.time()
        self.steps: Dict[str, float] = {}
        self._markers: Dict[str, float] = {}

    def mark(self, name: str) -> None:
        """Record a timestamp marker."""
        self._markers[name] = time.time()

    def elapsed(self, name: str | None = None) -> float:
        """Get elapsed time since start or since a marker."""
        current = time.time()
        if name:
            marker = self._markers.get(name)
            if marker:
                return current - marker
        return current - self.start_time

    def record_step(self, name: str, start_marker: str) -> None:
        """Record a step duration from a marker."""
        if start_marker in self._markers:
            self.steps[name] = self.elapsed(start_marker)

    def summary(self) -> Dict[str, Any]:
        """Get timing summary."""
        total = self.elapsed()
        return {
            "total_ms": round(total * 1000),
            "steps": {k: round(v * 1000) for k, v in self.steps.items()},
            "breakdown": self._format_breakdown(total),
        }

    def _format_breakdown(self, total: float) -> str:
        """Format timing breakdown as string."""
        if not self.steps:
            return f"Total: {round(total*1000)}ms"
        parts = []
        for name, duration in sorted(self.steps.items(), key=lambda x: -x[1]):
            # Simplify names: tool_adk_raginfo_dual -> rag, etc.
            short_name = name.replace("tool_adk_", "").replace("tool_", "").replace("_total", "")
            parts.append(f"{short_name}={round(duration*1000)}ms")
        return " | ".join(parts)

    def log(self, request_id: str) -> None:
        """Log timing summary to console."""
        total = self.elapsed()
        total_ms = round(total * 1000)
        print(f"[TIMING] {request_id[:8]} | {total_ms}ms total")
        if self.steps:
            for name, duration in sorted(self.steps.items(), key=lambda x: -x[1]):
                ms = round(duration * 1000)
                print(f"  - {name}: {ms}ms")

app = FastAPI(title="Profectus AI Support Agent", version="0.1.0")
session_store = get_session_store()
escalation_store = get_escalation_store()


@app.on_event("startup")
async def log_rag_assets() -> None:
    from profectus_ai.config import (
        DEFAULT_EMBEDDING_MODEL,
        INDEX_OVERVIEW_PATH,
        INDEX_OVERVIEW_META_PATH,
        LOCAL_FINAL_DATA_JSON,
        LOCAL_FINAL_FAISS_INDEX,
        LOCAL_FINAL_FAISS_METADATA,
    )

    logger.info(
        "[startup] RAG assets: faiss_index=%s, faiss_metadata=%s, corpus=%s, index_overview=%s, index_meta=%s",
        _describe_path(LOCAL_FINAL_FAISS_INDEX),
        _describe_path(LOCAL_FINAL_FAISS_METADATA),
        _describe_path(LOCAL_FINAL_DATA_JSON),
        _describe_path(INDEX_OVERVIEW_PATH),
        _describe_path(INDEX_OVERVIEW_META_PATH),
    )

    resolved_model = _resolve_embedding_model_path(DEFAULT_EMBEDDING_MODEL)
    if resolved_model:
        logger.info("[startup] Embedding model resolved: %s", resolved_model)
    else:
        logger.warning(
            "[startup] Embedding model not found in cache. "
            "Set PROFECTUS_EMBEDDING_MODEL_PATH or preload the model in the image."
        )


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    debug: Optional[bool] = False
    fast_mode: Optional[bool] = False


def _build_prompt(history: List[Dict[str, Any]], message: str) -> str:
    if not history:
        return message
    lines: List[str] = []
    total = 0
    for entry in history:
        role = entry.get("role", "user")
        text = entry.get("text", "")
        if not text:
            continue
        line = f"{role}: {text}"
        total += len(line)
        if total > HISTORY_MAX_CHARS:
            break
        lines.append(line)
    lines.append(f"user: {message}")
    return "Conversation so far:\n" + "\n".join(lines)


def _parse_escalation_reason(verdict: Optional[str]) -> str:
    if not verdict:
        return ""
    text = verdict.strip()
    if text.upper().startswith("ESCALATE"):
        parts = text.split(":", 1)
        return parts[1].strip() if len(parts) > 1 else ""
    return ""




def _trace_from_events(path: Path) -> Dict[str, Any]:
    steps: List[str] = []
    if not path.exists():
        return {"steps": steps}

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        for part in record.get("parts") or []:
            if part.get("type") == "function_call":
                name = part.get("name")
                if name == "adk_indexinfo":
                    steps.append("indexinfo: keyword search")
                elif name in {"adk_raginfo", "adk_raginfo_dual"}:
                    steps.append("raginfo: semantic search")
                elif name == "adk_read_spans":
                    args = part.get("args") or {}
                    count = len(args.get("chunk_ids") or [])
                    steps.append(f"read_spans: {count} span(s)")
                elif name == "adk_open_source":
                    steps.append("open_source: full document")
            if part.get("type") == "function_response":
                name = part.get("name")
                response = part.get("response") or {}
                if name == "adk_indexinfo" and isinstance(response.get("items"), list):
                    steps.append(f"indexinfo: {len(response['items'])} result(s)")
                if name in {"adk_raginfo", "adk_raginfo_dual"}:
                    by_source = response.get("by_source")
                    if isinstance(by_source, dict):
                        counts = {
                            source: len(items)
                            for source, items in by_source.items()
                            if isinstance(items, list)
                        }
                        if counts:
                            formatted = ", ".join(f"{k}={v}" for k, v in counts.items())
                            steps.append(f"raginfo: {formatted}")
                if name == "adk_read_spans" and isinstance(response.get("spans"), list):
                    steps.append(f"read_spans: {len(response['spans'])} retrieved")
    return {"steps": steps}


async def _handle_query(
    *,
    message: str,
    session_id: Optional[str],
    user_id: Optional[str],
    debug: bool,
    fast_mode: bool = False,
    progress_callback: Optional[Callable] = None,
    stream_callback: Optional[Callable] = None,
    log_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    classification: Optional[Dict[str, str]] = None
    try:
        query_type, reason = classify_query(message)
        classification = {"type": query_type.value, "reason": reason}
    except Exception:
        classification = None
    timing = TimingTracker()
    request_id = str(uuid.uuid4())
    if progress_callback:
        await progress_callback({"type": "progress", "stage": "start", "message": "Starting request..."})
    if not session_id:
        session_id = await session_store.create_session(user_id=user_id)

    timing.mark("session_ready")
    history = await session_store.list_messages(session_id, SESSION_HISTORY_LIMIT)
    prompt = _build_prompt(history, message)
    timing.record_step("db_fetch", "session_ready")

    trace: Optional[Dict[str, Any]] = None
    events_path: Optional[Path] = None
    if debug:
        events_dir = REPO_ROOT / "backend" / "data" / "api_events"
        events_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        events_path = events_dir / f"api_{session_id}_{ts}.jsonl"

    timing.mark("rag_start")
    if progress_callback:
        await progress_callback({"type": "progress", "stage": "retrieve", "message": "Preparing retrieval..."})
    rag_mode = "fast" if fast_mode else None
    try:
        answer, verdict, confidence = await run_query(
            prompt,
            print_events=False,
            events_path=events_path,
            skip_trivial=False,
            rag_mode=rag_mode,
            progress_callback=progress_callback,
            stream_callback=stream_callback,
            timing_tracker=timing,
        )
    except Exception as exc:
        logger.exception("run_query failed for session %s", session_id)
        await _emit_log(
            log_callback,
            level="error",
            message=f"run_query failed: {type(exc).__name__}: {exc}",
        )
        raise
    timing.record_step("rag_total", "rag_start")

    if debug and events_path:
        trace = _trace_from_events(events_path)

    # Always capture timing (not just in debug mode)
    timing_summary = timing.summary()

    await session_store.append_message(session_id, "user", message)
    await session_store.append_message(session_id, "assistant", answer)

    escalation_id = None
    if verdict and verdict.upper().startswith("ESCALATE"):
        reason = _parse_escalation_reason(verdict)
        escalation_id = await escalation_store.create_escalation(
            {
                "session_id": session_id,
                "user_id": user_id,
                "query": message,
                "answer": answer,
                "reason": reason,
            }
        )

    timing.log(request_id)
    log_event(
        "chat_response",
        request_id=request_id,
        session_id=session_id,
        user_id=user_id,
        verdict=verdict,
        confidence=confidence,
        escalated=bool(escalation_id),
        timing_ms=round(timing.elapsed() * 1000),
    )

    return {
        "request_id": request_id,
        "session_id": session_id,
        "answer": answer,
        "verdict": verdict,
        "confidence": confidence,
        "escalation_id": escalation_id,
        "trace": trace,
        "timing": timing_summary,
        "classification": classification,
    }


async def _emit_log(
    log_callback: Optional[Callable],
    *,
    level: str,
    message: str,
) -> None:
    if not log_callback or not message:
        return
    try:
        await log_callback(message, level=level)
    except Exception:
        return


def _strip_status_prefix(message: str) -> str:
    if not message:
        return ""
    if message.startswith("["):
        closing = message.find("]")
        if closing != -1:
            return message[closing + 1 :].strip()
    return message.strip()


def _parse_first_int(message: str) -> Optional[int]:
    for token in message.split():
        if token.isdigit():
            return int(token)
        digits = "".join(ch for ch in token if ch.isdigit())
        if digits:
            return int(digits)
    return None


def _parse_source_counts(message: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not message:
        return counts
    import re

    for match in re.finditer(r"([A-Za-z][A-Za-z ]+):\s*(\d+)", message):
        label = match.group(1).strip()
        count = int(match.group(2))
        counts[label] = count
    return counts


def _humanize_progress(stage: str, message: str, *, fast_mode: bool = False) -> Optional[str]:
    """Map internal progress stages to user-friendly text."""
    stage = (stage or "").lower()
    cleaned = _strip_status_prefix(message)
    mapping = {
        "start": "Getting things ready...",
        "retrieve": "Looking for relevant info...",
        "routing": "Understanding your question...",
        "keyword_search": "Checking our guides...",
        "semantic_search": "Searching our help center...",
        "opening": "Opening a relevant guide...",
        "reading": "Reviewing the details...",
        "drafting": "Writing your answer...",
    }

    if fast_mode and stage in {"semantic_search", "found"}:
        return None

    if stage == "tool_result":
        lowered = cleaned.lower()
        if "index result" in lowered:
            count = _parse_first_int(cleaned)
            if count is not None:
                return f"Found {count} index match(es)."
            return "Found index matches."
        if "semantic candidate" in lowered:
            counts = _parse_source_counts(cleaned)
            if counts:
                parts = []
                if "Help Center" in counts:
                    parts.append(f"{counts['Help Center']} docs")
                if "YouTube" in counts:
                    parts.append(f"{counts['YouTube']} videos")
                if parts:
                    return "Found " + ", ".join(parts) + "."
            count = _parse_first_int(cleaned)
            if count is not None:
                return f"Found {count} relevant result(s)."
        return None

    if stage == "found" and cleaned:
        counts = _parse_source_counts(cleaned)
        if counts:
            parts = []
            if "Help Center" in counts:
                parts.append(f"{counts['Help Center']} docs")
            if "YouTube" in counts:
                parts.append(f"{counts['YouTube']} videos")
            if parts:
                return "Found " + ", ".join(parts) + "."
        count = _parse_first_int(cleaned)
        if count is not None:
            return f"Found {count} relevant result(s)."
        return None

    if stage == "retrieved" and cleaned:
        count = _parse_first_int(cleaned)
        if count is not None:
            return f"Gathered {count} key passages."
        return "Gathering key passages..."

    if stage in mapping:
        return mapping[stage]
    if cleaned and len(cleaned) <= 80:
        return cleaned
    return "Working on your request..."


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/index-stats")
def index_stats() -> Dict[str, Any]:
    """Return statistics about the loaded index."""
    import json
    from profectus_ai.config import LOCAL_FINAL_DATA_JSON, LOCAL_FINAL_FAISS_METADATA

    stats = {
        "total_entries": 0,
        "sources": {},
        "last_updated": None,
    }

    # Load corpus for detailed stats
    try:
        with open(LOCAL_FINAL_DATA_JSON, "r", encoding="utf-8") as f:
            corpus = json.load(f)
            stats["total_entries"] = len(corpus)

            # Count by URL pattern to determine sources
            url_patterns = {
                "docs": 0,
                "blog": 0,
                "legal": 0,
                "youtube": 0,
                "other": 0,
            }

            for item in corpus:
                url = item.get("url", "").lower()
                source = item.get("source", "")

                if source == "YouTube":
                    url_patterns["youtube"] += 1
                elif "docs.profectus.ai" in url:
                    url_patterns["docs"] += 1
                elif "/post/" in url:
                    url_patterns["blog"] += 1
                elif any(x in url for x in ["disclaimer", "privacy", "terms"]):
                    url_patterns["legal"] += 1
                else:
                    url_patterns["other"] += 1

            # Only include non-zero sources
            stats["sources"] = {
                name: count for name, count in url_patterns.items() if count > 0
            }
    except FileNotFoundError:
        pass

    # Get file modification time
    try:
        stats["last_updated"] = LOCAL_FINAL_DATA_JSON.stat().st_mtime
    except FileNotFoundError:
        pass

    return stats


@app.post("/chat")
async def chat(request: ChatRequest) -> JSONResponse:
    payload = await _handle_query(
        message=request.message,
        session_id=request.session_id,
        user_id=request.user_id,
        debug=bool(request.debug),
        fast_mode=bool(request.fast_mode),
    )
    return JSONResponse(payload)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            text = await websocket.receive_text()
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "invalid_json"})
                continue
            message = str(data.get("message", "")).strip()
            if not message:
                await websocket.send_json({"error": "message_required"})
                continue

            # Progress callback for streaming updates
            async def send_progress(progress: Dict[str, Any]) -> None:
                stage = progress.get("stage", "")
                message = progress.get("message", "")
                if debug_flag:
                    payload = {"type": "progress", "stage": stage, "message": message}
                else:
                    friendly = _humanize_progress(stage, message, fast_mode=fast_mode_flag)
                    if not friendly:
                        return
                    payload = {"type": "progress", "stage": stage, "message": friendly}
                await websocket.send_json(payload)

            async def send_stream(text: str) -> None:
                await websocket.send_json({"type": "stream", "text": text})

            async def send_log(message: str, *, level: str = "info") -> None:
                normalized = (level or "info").lower()
                if normalized not in {"error", "warn", "warning"} and not debug_flag:
                    return
                payload = {"type": "log", "level": normalized, "message": message}
                await websocket.send_json(payload)

            debug_flag = bool(data.get("debug"))
            fast_mode_flag = bool(data.get("fast_mode"))
            try:
                response = await _handle_query(
                    message=message,
                    session_id=data.get("session_id"),
                    user_id=data.get("user_id"),
                    debug=debug_flag,
                    fast_mode=fast_mode_flag,
                    progress_callback=send_progress,
                    stream_callback=send_stream,
                    log_callback=send_log,
                )
                await websocket.send_json({"type": "response", **response})
            except Exception as exc:
                logger.exception("WebSocket request failed")
                await send_log(
                    f"Request failed: {type(exc).__name__}: {exc}",
                    level="error",
                )
                payload = {"type": "response", "error": "internal_error"}
                if debug_flag:
                    payload["error"] = str(exc)
                await websocket.send_json(payload)
    except WebSocketDisconnect:
        return


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    index_path = WEB_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse({"status": "ok", "message": "UI not configured."})
