from __future__ import annotations

import asyncio
import json
import os
import warnings
import logging
from random import choice
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from dotenv import load_dotenv

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from profectus_ai.adk.orchestrator import root_agent
from profectus_ai.adk.classifier import QueryType, classify_query
from profectus_ai.adk.tool_wrappers import rag_mode_override

from profectus_ai.agents.verification import verify_evidence
from profectus_ai.models import EvidenceSpan


def _require_api_key() -> None:
    if not os.environ.get("GOOGLE_API_KEY"):
        _load_env_file()
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is not set. Configure it to run ADK agents.")


def _load_env_file() -> None:
    candidates = [
        Path(__file__).resolve().parents[1] / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]
    for env_path in candidates:
        if env_path.exists():
            load_dotenv(env_path, override=False)
            break


def _disable_local_proxy() -> None:
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        value = os.environ.get(key)
        if value and ("127.0.0.1:9" in value or "localhost:9" in value):
            os.environ.pop(key, None)


def _suppress_adk_warnings() -> None:
    original = warnings.showwarning

    def _showwarning(message, category, filename, lineno, file=None, line=None):
        text = str(message)
        if "non-text parts in the response" in text:
            return
        return original(message, category, filename, lineno, file=file, line=line)

    warnings.showwarning = _showwarning
    warnings.filterwarnings(
        "ignore",
        message=".*non-text parts in the response.*",
    )
    logging.getLogger("google.genai").setLevel(logging.ERROR)
    logging.getLogger("google.genai.types").setLevel(logging.ERROR)
    try:
        import google.genai.types as genai_types

        if hasattr(genai_types, "_response_text_non_text_warning_logged"):
            genai_types._response_text_non_text_warning_logged = True
    except Exception:
        pass


def _trivial_response(query: str) -> str:
    normalized = query.strip().lower()
    if normalized in {"what", "what?", "wat", "wut", "huh", "huh?"}:
        return "Can you share a bit more detail about what you need help with?"
    if normalized in {"lol", "lmao", "rofl", "haha", "hahaha", "hehe"}:
        return "Thanks! What can I help you with?"
    if any(token in normalized for token in ("thank", "thanks", "thx", "ty")):
        return "You're welcome! Anything else I can help with?"
    if any(token in normalized for token in ("hi", "hello", "hey", "hiya", "howdy")):
        return "Hi! How can I help with Profectus today?"
    if any(token in normalized for token in ("bye", "goodbye", "cya", "see you")):
        return "Take care! If you need anything else, I'm here."
    return choice(
        [
            "Happy to help. What can I do for you?",
            "I'm here to help—what would you like to know?",
            "Sure. What can I help you with?",
        ]
    )


def _assistant_asked_question(text: str | None) -> bool:
    if not text:
        return False
    lowered = text.strip().lower()
    if "?" in lowered:
        return True
    for phrase in (
        "would you like",
        "do you want",
        "should i",
        "shall i",
        "want me to",
        "can i",
        "may i",
        "need me to",
    ):
        if phrase in lowered:
            return True
    return False


def _should_skip_trivial(
    query: str, query_type: QueryType, last_assistant: str | None
) -> bool:
    if query_type != QueryType.TRIVIAL:
        return False
    normalized = query.strip().lower()
    if normalized in {"yes", "y", "ye", "yeah", "yep", "sure", "ok", "okay", "pls", "please"}:
        if _assistant_asked_question(last_assistant):
            return False
    return True


def _log_trivial_event(events_file, query: str, response: str) -> None:
    if not events_file:
        return
    record = {
        "event_type": "LocalTrivialResponse",
        "query": query,
        "response": response,
        "final_response": True,
    }
    events_file.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")


def _event_parts(event: object) -> list[dict]:
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None) if content else None
    if not parts:
        return []

    extracted: list[dict] = []
    for part in parts:
        function_call = getattr(part, "function_call", None)
        if function_call is not None:
            extracted.append(
                {
                    "type": "function_call",
                    "name": getattr(function_call, "name", None),
                    "id": getattr(function_call, "id", None),
                    "args": getattr(function_call, "args", None),
                }
            )
            continue

        function_response = getattr(part, "function_response", None)
        if function_response is not None:
            extracted.append(
                {
                    "type": "function_response",
                    "name": getattr(function_response, "name", None),
                    "id": getattr(function_response, "id", None),
                    "response": getattr(function_response, "response", None),
                }
            )
            continue

        text = getattr(part, "text", None)
        if text is not None:
            extracted.append({"type": "text", "text": text})

    return extracted


def _event_agent_label(event: object) -> str:
    for attr in ("agent_name", "agent", "name", "agent_id"):
        value = getattr(event, attr, None)
        if isinstance(value, str):
            return value
        if hasattr(value, "name"):
            label = getattr(value, "name", None)
            if isinstance(label, str):
                return label
    return ""


def _event_to_record(event: object) -> dict:
    record = {"event_type": type(event).__name__}
    parts = _event_parts(event)
    if parts:
        record["parts"] = parts
        text = " ".join(part["text"] for part in parts if part.get("type") == "text")
        if text:
            record["text"] = text
    is_final = getattr(event, "is_final_response", None)
    if callable(is_final):
        record["final_response"] = is_final()
    return record


def _summarize_response(response: object) -> dict:
    if not isinstance(response, dict):
        return {"type": type(response).__name__}
    summary: dict[str, object] = {"keys": list(response.keys())}
    if "items" in response and isinstance(response["items"], list):
        summary["items"] = len(response["items"])
    if "candidates" in response and isinstance(response["candidates"], list):
        summary["candidates"] = len(response["candidates"])
    if "spans" in response and isinstance(response["spans"], list):
        summary["spans"] = len(response["spans"])
    if "total" in response:
        summary["total"] = response["total"]
    return summary


def _friendly_tool_call(name: str, args: dict) -> str:
    if name == "adk_indexinfo":
        return "Searching our guides..."
    if name in {"adk_raginfo", "adk_raginfo_dual"}:
        return "Looking for relevant info..."
    if name == "adk_read_spans":
        count = len(args.get("chunk_ids") or [])
        return f"Pulling {count} key passage(s)..."
    if name == "adk_open_source":
        doc_id = args.get("doc_id") if isinstance(args, dict) else None
        return "Opening a relevant guide..."
    return ""


def _friendly_tool_response(name: str, response: dict) -> str:
    if name == "adk_indexinfo":
        total = response.get("total")
        items = response.get("items")
        count = len(items) if isinstance(items, list) else None
        if count is not None and total is not None:
            return f"Found {count} matches."
        if count is not None:
            return f"Found {count} matches."
    if name in {"adk_raginfo", "adk_raginfo_dual"}:
        by_source = response.get("by_source")
        if isinstance(by_source, dict):
            counts = {
                source: len(candidates)
                for source, candidates in by_source.items()
                if isinstance(candidates, list)
            }
            if counts:
                formatted = ", ".join(f"{key}: {val}" for key, val in counts.items())
                return f"Found results ({formatted})."
        candidates = response.get("candidates")
        if isinstance(candidates, list):
            return f"Found {len(candidates)} relevant results."
    if name == "adk_read_spans":
        spans = response.get("spans")
        if isinstance(spans, list):
            return f"Pulled {len(spans)} key passage(s)."
    if name == "adk_open_source":
        return "Loaded a relevant guide."
    return ""


async def _emit_progress(
    progress_callback: Optional[Callable],
    *,
    stage: str,
    message: str,
    state: Dict[str, str],
) -> None:
    if not progress_callback or not message:
        return
    if message == state.get("last_message"):
        return
    state["last_message"] = message
    try:
        await progress_callback({"type": "progress", "stage": stage, "message": message})
    except Exception:
        # Ignore progress errors to avoid breaking the main request flow.
        return


def _extract_answer_content(text: str, in_answer_tag: bool) -> tuple[str, bool]:
    """
    Extract answer content from text, tracking whether we're inside <answer> tags.
    Returns (answer_content, new_in_answer_tag).
    """
    import re
    content = ""
    remaining = text
    new_in_answer_tag = in_answer_tag

    while remaining:
        if new_in_answer_tag:
            # We're inside <answer>, look for closing tag
            closing_idx = remaining.find("</answer>")
            if closing_idx != -1:
                # Found closing tag, extract content before it
                content += remaining[:closing_idx]
                remaining = remaining[closing_idx + 9:]  # Skip </answer>
                new_in_answer_tag = False
            else:
                # No closing tag yet, entire remaining content is answer
                content += remaining
                break
        else:
            # We're outside <answer>, look for opening tag
            opening_idx = remaining.find("<answer>")
            if opening_idx != -1:
                # Found opening tag, skip content before it
                remaining = remaining[opening_idx + 8:]  # Skip <answer>
                new_in_answer_tag = True
            else:
                # No opening tag, nothing to extract
                break

    return content, new_in_answer_tag


async def _emit_stream(
    stream_callback: Optional[Callable],
    *,
    text: str,
    state: Dict[str, str],
) -> None:
    if not stream_callback or not text:
        return
    if text == state.get("last_text"):
        return
    state["last_text"] = text
    try:
        await stream_callback(text)
    except Exception:
        return


def _print_event_parts(parts: list[dict], *, state: dict) -> None:
    for part in parts:
        if part.get("type") == "function_call":
            name = part.get("name") or "unknown_tool"
            args = part.get("args") or {}
            message = _friendly_tool_call(name, args if isinstance(args, dict) else {})
            if message and message != state.get("last_message"):
                print(message)
                state["last_message"] = message
        elif part.get("type") == "function_response":
            name = part.get("name") or "unknown_tool"
            response = part.get("response") or {}
            if isinstance(response, dict):
                message = _friendly_tool_response(name, response)
                if message:
                    print(message)
                    state["last_message"] = message
            if name in {"adk_read_spans", "adk_open_source"}:
                if not state.get("answer_prompted"):
                    print("[step] Drafting answer...")
                    state["answer_prompted"] = True


def _parse_verdict(text: str) -> tuple[Optional[str], Optional[float]]:
    """Parse verdict from verifier output.

    Returns (verdict, confidence) tuple.
    Verdict is "VERIFIED", "ESCALATE", or None.
    Confidence is 0.0-1.0 if available, None otherwise.
    """
    if not text:
        return None, None

    stripped = text.strip()

    # New format: <answer>...</answer><verdict>...</verdict>
    # Extract the verdict from <verdict> tags
    import re
    verdict_match = re.search(r'<verdict>\s*(.*?)\s*</verdict>', stripped, re.DOTALL)
    if verdict_match:
        verdict_text = verdict_match.group(1).strip()
        try:
            import json
            data = json.loads(verdict_text)
            verdict = data.get("verdict", "").upper()
            confidence = data.get("confidence")
            if verdict in ("VERIFIED", "ESCALATE"):
                return verdict, confidence
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # Try JSON format first (standalone JSON with confidence)
    try:
        import json
        data = json.loads(stripped)
        verdict = data.get("verdict", "").upper()
        confidence = data.get("confidence")
        if verdict in ("VERIFIED", "ESCALATE"):
            return verdict, confidence
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Fallback to old format
    upper = stripped.upper()
    if upper.startswith("VERIFIED"):
        return "VERIFIED", None
    elif upper.startswith("ESCALATE"):
        return "ESCALATE", None
    return None, None


def _extract_answer_from_verdict(text: str) -> str:
    """Extract the original answer from verifier output.

    The verifier now returns: <answer>...</answer><verdict>...</verdict>
    This function extracts the <answer> portion.
    """
    if not text:
        return ""

    stripped = text.strip()

    # New format: <answer>...</answer><verdict>...</verdict>
    import re
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', stripped, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()

    # No answer tags found, return original text
    return stripped


async def run_query(
    query: str,
    *,
    user_id: str = "local-user",
    session_id: str = "local-session",
    events_path: Optional[Path] = None,
    print_events: bool = False,
    skip_trivial: bool = False,
    rag_mode: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
    stream_callback: Optional[Callable] = None,
    timing_tracker: Optional[Any] = None,  # TimingTracker from app.py
) -> tuple[str, Optional[str], Optional[float]]:
    content = types.Content(parts=[types.Part.from_text(text=query)], role="user")
    final_text = ""
    verdict: Optional[str] = None
    confidence: Optional[float] = None
    assistant_texts: list[str] = []
    events_file = open(events_path, "w", encoding="utf-8") if events_path else None
    print_state = {"last_message": None, "answer_prompted": False}
    progress_state: Dict[str, str] = {"last_message": ""}
    answer_progress_sent = False
    tool_markers: Dict[str, str] = {}  # Track active tool calls for timing
    evidence_ready = False
    stream_state: Dict[str, str] = {"last_text": ""}
    stream_buffer = ""
    answer_buffer = ""  # Accumulated answer content (inside <answer> tags)
    in_answer_tag = False  # Track if we're currently inside <answer> tags

    collected_spans: list[EvidenceSpan] = []
    

    if print_events:
        print(f"[step] Processing query: {query}")
    try:
        await _emit_progress(
            progress_callback,
            stage="routing",
            message="Routing query...",
            state=progress_state,
        )
        if skip_trivial:
            query_type, _ = classify_query(query)
            if _should_skip_trivial(query, query_type, None):
                response = _trivial_response(query)
                if print_events:
                    print("[step] Detected trivial query; skipping retrieval.")
                _log_trivial_event(events_file, query, response)
                return response, None
        _disable_local_proxy()
        _suppress_adk_warnings()
        _require_api_key()
        session_service = InMemorySessionService()
        runner = Runner(
            agent=root_agent,
            app_name="profectus-support",
            session_service=session_service,
            auto_create_session=True,
        )
        with rag_mode_override(rag_mode):
            async for event in runner.run_async(
                user_id=user_id, session_id=session_id, new_message=content
            ):
                if events_file:
                    record = _event_to_record(event)
                    events_file.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")
                if print_events:
                    parts = _event_parts(event)
                    if parts:
                        _print_event_parts(parts, state=print_state)

                # Send progress updates and track timing
                parts = _event_parts(event)
                for part in parts:
                    if part.get("type") == "function_call":
                        name = part.get("name")
                        args = part.get("args") or {}
                        marker = f"tool_{name}"
                        if timing_tracker:
                            timing_tracker.mark(marker)
                        tool_markers[name] = marker

                        if progress_callback:
                            stage = "tool"
                            if name == "adk_indexinfo":
                                stage = "keyword_search"
                            elif name in {"adk_raginfo", "adk_raginfo_dual"}:
                                stage = "semantic_search"
                            elif name == "adk_read_spans":
                                stage = "reading"
                            elif name == "adk_open_source":
                                stage = "opening"
                            message = _friendly_tool_call(name, args if isinstance(args, dict) else {})
                            if not message:
                                message = "Calling retrieval tool..."
                            await _emit_progress(
                                progress_callback,
                                stage=stage,
                                message=message,
                                state=progress_state,
                            )

                    elif part.get("type") == "function_response":
                        name = part.get("name")

                        response_payload = part.get("response") or {}
                        
                        if name == "adk_read_spans":
                            raw_spans = response_payload.get("spans", [])
                            for s in raw_spans:
                                try:
                                    collected_spans.append(EvidenceSpan(**s))
                                except Exception as e:
                                    print(f"[NOTICE]Error converting EvidenceSpan (read_spans): {e}")

                        elif name in {"adk_raginfo", "adk_raginfo_dual"}:
                            candidates = response_payload.get("candidates", [])
                            
                            # Fallback: if 'candidates' is empty, try retrieving it from 'by_source'
                            if not candidates:
                                by_source = response_payload.get("by_source", {})
                                for src_list in by_source.values():
                                    if isinstance(src_list, list):
                                        candidates.extend(src_list)

                            # Local classes to ensure compatibility and avoid strict validations
                            class FlexibleProvenance:
                                def __init__(self, source):
                                    self.source = source

                            class FlexibleEvidenceSpan:
                                def __init__(self, span_id, text, provenance, score):
                                    self.span_id = span_id
                                    self.text = text
                                    self.provenance = provenance
                                    self.score = score

                            for c in candidates:
                                try:
                                    #1. Robust text extraction (attempts multiple keys)
                                    text_content = c.get("content") or c.get("snippet") or c.get("text") or ""
                                    if not text_content: 
                                        continue 

                                    #2. Extracting metadata with default values
                                    source_val = c.get("source", "Unknown Source")
                                    chunk_id = str(c.get("chunk_id") or c.get("id") or "search_snippet")
                                    score_val = float(c.get("score", 1.0))
                                    
                                    #3. Creating the Flexible Object
                                    prov_obj = FlexibleProvenance(source_val)
                                    span_obj = FlexibleEvidenceSpan(chunk_id, text_content, prov_obj, score_val)
                                    
                                    collected_spans.append(span_obj)
                                    
                                except Exception:
                                    # In production, we ignore individual failures to avoid disrupting the workflow.
                                    continue

                        if timing_tracker and name in tool_markers:
                            timing_tracker.record_step(f"tool_{name}", tool_markers[name])
                            del tool_markers[name]
                        
                        if name in {"adk_read_spans", "adk_open_source"}:
                            evidence_ready = True

                        if progress_callback:
                            response = part.get("response") or {}
                            message = _friendly_tool_response(name, response if isinstance(response, dict) else {})
                            stage = "tool_result"
                            if name in {"adk_raginfo", "adk_raginfo_dual"}:
                                stage = "found"
                                if not message:
                                    by_source = response.get("by_source", {})
                                    if isinstance(by_source, dict):
                                        total = sum(len(items) for items in by_source.values() if isinstance(items, list))
                                        message = f"Found {total} result(s)."
                            elif name == "adk_read_spans":
                                stage = "retrieved"
                                if not message:
                                    spans = response.get("spans", [])
                                    if isinstance(spans, list):
                                        message = f"Retrieved {len(spans)} span(s)."
                            await _emit_progress(
                                progress_callback,
                                stage=stage,
                                message=message,
                                state=progress_state,
                            )

                parts = _event_parts(event)
                for part in parts:
                    if part.get("type") == "text":
                        text = part.get("text") or ""
                        if text:
                            if progress_callback and not answer_progress_sent:
                                await _emit_progress(
                                    progress_callback,
                                    stage="drafting",
                                    message="Drafting answer...",
                                    state=progress_state,
                                )
                                answer_progress_sent = True
                            assistant_texts.append(text)
                            if stream_callback:
                                # Always buffer text so we can detect <answer> tags,
                                # but only stream content extracted from <answer>.
                                if stream_buffer and text.startswith(stream_buffer):
                                    stream_buffer = text
                                elif stream_buffer and stream_buffer.startswith(text):
                                    pass
                                else:
                                    stream_buffer = stream_buffer + text

                                # Extract answer content from the accumulated buffer
                                answer_chunk, in_answer_tag = _extract_answer_content(stream_buffer, in_answer_tag)

                                if answer_chunk:
                                    normalized_chunk = answer_chunk.strip()
                                    if normalized_chunk:
                                        answer_buffer = normalized_chunk
                                        await _emit_stream(
                                            stream_callback,
                                            text=answer_buffer,
                                            state=stream_state,
                                        )
                if event.is_final_response():
                    for part in parts:
                        if part.get("type") == "text":
                            final_text = part.get("text") or final_text
                            break
                    verdict, _ = _parse_verdict(final_text)
                    
                    # Extract the actual answer from the verdict output
                    extracted_answer = _extract_answer_from_verdict(final_text)
                    if extracted_answer and extracted_answer != final_text:
                        final_text = extracted_answer
    finally:
        if events_file:
            events_file.close()
    # Fallback: if we didn't get a proper answer from the verifier, try to find one in assistant_texts
    if not final_text or final_text.strip().startswith("<verdict>") or final_text.strip().startswith('{"verdict"') or not final_text.strip():
        # First try the answer_buffer (from streaming extraction)
        if answer_buffer and answer_buffer.strip():
            final_text = answer_buffer.strip()
        elif assistant_texts:
            for candidate in reversed(assistant_texts):
                stripped = candidate.strip()
                upper = stripped.upper()
                # Accept any non-empty text that isn't a verdict/escalation marker
                if stripped and not (upper.startswith("VERIFIED") or upper.startswith("ESCALATE") or stripped.startswith("<") or stripped.startswith('{"verdict"')):
                    final_text = candidate
                    break
            # If still empty, just use the last assistant text (better than nothing)
            if not final_text or not final_text.strip():
                for candidate in reversed(assistant_texts):
                    if candidate.strip():
                        final_text = candidate
                        break
    if print_events and not print_state.get("answer_prompted"):
        print("[step] Drafting answer...")
    # Always ensure we have a non-empty final_text to return
    if not final_text or not final_text.strip():
        if answer_buffer and answer_buffer.strip():
            final_text = answer_buffer.strip()
        elif assistant_texts:
            # Last resort: use the most recent assistant text
            for candidate in reversed(assistant_texts):
                if candidate.strip():
                    final_text = candidate
                    break
    
    if stream_callback and final_text and final_text.strip():
        stream_text = answer_buffer if answer_buffer and answer_buffer.strip() else final_text
        
        await _emit_stream(stream_callback, text=stream_text.strip(), state=stream_state)
    if final_text and not skip_trivial:
        verification_result = verify_evidence(
            spans=collected_spans,
            query=query,
            confidence_threshold=0.5
        )

        confidence = verification_result.confidence

        if not verification_result.ok:
            verdict = "ESCALATE"
            print(f"[Trust System]Scaled by low confidence ({confidence:.2f}). Reason: {verification_result.reason}")
        else:
            verdict = "ANSWER"
    return final_text, verdict, confidence


def run_query_sync(
    query: str,
    *,
    events_path: Optional[Path] = None,
    print_events: bool = False,
    skip_trivial: bool = True,
) -> tuple[str, Optional[str], Optional[float]]:
    return asyncio.run(
        run_query(
            query,
            events_path=events_path,
            print_events=print_events,
            skip_trivial=skip_trivial,
        )
    )


async def run_interactive(
    *,
    user_id: str,
    session_id: str,
    events_path: Optional[Path],
    print_events: bool,
    initial_query: Optional[str],
    skip_trivial: bool,
) -> None:
    _disable_local_proxy()
    _suppress_adk_warnings()
    _require_api_key()
    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent,
        app_name="profectus-support",
        session_service=session_service,
        auto_create_session=True,
    )
    events_file = open(events_path, "a", encoding="utf-8") if events_path else None
    try:
        last_assistant: Optional[str] = None
        if initial_query:
            if print_events:
                print(f"[step] Processing query: {initial_query}")
            content = types.Content(parts=[types.Part.from_text(text=initial_query)], role="user")
            final_text = ""
            verdict: Optional[str] = None
            confidence: Optional[float] = None
            assistant_texts: list[str] = []
            print_state = {"last_message": None, "answer_prompted": False}
            if skip_trivial:
                query_type, _ = classify_query(initial_query)
                if _should_skip_trivial(initial_query, query_type, None):
                    response = _trivial_response(initial_query)
                    if print_events:
                        print("[step] Detected trivial query; skipping retrieval.")
                    _log_trivial_event(events_file, initial_query, response)
                    print(f"assistant> {response}")
                    print("")
                    last_assistant = response
                    initial_query = None
            if initial_query:
                async for event in runner.run_async(
                    user_id=user_id, session_id=session_id, new_message=content
                ):
                    if events_file:
                        record = _event_to_record(event)
                        events_file.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")
                    if print_events:
                        parts = _event_parts(event)
                        if parts:
                            _print_event_parts(parts, state=print_state)
                    parts = _event_parts(event)
                    for part in parts:
                        if part.get("type") == "text":
                            text = part.get("text") or ""
                            if text:
                                assistant_texts.append(text)
                    if event.is_final_response():
                        for part in parts:
                            if part.get("type") == "text":
                                final_text = part.get("text") or final_text
                                break
                        verdict, confidence = _parse_verdict(final_text)
                        # Extract the actual answer from the verdict output
                        extracted_answer = _extract_answer_from_verdict(final_text)
                        if extracted_answer and extracted_answer != final_text:
                            final_text = extracted_answer
                # Fallback: if we didn't get a proper answer from the verifier
                if not final_text or final_text.strip().startswith("<verdict>") or final_text.strip().startswith('{"verdict"'):
                    if assistant_texts:
                        for candidate in reversed(assistant_texts):
                            stripped = candidate.strip()
                            upper = stripped.upper()
                            if stripped and not (upper.startswith("VERIFIED") or upper.startswith("ESCALATE") or stripped.startswith("<")):
                                final_text = candidate
                                break
                if print_events and not print_state.get("answer_prompted"):
                    print("[step] Drafting answer...")
                print(f"assistant> {final_text}")
                if verdict:
                    if print_events:
                        print("[step] Verifying answer...")
                    conf_str = f" (confidence: {confidence:.1f})" if confidence is not None else ""
                    print(f"[verification] {verdict}{conf_str}")
                print("")
                if final_text:
                    last_assistant = final_text

        while True:
            try:
                query = input("user> ").strip()
            except EOFError:
                break
            if not query or query.lower() in {"exit", "quit"}:
                break
            content = types.Content(parts=[types.Part.from_text(text=query)], role="user")
            final_text = ""
            verdict = None
            confidence = None
            assistant_texts = []
            print_state = {"last_message": None, "answer_prompted": False}
            if print_events:
                print(f"[step] Processing query: {query}")
            if skip_trivial:
                query_type, _ = classify_query(query)
                if _should_skip_trivial(query, query_type, last_assistant):
                    response = _trivial_response(query)
                    if print_events:
                        print("[step] Detected trivial query; skipping retrieval.")
                    _log_trivial_event(events_file, query, response)
                    print(f"assistant> {response}")
                    print("")
                    continue
            async for event in runner.run_async(
                user_id=user_id, session_id=session_id, new_message=content
            ):
                if events_file:
                    record = _event_to_record(event)
                    events_file.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")
                if print_events:
                    parts = _event_parts(event)
                    if parts:
                        _print_event_parts(parts, state=print_state)
                parts = _event_parts(event)
                for part in parts:
                    if part.get("type") == "text":
                        text = part.get("text") or ""
                        if text:
                            assistant_texts.append(text)
                if event.is_final_response():
                    for part in parts:
                        if part.get("type") == "text":
                            final_text = part.get("text") or final_text
                            break
                    verdict, confidence = _parse_verdict(final_text)
                    # Extract the actual answer from the verdict output
                    extracted_answer = _extract_answer_from_verdict(final_text)
                    if extracted_answer and extracted_answer != final_text:
                        final_text = extracted_answer
            # Fallback: if we didn't get a proper answer from the verifier
            if not final_text or final_text.strip().startswith("<verdict>") or final_text.strip().startswith('{"verdict"'):
                if assistant_texts:
                    for candidate in reversed(assistant_texts):
                        stripped = candidate.strip()
                        upper = stripped.upper()
                        if stripped and not (upper.startswith("VERIFIED") or upper.startswith("ESCALATE") or stripped.startswith("<")):
                            final_text = candidate
                            break
            if print_events and not print_state.get("answer_prompted"):
                print("[step] Drafting answer...")
            print(f"assistant> {final_text}")
            if verdict:
                if print_events:
                    print("[step] Verifying answer...")
                conf_str = f" (confidence: {confidence:.1f})" if confidence is not None else ""
                print(f"[verification] {verdict}{conf_str}")
            print("")
            if final_text:
                last_assistant = final_text
    finally:
        if events_file:
            events_file.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the ADK agent for a single query.")
    parser.add_argument("query", nargs="?", help="User query text")
    parser.add_argument(
        "--log-events",
        action="store_true",
        help="Write event stream to a JSONL file in data/.",
    )
    parser.add_argument(
        "--print-events",
        action="store_true",
        help="Print tool calls/responses to stdout.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run an interactive session that preserves context between turns.",
    )
    parser.add_argument(
        "--no-trivial-skip",
        action="store_true",
        help="Disable fast-path for trivial queries (deprecated).",
    )
    parser.add_argument(
        "--skip-trivial",
        action="store_true",
        help="Enable fast-path for trivial/off-topic queries.",
    )
    parser.add_argument("--user-id", default="local-user")
    parser.add_argument("--session-id", default="local-session")
    parser.add_argument(
        "--events-path",
        type=str,
        default=None,
        help="Override the event log path (default: data/adk_events_<timestamp>.jsonl).",
    )
    args = parser.parse_args()

    events_path = None
    if args.log_events or args.events_path:
        if args.events_path:
            events_path = Path(args.events_path)
        else:
            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            events_path = Path(__file__).resolve().parents[1] / "data" / f"adk_events_{timestamp}.jsonl"
        events_path.parent.mkdir(parents=True, exist_ok=True)
    skip_trivial = bool(args.skip_trivial)
    if args.no_trivial_skip:
        skip_trivial = False

    if args.interactive:
        asyncio.run(
            run_interactive(
                user_id=args.user_id,
                session_id=args.session_id,
                events_path=events_path,
                print_events=args.print_events,
                initial_query=args.query,
                skip_trivial=skip_trivial,
            )
        )
    else:
        if not args.query:
            parser.error("query is required unless --interactive is set")
        response, verdict, confidence = run_query_sync(
            args.query,
            events_path=events_path,
            print_events=args.print_events,
            skip_trivial=skip_trivial,
        )
        print(response)
        if verdict:
            conf_str = f" (confidence: {confidence:.1f})" if confidence is not None else ""
            print(f"[verification] {verdict}{conf_str}")
    if events_path:
        print("")
        print(f"Event log written to {events_path}")
