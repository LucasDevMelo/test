from __future__ import annotations

import argparse
import json
import os
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple
from xml.etree import ElementTree as ET

import pandas as pd
from openpyxl.utils.cell import coordinate_from_string, column_index_from_string

from profectus_ai.adk_runner import run_query_sync
from profectus_ai.query_normalization import normalize_query


THREAD_COMMENTS_PATH = "xl/threadedComments/threadedComment1.xml"
THREAD_NS = {
    "x18tc": "http://schemas.microsoft.com/office/spreadsheetml/2018/threadedcomments"
}


def load_threaded_comments(xlsx_path: Path) -> Dict[str, List[str]]:
    comments: Dict[str, List[str]] = {}
    if not xlsx_path.exists():
        return comments
    with zipfile.ZipFile(xlsx_path) as zf:
        if THREAD_COMMENTS_PATH not in zf.namelist():
            return comments
        data = zf.read(THREAD_COMMENTS_PATH)
    root = ET.fromstring(data)
    for node in root.findall("x18tc:threadedComment", THREAD_NS):
        ref = node.attrib.get("ref")
        text_node = node.find("x18tc:text", THREAD_NS)
        text = (text_node.text or "").strip() if text_node is not None else ""
        if ref and text:
            comments.setdefault(ref, []).append(text)
    return comments


def map_comments_to_rows(
    comments: Dict[str, List[str]],
    columns: List[str],
) -> Dict[int, List[str]]:
    mapped: Dict[int, List[str]] = {}
    for ref, texts in comments.items():
        col_letter, row = coordinate_from_string(ref)
        col_idx = column_index_from_string(col_letter) - 1
        col_name = columns[col_idx] if 0 <= col_idx < len(columns) else col_letter
        label = col_name
        if isinstance(label, str) and label.lower().startswith("unnamed"):
            label = ""
        df_index = row - 2  # header is row 1
        if df_index < 0:
            continue
        for text in texts:
            entry = f"{label}: {text}" if label else text
            mapped.setdefault(df_index, []).append(entry)
    return mapped


def extract_comments(xlsx_path: Path, df: pd.DataFrame) -> List[str]:
    comments = load_threaded_comments(xlsx_path)
    by_row = map_comments_to_rows(comments, list(df.columns))
    out: List[str] = []
    for idx in range(len(df)):
        items = by_row.get(idx, [])
        out.append(" | ".join(items) if items else "")
    return out


def _has_citation(text: str) -> bool:
    return bool(re.search(r"https?://", text or ""))


def _get_api_key() -> str | None:
    key = os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    return None


def evaluate_with_llm(
    *,
    query: str,
    previous_output: str,
    previous_comment: str,
    modified_output: str,
    new_output: str,
    model: str,
) -> Tuple[str, str]:
    try:
        from google import genai
    except Exception:
        return "skipped", "google.genai not available"

    api_key = _get_api_key()
    if not api_key:
        return "skipped", "GOOGLE_API_KEY not set"

    baseline = modified_output.strip() if modified_output.strip() else previous_output
    prompt = (
        "You are evaluating whether a new answer improved compared to a baseline answer.\n"
        "The reviewer comment refers to the ORIGINAL answer; a MODIFIED answer may exist and is the most\n"
        "recent improved baseline if present. Judge the new answer against the baseline and ensure it does\n"
        "not regress on the reviewer comment.\n"
        "Return JSON only: {\"judgement\":\"better|same|worse|unsure\",\"rationale\":\"...\"}\n\n"
        f"Query: {query}\n\n"
        f"Original Answer:\n{previous_output}\n\n"
        f"Modified Answer (if any):\n{modified_output}\n\n"
        f"Baseline Answer Used for Comparison:\n{baseline}\n\n"
        f"Reviewer Comment:\n{previous_comment}\n\n"
        f"New Answer:\n{new_output}\n"
    )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model, contents=prompt)
    text = (response.text or "").strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return "unsure", "could not parse eval response"
    try:
        payload = json.loads(match.group(0))
        return payload.get("judgement", "unsure"), payload.get("rationale", "")
    except json.JSONDecodeError:
        return "unsure", "invalid eval json"


def _safe_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    candidate = path.with_name(f"{path.stem}-{timestamp}{path.suffix}")
    return candidate


def _should_retry(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in [
            "rate limit",
            "429",
            "resource_exhausted",
            "temporarily unavailable",
            "timeout",
        ]
    )


def _parse_event_log(path: Path) -> Dict[str, object]:
    summary = {
        "tool_calls": {},
        "indexinfo_total": "",
        "indexinfo_items": "",
        "indexinfo_urls": "",
        "raginfo_candidates": "",
        "raginfo_by_source": "",
        "raginfo_weights": "",
        "read_spans": "",
        "evidence_urls": "",
        "open_source_calls": 0,
    }
    if not path.exists():
        return summary

    index_urls = []
    evidence_urls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        parts = record.get("parts") or []
        for part in parts:
            if part.get("type") == "function_call":
                name = part.get("name", "unknown_tool")
                summary["tool_calls"][name] = summary["tool_calls"].get(name, 0) + 1
            if part.get("type") != "function_response":
                continue
            name = part.get("name")
            response = part.get("response") or {}
            if name == "adk_indexinfo":
                summary["indexinfo_total"] = response.get("total", "")
                items = response.get("items") or []
                summary["indexinfo_items"] = len(items)
                for item in items:
                    url = item.get("doc_url") or item.get("provenance", {}).get("url")
                    if url:
                        index_urls.append(url)
            if name in {"adk_raginfo", "adk_raginfo_dual"}:
                candidates = response.get("candidates") or []
                summary["raginfo_candidates"] = len(candidates)
                by_source = response.get("by_source")
                if isinstance(by_source, dict):
                    counts = {
                        source: len(items)
                        for source, items in by_source.items()
                        if isinstance(items, list)
                    }
                    summary["raginfo_by_source"] = json.dumps(counts, ensure_ascii=True)
                weights = response.get("source_weights")
                if isinstance(weights, dict):
                    summary["raginfo_weights"] = json.dumps(weights, ensure_ascii=True)
            if name == "adk_read_spans":
                spans = response.get("spans") or []
                summary["read_spans"] = len(spans)
                for span in spans:
                    prov = span.get("provenance") or {}
                    url = prov.get("url")
                    if url:
                        evidence_urls.append(url)
            if name == "adk_open_source":
                summary["open_source_calls"] = summary.get("open_source_calls", 0) + 1
    summary["indexinfo_urls"] = " | ".join(sorted(set(index_urls)))
    summary["evidence_urls"] = " | ".join(sorted(set(evidence_urls)))
    summary["tool_calls"] = json.dumps(summary["tool_calls"], ensure_ascii=True)
    return summary


def _run_query_with_retry(
    *,
    query: str,
    events_path: Path,
    retries: int,
    backoff_s: float,
) -> Tuple[str, str, float, str]:
    """Run query with retry logic.

    Returns:
        Tuple of (answer, verdict, confidence, error).
        On error, answer and verdict are empty strings, confidence is 0.0, and error contains the message.
    """
    attempt = 0
    last_error = ""
    while attempt <= retries:
        try:
            answer, verdict, confidence = run_query_sync(
                query,
                events_path=events_path,
                print_events=False,
            )
            return answer, verdict or "", confidence or 0.0, ""
        except Exception as exc:
            last_error = str(exc)
            if attempt >= retries or not _should_retry(exc):
                break
            sleep_for = backoff_s * (2 ** attempt)
            print(f"[eval] retrying after error: {exc} (sleep {sleep_for:.1f}s)")
            time.sleep(sleep_for)
            attempt += 1
    return "", "", 0.0, last_error


def run_eval(
    *,
    input_path: Path,
    output_path: Path,
    limit: int | None,
    model: str,
    do_eval: bool,
    workers: int,
    retries: int,
    backoff_s: float,
    events_dir: Path,
) -> None:
    print(f"[eval] loading: {input_path}")
    df = pd.read_excel(input_path)
    print(f"[eval] rows: {len(df)}")
    comments = extract_comments(input_path, df)
    df["Original Comment"] = comments

    new_answers: List[str] = [""] * len(df)
    verdicts: List[str] = [""] * len(df)
    confidences: List[float] = [0.0] * len(df)
    citations: List[bool] = [False] * len(df)
    eval_judgements: List[str] = ["skipped"] * len(df)
    eval_notes: List[str] = ["not run"] * len(df)
    errors: List[str] = [""] * len(df)
    normalized_queries: List[str] = [""] * len(df)
    tool_calls: List[str] = [""] * len(df)
    indexinfo_items: List[object] = [""] * len(df)
    indexinfo_total: List[object] = [""] * len(df)
    indexinfo_urls: List[str] = [""] * len(df)
    raginfo_candidates: List[object] = [""] * len(df)
    raginfo_by_source: List[str] = [""] * len(df)
    raginfo_weights: List[str] = [""] * len(df)
    read_spans: List[object] = [""] * len(df)
    evidence_urls: List[str] = [""] * len(df)
    open_source_calls: List[object] = [""] * len(df)
    event_log_paths: List[str] = [""] * len(df)

    events_dir.mkdir(parents=True, exist_ok=True)
    max_rows = min(limit, len(df)) if limit is not None else len(df)
    print(f"[eval] workers={workers} retries={retries} backoff={backoff_s}s")

    def _process_row(idx: int) -> None:
        row = df.iloc[idx]
        query = row.get("Query", "")
        normalized = normalize_query(query)
        normalized_queries[idx] = normalized
        events_path = events_dir / f"row_{idx+1}.jsonl"
        event_log_paths[idx] = str(events_path)
        answer, verdict, confidence, error = _run_query_with_retry(
            query=query,
            events_path=events_path,
            retries=retries,
            backoff_s=backoff_s,
        )
        if error:
            eval_judgements[idx] = "error"
            eval_notes[idx] = error
            errors[idx] = error
        else:
            if do_eval:
                modified_output = row.get("Modified Output", "") or ""
                judgement, rationale = evaluate_with_llm(
                    query=query,
                    previous_output=row.get("Output", "") or "",
                    previous_comment=comments[idx] if idx < len(comments) else "",
                    modified_output=modified_output if isinstance(modified_output, str) else "",
                    new_output=answer,
                    model=model,
                )
                eval_judgements[idx] = judgement
                eval_notes[idx] = rationale
            else:
                eval_judgements[idx] = "skipped"
                eval_notes[idx] = "eval disabled"
        new_answers[idx] = answer
        verdicts[idx] = verdict
        confidences[idx] = confidence
        citations[idx] = _has_citation(answer)

        summary = _parse_event_log(events_path)
        tool_calls[idx] = summary.get("tool_calls", "")
        indexinfo_items[idx] = summary.get("indexinfo_items", "")
        indexinfo_total[idx] = summary.get("indexinfo_total", "")
        indexinfo_urls[idx] = summary.get("indexinfo_urls", "")
        raginfo_candidates[idx] = summary.get("raginfo_candidates", "")
        raginfo_by_source[idx] = summary.get("raginfo_by_source", "")
        raginfo_weights[idx] = summary.get("raginfo_weights", "")
        read_spans[idx] = summary.get("read_spans", "")
        evidence_urls[idx] = summary.get("evidence_urls", "")
        open_source_calls[idx] = summary.get("open_source_calls", "")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_row, idx): idx for idx in range(max_rows)}
        for i, future in enumerate(as_completed(futures), start=1):
            idx = futures[future]
            try:
                future.result()
            except Exception as exc:
                errors[idx] = str(exc)
                eval_judgements[idx] = "error"
                eval_notes[idx] = str(exc)
            if i % 5 == 0 or i == max_rows:
                print(f"[eval] completed {i}/{max_rows}")

    df["New Answer"] = new_answers
    df["Verification"] = verdicts
    df["Confidence"] = confidences
    df["Has Citation"] = citations
    df["Eval Judgement"] = eval_judgements
    df["Eval Notes"] = eval_notes
    df["Eval Error"] = errors
    df["Normalized Query"] = normalized_queries
    df["Tool Calls"] = tool_calls
    df["Indexinfo Items"] = indexinfo_items
    df["Indexinfo Total"] = indexinfo_total
    df["Indexinfo URLs"] = indexinfo_urls
    df["Raginfo Candidates"] = raginfo_candidates
    df["Raginfo By Source"] = raginfo_by_source
    df["Raginfo Weights"] = raginfo_weights
    df["Read Spans"] = read_spans
    df["Evidence URLs"] = evidence_urls
    df["Open Source Calls"] = open_source_calls
    df["Event Log Path"] = event_log_paths

    output_path.parent.mkdir(parents=True, exist_ok=True)
    target = _safe_output_path(output_path)
    try:
        df.to_excel(target, index=False)
        print(f"[eval] wrote {target}")
    except PermissionError:
        fallback = _safe_output_path(output_path)
        df.to_excel(fallback, index=False)
        print(f"[eval] wrote {fallback} (original locked)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate new architecture on an XLSX test set.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "Final Output test.xlsx",
        help="Input XLSX file path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "Final Output test - new_arch.xlsx",
        help="Output XLSX file path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows (for quick tests)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("PROFECTUS_ADK_MODEL", "gemini-2.5-flash"),
        help="Model to use for evaluation (if enabled)",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip LLM evaluation step",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries per query for transient errors",
    )
    parser.add_argument(
        "--backoff",
        type=float,
        default=5.0,
        help="Initial backoff seconds between retries",
    )
    parser.add_argument(
        "--events-dir",
        type=Path,
        default=Path(__file__).parent / "events",
        help="Directory for per-query ADK event logs",
    )
    args = parser.parse_args()

    run_eval(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        model=args.model,
        do_eval=not args.no_eval,
        workers=args.workers,
        retries=args.retries,
        backoff_s=args.backoff,
        events_dir=args.events_dir,
    )


if __name__ == "__main__":
    main()
