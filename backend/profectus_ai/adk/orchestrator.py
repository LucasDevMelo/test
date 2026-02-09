"""ADK orchestrator with evidence-first architecture.

Defines the ADK agent graph for the Profectus support agent:
1. Parallel retrieval (indexinfo + raginfo)
2. Evidence reader (REQUIRED - fetches exact text)
3. Answer generator (uses evidence, no direct tools)
4. Verifier (checks evidence coverage)
"""
from __future__ import annotations

import os
from typing import Any, Callable, Optional

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from profectus_ai.adk.prompts import (
    ANSWER_WITH_CITATIONS_INSTRUCTION,
    EVIDENCE_READER_INSTRUCTION,
    INDEXINFO_INSTRUCTION,
    RAGINFO_INSTRUCTION,
    VERIFY_INSTRUCTION,
    build_router_prompt,
    build_orchestrator_prompt,
)
from profectus_ai.adk.tool_wrappers import (
    adk_indexinfo,
    adk_open_source,
    adk_raginfo_dual,
    adk_read_spans,
)


DEFAULT_MODEL = os.environ.get("PROFECTUS_ADK_MODEL", "gemini-2.5-flash")


def _stash_tool_result(
    key: str,
) -> Callable[..., dict]:
    """Create a callback that stashes tool results in session state."""
    def _callback(*args: Any, **kwargs: Any) -> dict:
        _, _, tool_context, result = _normalize_tool_callback(args, kwargs)
        if tool_context is not None:
            tool_context.state[f"temp:{key}"] = {} if result is None else result
        return result

    return _callback


def _stash_by_tool(*args: Any, **kwargs: Any) -> dict:
    """Callback that stashes results under the tool's name."""
    tool, _, tool_context, result = _normalize_tool_callback(args, kwargs)
    if tool_context is not None:
        tool_name = getattr(tool, "name", "unknown_tool")
        tool_context.state[f"temp:{tool_name}"] = {} if result is None else result
    return result


def _normalize_tool_callback(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[Optional[BaseTool], Optional[dict], Optional[ToolContext], Any]:
    """Normalize positional and keyword args from tool callbacks."""
    tool = kwargs.get("tool")
    tool_args = kwargs.get("tool_args")
    tool_context = kwargs.get("tool_context")
    result = kwargs.get("result")

    if tool is None and len(args) > 0:
        tool = args[0]
    if tool_args is None and len(args) > 1:
        tool_args = args[1]
    if tool_context is None and len(args) > 2:
        tool_context = args[2]
    if result is None and len(args) > 3:
        result = args[3]

    return tool, tool_args, tool_context, result


def build_root_agent() -> SequentialAgent:
    """Build the root ADK agent with evidence-first architecture.

    Architecture:
    1. ParallelAgent: indexinfo + raginfo run concurrently
    2. evidence_reader: MUST fetch evidence (read_spans/open_source)
    3. answer_agent: Generates answer (NO tools - uses temp:evidence)
    4. verify_agent: Checks evidence coverage

    Key change: answer_agent has NO tools, forcing it to use
    the evidence fetched by evidence_reader.
    """
    # Phase 1: Parallel retrieval
    router_agent = LlmAgent(
        name="router_agent",
        model=DEFAULT_MODEL,
        instruction=build_router_prompt(),
        tools=[],
        output_key="temp:route",
    )

    index_agent = LlmAgent(
        name="indexinfo_agent",
        model=DEFAULT_MODEL,
        instruction=INDEXINFO_INSTRUCTION,
        tools=[adk_indexinfo],
        after_tool_callback=_stash_tool_result("indexinfo"),
        output_key="temp:indexinfo_summary",
    )

    rag_agent = LlmAgent(
        name="raginfo_agent",
        model=DEFAULT_MODEL,
        instruction=RAGINFO_INSTRUCTION,
        tools=[adk_raginfo_dual],
        after_tool_callback=_stash_tool_result("raginfo"),
        output_key="temp:raginfo_summary",
    )

    parallel_retrieval = ParallelAgent(
        name="parallel_retrieval",
        sub_agents=[index_agent, rag_agent],
    )

    # Phase 2: Evidence reader (REQUIRED step)
    evidence_reader = LlmAgent(
        name="evidence_reader",
        model=DEFAULT_MODEL,
        instruction=EVIDENCE_READER_INSTRUCTION,
        tools=[adk_read_spans, adk_open_source],
        after_tool_callback=_stash_by_tool,
        output_key="temp:evidence_summary",
    )

    # Phase 3: Answer generator (NO TOOLS - must use evidence)
    answer_agent = LlmAgent(
        name="answer_agent",
        model=DEFAULT_MODEL,
        instruction=ANSWER_WITH_CITATIONS_INSTRUCTION,
        tools=[],  # Key: no tools = must use temp:evidence
        output_key="temp:answer",
    )

    # Phase 4: Verifier
    verify_agent = LlmAgent(
        name="verify_agent",
        model=DEFAULT_MODEL,
        instruction=VERIFY_INSTRUCTION,
        output_key="temp:verification",
    )

    # Compose into sequential flow
    root_agent = SequentialAgent(
        name="profectus_root_agent",
        sub_agents=[router_agent, parallel_retrieval, evidence_reader, answer_agent, verify_agent],
    )

    return root_agent


# Build the agent on module load
root_agent = build_root_agent()
