"""Hierarchy summary generator for system prompts.

Generates a tree-map overview of available content from the index
to be included in the agent's system prompt, giving it awareness
of what knowledge is available before retrieval.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List

from profectus_ai.evidence.store import get_index_store

logger = logging.getLogger(__name__)


def build_hierarchy_summary(max_categories: int = 10, max_examples: int = 3) -> str:
    """Build a text summary of available content hierarchy.

    Args:
        max_categories: Maximum categories to show per source.
        max_examples: Maximum example titles per category.

    Returns:
        Formatted markdown text describing available content.
    """
    index = get_index_store()
    entries = index.all_entries()

    if not entries:
        return "## Available Content\n\nNo content available.\n"

    # Group by source
    by_source: Dict[str, List] = defaultdict(list)
    for entry in entries:
        by_source[entry.source].append(entry)

    lines = ["## Available Content\n"]

    for source in sorted(by_source.keys()):
        items = by_source[source]
        lines.append(f"### {source} ({len(items)} documents)\n")

        # Group by top-level hierarchy (first path component)
        by_category: Dict[str, List] = defaultdict(list)
        for item in items:
            if item.hierarchy:
                category = item.hierarchy[0]
            else:
                # Use first significant word from title as fallback
                category = _extract_category_from_title(item.title)
            by_category[category].append(item)

        # Sort categories by document count (most first)
        sorted_categories = sorted(
            by_category.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:max_categories]

        for category, cat_items in sorted_categories:
            # Get example titles
            example_titles = [
                _truncate(item.title, 50)
                for item in cat_items[:max_examples]
            ]
            examples = ", ".join(example_titles)
            if len(cat_items) > max_examples:
                examples += "..."
            lines.append(f"- **{category}** ({len(cat_items)}): {examples}")

        lines.append("")  # Blank line between sources

    return "\n".join(lines)


def build_topic_tags_summary(max_tags: int = 20) -> str:
    """Build a summary of common tags/topics across all content.

    Returns:
        Formatted text listing the most common tags.
    """
    index = get_index_store()
    entries = index.all_entries()

    # Count tag frequency
    tag_counts: Dict[str, int] = defaultdict(int)
    for entry in entries:
        for tag in entry.tags:
            normalized = tag.lower().strip()
            if normalized:
                tag_counts[normalized] += 1

    # Sort by frequency
    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:max_tags]

    if not sorted_tags:
        return "## Common Topics\n\nNo tags available.\n"

    lines = ["## Common Topics\n"]
    tag_list = [f"{tag} ({count})" for tag, count in sorted_tags]
    lines.append(", ".join(tag_list))
    lines.append("")

    return "\n".join(lines)


def build_full_context_summary() -> str:
    """Build the complete context summary for system prompts.

    Combines hierarchy overview and topic tags into a single
    context block for the agent.
    """
    index = get_index_store()
    entries = index.all_entries()
    by_source: Dict[str, int] = defaultdict(int)
    for entry in entries:
        by_source[entry.source] += 1
    sources_lines = ["## Sources\n"]
    if by_source:
        for source, count in sorted(by_source.items()):
            sources_lines.append(f"- {source}: {count} documents")
    else:
        sources_lines.append("- No sources available")
    sources_lines.append("")

    hierarchy = build_hierarchy_summary()
    topics = build_topic_tags_summary()

    return "\n".join(sources_lines) + f"{hierarchy}\n{topics}"


def _extract_category_from_title(title: str) -> str:
    """Extract a category name from a document title."""
    # Remove common prefixes
    for prefix in ["Introduction to ", "Getting Started with ", "How to "]:
        if title.startswith(prefix):
            title = title[len(prefix):]
            break

    # Take first word or phrase before separator
    for sep in ["|", "-", ":", "–"]:
        if sep in title:
            title = title.split(sep)[0].strip()
            break

    # Limit length
    return _truncate(title, 30)


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max length, adding ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."
