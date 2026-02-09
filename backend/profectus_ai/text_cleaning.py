from __future__ import annotations

import re

_TRANSCRIPT_REPLACEMENTS = (
    (r"\bno\s+ode\b", "no-code"),
    (r"\bth\s+rou\s+gh\b", "through"),
    (r"\bpre\s+de\s+fine'?d\b", "predefined"),
    (r"\bdown\s+si\s+de\b", "downside"),
    (r"\bsub\s+scribe\b", "subscribe"),
    (r"\bweb\s+ina\s+r\b", "webinar"),
    (r"\btr\s+ada\s+ble\b", "tradable"),
    (r"\bpro\s+fi'?t\b", "profit"),
    (r"\bdel\s+ete\b", "delete"),
)


def clean_transcript_text(text: str) -> str:
    """Normalize common OCR/ASR artifacts in YouTube transcripts."""
    cleaned = " ".join(text.split())
    for pattern, replacement in _TRANSCRIPT_REPLACEMENTS:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return cleaned
