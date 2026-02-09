from __future__ import annotations

from urllib.parse import urlencode, urlparse, urlunparse

from profectus_ai.models import Provenance


def build_deep_link(provenance: Provenance) -> str:
    url = provenance.url
    parsed = urlparse(url)
    query = dict()
    if parsed.query:
        for pair in parsed.query.split("&"):
            if "=" in pair:
                key, value = pair.split("=", 1)
                query[key] = value
    fragment = parsed.fragment

    if provenance.timestamp_start is not None:
        query["t"] = str(int(provenance.timestamp_start))

    if provenance.anchor:
        fragment = provenance.anchor

    rebuilt = urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urlencode(query, doseq=True),
            fragment,
            "",
        )
    )
    return rebuilt
