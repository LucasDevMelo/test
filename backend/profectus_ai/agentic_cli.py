from __future__ import annotations

import json
from pathlib import Path

from profectus_ai.agents.runner import AgenticRunner


def _write_output(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the agentic retrieval runner.")
    parser.add_argument("query", help="User query text")
    parser.add_argument("--out", default="data/agentic_output.json")
    args = parser.parse_args()

    runner = AgenticRunner()
    result = runner.run(args.query)
    _write_output(result.model_dump(), Path(args.out))
    print(f"[agentic] wrote {args.out}")
