from __future__ import annotations

import json
import logging
from typing import Any, Dict


def log_event(message: str, **fields: Any) -> None:
    payload: Dict[str, Any] = {"message": message, **fields}
    logging.getLogger("profectus").info(json.dumps(payload, ensure_ascii=True))
