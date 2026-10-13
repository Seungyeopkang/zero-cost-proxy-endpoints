"""config-hash 붙은 jsonl 로깅 (연구 프로토콜 기록 규칙)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def config_hash(config: dict[str, Any], length: int = 12) -> str:
    payload = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:length]


def append_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            n += 1
    return n


def run_header(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "_type": "header",
        "config_hash": config_hash(config),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
    }
