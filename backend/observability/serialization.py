from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


UTC = timezone.utc


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return to_serializable(asdict(value))
    if isinstance(value, datetime):
        return _to_utc(value).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseException):
        return {
            "type": value.__class__.__name__,
            "message": str(value),
        }
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_serializable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def json_dumps(value: Any) -> str:
    return json.dumps(
        to_serializable(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def json_bytes(value: Any) -> bytes:
    return json_dumps(value).encode("utf-8")


def summarize_value(value: Any, *, max_items: int = 5) -> dict[str, Any]:
    item = to_serializable(value)
    if isinstance(item, dict):
        keys = sorted(item.keys())
        return {
            "type": "object",
            "size": len(item),
            "keys": keys[:max_items],
        }
    if isinstance(item, list):
        preview_types = [type(entry).__name__ for entry in item[:max_items]]
        return {
            "type": "array",
            "size": len(item),
            "preview_types": preview_types,
        }
    if isinstance(item, str):
        return {
            "type": "string",
            "chars": len(item),
        }
    if isinstance(item, bool):
        return {"type": "bool", "value": item}
    if isinstance(item, int):
        return {"type": "int", "value": item}
    if isinstance(item, float):
        return {"type": "float", "value": item}
    if item is None:
        return {"type": "null"}
    return {
        "type": type(item).__name__,
        "value": str(item),
    }
