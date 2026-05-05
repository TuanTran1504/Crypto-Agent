from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from .serialization import json_dumps, summarize_value, to_serializable


ACTION_ALLOW = "allow"
ACTION_DROP = "drop"
ACTION_HASH = "hash"
ACTION_MASK = "mask"
ACTION_SUMMARY_ONLY = "summary_only"

ALLOWED_ACTIONS = {
    ACTION_ALLOW,
    ACTION_DROP,
    ACTION_HASH,
    ACTION_MASK,
    ACTION_SUMMARY_ONLY,
}


@dataclass(frozen=True)
class RedactionRule:
    pattern: str
    action: str

    def __post_init__(self):
        if self.action not in ALLOWED_ACTIONS:
            raise ValueError(f"Unsupported redaction action: {self.action}")


class TraceRedactor:
    def __init__(self, rules: list[RedactionRule] | None = None):
        self._rules = list(rules or [])

    def redact(self, value: Any, path: tuple[str, ...] = ()) -> Any:
        item = to_serializable(value)
        return self._redact_value(item, tuple(str(part) for part in path))

    def _redact_value(self, value: Any, path: tuple[str, ...]) -> Any:
        rule = self._best_rule(path)
        if rule and rule.action != ACTION_ALLOW:
            return self._apply(rule.action, value)

        if isinstance(value, dict):
            return {
                str(key): self._redact_value(item, path + (str(key),))
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                self._redact_value(item, path + (str(index),))
                for index, item in enumerate(value)
            ]
        return value

    def _best_rule(self, path: tuple[str, ...]) -> RedactionRule | None:
        matches: list[tuple[int, RedactionRule]] = []
        for rule in self._rules:
            pattern = tuple(part for part in rule.pattern.split(".") if part)
            if _match_pattern(pattern, path):
                matches.append((len(pattern), rule))
        if not matches:
            return None
        matches.sort(key=lambda item: item[0], reverse=True)
        return matches[0][1]

    @staticmethod
    def _apply(action: str, value: Any) -> Any:
        if action == ACTION_DROP:
            return {
                "_trace_redacted": True,
                "mode": ACTION_DROP,
            }
        if action == ACTION_MASK:
            if isinstance(value, str):
                return "[REDACTED]"
            return {
                "_trace_redacted": True,
                "mode": ACTION_MASK,
                "type": type(value).__name__,
            }
        if action == ACTION_HASH:
            digest = hashlib.sha256(json_dumps(value).encode("utf-8")).hexdigest()
            return {
                "_trace_redacted": True,
                "mode": ACTION_HASH,
                "sha256": digest,
                "summary": summarize_value(value),
            }
        if action == ACTION_SUMMARY_ONLY:
            return {
                "_trace_redacted": True,
                "mode": ACTION_SUMMARY_ONLY,
                "summary": summarize_value(value),
            }
        return value


def _match_pattern(pattern: tuple[str, ...], path: tuple[str, ...]) -> bool:
    if not pattern:
        return not path
    if pattern[0] == "**":
        if len(pattern) == 1:
            return True
        return any(_match_pattern(pattern[1:], path[index:]) for index in range(len(path) + 1))
    if not path:
        return False
    token = pattern[0]
    if token != "*" and token != path[0]:
        return False
    return _match_pattern(pattern[1:], path[1:])
