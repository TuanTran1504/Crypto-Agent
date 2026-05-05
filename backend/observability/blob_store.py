from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any

from .serialization import json_bytes


class LocalBlobStore:
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def put_json(self, value: Any, *, prefix: str = "artifact") -> dict[str, Any]:
        return self.put_bytes(
            json_bytes(value),
            suffix=".json",
            prefix=prefix,
        )

    def put_text(self, value: str, *, prefix: str = "artifact") -> dict[str, Any]:
        return self.put_bytes(
            value.encode("utf-8"),
            suffix=".txt",
            prefix=prefix,
        )

    def put_bytes(
        self,
        data: bytes,
        *,
        suffix: str = ".bin",
        prefix: str = "artifact",
    ) -> dict[str, Any]:
        digest = hashlib.sha256(data).hexdigest()
        subdir = self.base_dir / digest[:2]
        subdir.mkdir(parents=True, exist_ok=True)
        filename = f"{prefix}_{digest}{suffix}"
        blob_path = subdir / filename
        with self._lock:
            if not blob_path.exists():
                blob_path.write_bytes(data)
        return {
            "blob_uri": str(blob_path),
            "sha256": digest,
            "size_bytes": len(data),
        }
