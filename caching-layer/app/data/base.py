"""Data access abstractions for cache storage backends."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol


class CacheStore(Protocol):
    """Protocol for cache persistence layers."""

    def put(self, key: str, value: Any, ttl_seconds: int) -> None:
        ...

    def get(self, key: str) -> Optional[Any]:
        ...

    def purge(self, key: str) -> None:
        ...


@dataclass
class InMemoryCacheStore:
    """Very small in-memory cache used by tests and early prototypes."""

    _store: Dict[str, Any] = field(default_factory=dict)

    def put(self, key: str, value: Any, ttl_seconds: int) -> None:  # pragma: no cover - placeholder
        self._store[key] = {"value": value, "ttl_seconds": ttl_seconds}

    def get(self, key: str) -> Optional[Any]:  # pragma: no cover - placeholder
        record = self._store.get(key)
        if not record:
            return None
        return record["value"]

    def purge(self, key: str) -> None:  # pragma: no cover - placeholder
        self._store.pop(key, None)
