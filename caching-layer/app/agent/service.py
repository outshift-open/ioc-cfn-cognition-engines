"""Domain service that coordinates cache operations."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class CacheMetrics:
    """Lightweight metrics object for cache orchestration."""

    primed_requests: int = 0
    last_prime_timestamp: Optional[datetime] = None
    tracked_keys: List[str] = field(default_factory=list)
    recent_errors: List[str] = field(default_factory=list)


class CacheOrchestratorService:
    """Stub service that simulates cache coordination work."""

    def __init__(self, namespace: str, default_ttl_seconds: int) -> None:
        self.namespace = namespace
        self.default_ttl_seconds = default_ttl_seconds
        self.metrics = CacheMetrics()
        self._initialized = True

    def get_cache_status(self) -> Dict[str, Any]:
        """Return the most recent cache stats without touching any backend."""
        return {
            "namespace": self.namespace,
            "default_ttl_seconds": self.default_ttl_seconds,
            "primed_requests": self.metrics.primed_requests,
            "tracked_keys": list(self.metrics.tracked_keys),
            "last_prime_timestamp": self._format_timestamp(self.metrics.last_prime_timestamp),
        }

    def prime_cache(self, keys: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Simulate cache priming by recording the request."""
        selected_keys = keys or [f"{self.namespace}:synthetic-key"]
        payload_metadata = metadata or {}
        timestamp = datetime.utcnow()

        self.metrics.primed_requests += 1
        self.metrics.last_prime_timestamp = timestamp
        self.metrics.tracked_keys = selected_keys[-5:]

        return {
            "primed_keys": selected_keys,
            "applied_namespace": self.namespace,
            "ttl_seconds": payload_metadata.get("ttl_seconds", self.default_ttl_seconds),
            "metadata": payload_metadata,
            "diagnostics": {
                "request_number": self.metrics.primed_requests,
                "processed_at": self._format_timestamp(timestamp),
            },
        }

    def report_health_and_diagnostics(self) -> Dict[str, Any]:
        """Expose a health payload consistent with other services."""
        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "initialized": self._initialized,
            "metrics": self.get_operational_metrics(),
            "recent_errors": list(self.metrics.recent_errors[-5:]),
        }

    def get_operational_metrics(self) -> Dict[str, Any]:
        """Provide metrics in plain dict form for FastAPI responses."""
        return {
            "primed_requests": self.metrics.primed_requests,
            "last_prime_timestamp": self._format_timestamp(self.metrics.last_prime_timestamp),
            "tracked_keys": list(self.metrics.tracked_keys),
        }

    @staticmethod
    def _format_timestamp(value: Optional[datetime]) -> Optional[str]:
        if not value:
            return None
        return value.replace(microsecond=0).isoformat(timespec="seconds") + "Z"
