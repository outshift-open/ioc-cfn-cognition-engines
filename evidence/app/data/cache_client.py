"""
Cache layer client for similarity search when no vector DB is hooked.
Calls POST /api/v1/cache/search with query_vector and k; returns list of { id, score, text }.
Convention: cache stores concept id in 'text' when primed from the graph, for graph lookup.
"""
from typing import Any, Dict, List

import httpx


class CacheClient:
    """Sync client for caching-layer search. Use from thread (e.g. asyncio.to_thread)."""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    def search(self, query_vec: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Return list of { id, score, text } from cache similarity search by vector."""
        raw = query_vec or []
        flat: List[float] = []
        for x in raw:
            if isinstance(x, (list, tuple)):
                flat.extend(float(v) for v in x)
            else:
                flat.append(float(x))
        with httpx.Client(base_url=self._base, timeout=self._timeout) as client:
            r = client.post(
                "/api/v1/cache/search",
                json={"vector": flat, "k": k},
            )
            if r.status_code in (422, 500, 503):
                return []
            r.raise_for_status()
            data = r.json()
        return data.get("results", [])

    def search_by_text(self, text: str, k: int = 5) -> List[Dict[str, Any]]:
        """Return list of { id, score, text } from cache similarity search by entity string. Cache embeds text server-side."""
        with httpx.Client(base_url=self._base, timeout=self._timeout) as client:
            r = client.post(
                "/api/v1/cache/search",
                json={"text": (text or "").strip(), "k": k},
            )
            if r.status_code in (422, 500, 503):
                return []
            r.raise_for_status()
            data = r.json()
        return data.get("results", [])
