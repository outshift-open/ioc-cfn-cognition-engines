"""
HTTP DataRepository: calls mocked-db (or any compatible) server.
Set DATA_LAYER_BASE_URL or MOCKED_DB_BASE_URL (e.g. http://localhost:8088) to use this instead of MockDataRepository.
"""
from typing import Any, Dict, List, Optional

import httpx

from .base import DataRepository


class HttpDataRepository:
    """DataRepository implementation that calls mocked-db HTTP API."""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self.graph_base_url = f"{self._base}/api/v1/graph"

    def _client_sync(self) -> httpx.Client:
        return httpx.Client(base_url=self._base, timeout=self._timeout)

    async def _client_async(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._base, timeout=self._timeout)
        return self._client

    async def fetch_records(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []

    async def neighbors(self, concept_id: str) -> Dict[str, Any]:
        client = await self._client_async()
        r = await client.get(f"/api/v1/graph/neighbors/{concept_id}")
        r.raise_for_status()
        return r.json()

    async def neighbors_by_name(self, name: str) -> Dict[str, Any]:
        """Get concept and its one-hop neighbours by concept name only (no ID required)."""
        if not (name or str(name).strip()):
            return {"records": []}
        client = await self._client_async()
        r = await client.get("/api/v1/graph/neighbors/by_name", params={"name": str(name).strip()})
        r.raise_for_status()
        return r.json()

    async def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int,
        limit: int,
        relations: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        client = await self._client_async()
        r = await client.post(
            "/api/v1/graph/paths",
            json={
                "source_id": source_id,
                "target_id": target_id,
                "max_depth": max_depth,
                "limit": limit,
                "relations": relations,
            },
        )
        r.raise_for_status()
        return r.json()

    async def get_concepts_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        if not ids:
            return []
        client = await self._client_async()
        r = await client.post("/api/v1/graph/concepts/by_ids", json={"ids": ids})
        r.raise_for_status()
        data = r.json()
        return data.get("concepts", [])

    async def get_concepts_by_name(self, name: str) -> List[Dict[str, Any]]:
        """Get concepts from the graph by exact name."""
        if not (name or str(name).strip()):
            return []
        client = await self._client_async()
        r = await client.get("/api/v1/graph/concepts/by_name", params={"name": str(name).strip()})
        r.raise_for_status()
        data = r.json()
        return data.get("concepts", [])

    def search_similar_with_neighbors(self, query_vec: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Synchronous call used by multi_entity via asyncio.to_thread."""
        raw = query_vec or []
        flat: List[float] = []
        for x in raw:
            if isinstance(x, (list, tuple)):
                flat.extend(float(v) for v in x)
            else:
                flat.append(float(x))
        with self._client_sync() as client:
            r = client.post(
                "/api/v1/semantic/similar",
                json={"query_vector": flat, "k": k},
            )
            if r.status_code in (422, 503, 500):
                return []
            r.raise_for_status()
            data = r.json()
            return data.get("results", [])

    def post_to_data_logic_svc(self, url: str, payload: Dict[str, Any]) -> Any:
        """Used by multi_entity for pathfinding."""
        with self._client_sync() as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            return type("Response", (), {"status_code": r.status_code, "json": lambda self=None: data})()
