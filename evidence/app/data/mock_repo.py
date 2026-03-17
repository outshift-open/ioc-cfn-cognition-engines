from typing import Dict, Any, List, Optional
from .base import DataRepository


class MockDataRepository:
    async def fetch_records(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [{"id": 1, "value": 42}, {"id": 2, "value": 99}]

    async def neighbors(self, concept_id: str) -> Dict[str, Any]:
        return {
            "records": [
                {
                    "node": {"id": concept_id, "name": concept_id},
                    "relationships": [],
                    "neighbors": [],
                }
            ]
        }

    async def find_paths(
        self, source_id: str, target_id: str, max_depth: int, limit: int, relations: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        return {
            "status": "success",
            "paths": [
                {
                    "node_ids": [source_id, target_id],
                    "edges": [
                        {"from_id": source_id, "to_id": target_id, "relation": "RELATED_TO"},
                    ],
                    "path_length": 1,
                    "symbolic": f"{source_id} -RELATED_TO-> {target_id}",
                }
            ],
        }

    async def get_concepts_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        return [{"id": i, "name": i, "description": "", "type": "concept"} for i in ids or []]

    async def get_concepts_by_name(self, name: str) -> List[Dict[str, Any]]:
        """Search by concept name; returns list of matching concepts (mock: single match with id=name)."""
        if not (name or str(name).strip()):
            return []
        n = str(name).strip()
        return [{"id": n, "name": n, "description": "", "type": "concept"}]

    async def neighbors_by_name(self, name: str) -> Dict[str, Any]:
        """Get concept and its one-hop neighbours by concept name only (mock: resolve by name then neighbors(id))."""
        if not (name or str(name).strip()):
            return {"records": []}
        concepts = await self.get_concepts_by_name(name)
        if not concepts:
            return {"records": []}
        cid = (concepts[0] or {}).get("id")
        if not cid:
            return {"records": []}
        return await self.neighbors(cid)

    # ---- Additional methods/attrs to match algorithm expectations ----

    # Optional base URL attribute used for building paths calls (matches HttpDataRepository.graph_base_url usage)
    graph_base_url: str = "http://mock-data-logic"

    # Synchronous method used via asyncio.to_thread(...) for top-k similar with neighbors
    def search_similar_with_neighbors(self, query_vec, k: int = 5) -> List[Dict[str, Any]]:
        # Return empty enriched results but with the correct shape
        # [
        #   {
        #     "distance": float,
        #     "concept": { "id": str, "name": str, ... },
        #     "relations": [ { "id": ..., "node_ids": [...], "relationship": "...", "attributes": {...} } ],
        #     "neighbor_concepts": [ { "id": str, "name": str, ... } ]
        #   }
        # ]
        return []

    class _MockResponse:
        def __init__(self, data: Dict[str, Any], status_code: int = 200):
            self._data = data
            self.status_code = status_code

        def json(self) -> Dict[str, Any]:
            return self._data

    # Requests-like POST used by multi-entity _paths_call(...); returns success with empty paths
    def post_to_data_logic_svc(self, url: str, payload: Dict[str, Any]) -> "MockDataRepository._MockResponse":
        return self._MockResponse({"status": "success", "paths": []}, status_code=200)

