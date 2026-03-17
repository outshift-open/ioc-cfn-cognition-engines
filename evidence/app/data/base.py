from typing import Protocol, List, Dict, Any, Optional


class DataRepository(Protocol):
    async def fetch_records(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        ...

    async def neighbors(self, concept_id: str) -> Dict[str, Any]:
        ...

    async def find_paths(
        self, source_id: str, target_id: str, max_depth: int, limit: int, relations: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        ...

    async def get_concepts_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        ...

    async def get_concepts_by_name(self, name: str) -> List[Dict[str, Any]]:
        """Return concepts from the graph that match the given name."""
        ...

    async def neighbors_by_name(self, name: str) -> Dict[str, Any]:
        """Return the concept and its one-hop neighbours from the graph, looked up by concept name only. Same shape as neighbors(concept_id)."""
        ...

