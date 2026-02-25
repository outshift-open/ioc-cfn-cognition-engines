"""
Semantic search: store concept embeddings in Neo4j and search by vector.
Evidence-agent expects search_similar_with_neighbors(query_vec, k) -> list of
{ distance, concept, relations, neighbor_concepts }.
"""
import asyncio
import json
from typing import Any, Dict, List, Optional

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


class SemanticSearch:
    """Generate and search concept embeddings. Uses Neo4j store for persistence."""

    def __init__(
        self,
        store: Any,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
    ) -> None:
        self._store = store
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if not ST_AVAILABLE:
            raise RuntimeError("sentence_transformers not available. pip install sentence-transformers")
        self._model = SentenceTransformer(self._model_name)

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text (async wrapper)."""
        loop = asyncio.get_event_loop()
        def _encode():
            self._ensure_model()
            emb = self._model.encode([text], convert_to_numpy=True)
            return emb[0].tolist()
        return await loop.run_in_executor(None, _encode)

    async def index_concept(self, concept_id: str, name: str, description: str) -> bool:
        """Generate embedding from name+description and store on Concept node."""
        text = f"{name} {description}".strip() or concept_id
        emb = await self.generate_embedding(text)
        driver = self._store._driver
        if not driver:
            return False
        async with driver.session() as session:
            await session.run(
                "MATCH (c:Concept {id: $id}) SET c.embedding = $embedding",
                id=concept_id,
                embedding=emb,
            )
        return True

    async def search_similar_with_neighbors(
        self,
        query_vec: List[float],
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Return list of { distance, concept, relations, neighbor_concepts }.
        Fetches all concepts with embeddings, computes cosine similarity, returns top k with neighbors.
        """
        import math
        driver = self._store._driver
        if not driver:
            return []
        # Fetch all concepts; filter to those with embedding in Python (avoids Neo4j warning when property missing)
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (c:Concept)
                RETURN c.id AS id, c.name AS name, c.description AS description, c.type AS type, c.embedding AS embedding
                """
            )
            rows = []
            async for record in result:
                if record.get("embedding") is None:
                    continue
                rows.append({
                    "id": record["id"],
                    "name": record["name"] or "",
                    "description": record["description"] or "",
                    "type": record["type"] or "concept",
                    "embedding": record["embedding"],
                })
        if not rows or not query_vec:
            return []
        # Cosine similarity: score = dot(a,b) / (norm(a)*norm(b)); we use distance = 1 - similarity so lower is better
        q = query_vec
        q_norm = math.sqrt(sum(x * x for x in q))
        if q_norm <= 0:
            return []
        scored = []
        for r in rows:
            emb = r.get("embedding") or []
            if len(emb) != len(q):
                continue
            dot = sum(a * b for a, b in zip(q, emb))
            e_norm = math.sqrt(sum(x * x for x in emb))
            if e_norm <= 0:
                continue
            similarity = dot / (q_norm * e_norm)
            distance = 1.0 - similarity
            scored.append((distance, r))
        scored.sort(key=lambda x: x[0])
        top = scored[:k]
        # For each top concept, get neighbors and relations from store
        out = []
        for distance, r in top:
            concept_id = r["id"]
            neighbors_result = await self._store.neighbors(concept_id)
            records = neighbors_result.get("records") or []
            relations = []
            neighbor_concepts = []
            for rec in records:
                for rel in rec.get("relationships") or []:
                    if rel.get("relationship"):
                        relations.append({
                            "id": rel.get("id"),
                            "node_ids": rel.get("node_ids") or [],
                            "relationship": rel.get("relationship"),
                            "attributes": rel.get("attributes"),
                        })
                for n in rec.get("neighbors") or []:
                    neighbor_concepts.append(n)
            concept = {
                "id": r["id"],
                "name": r["name"],
                "description": r["description"],
                "type": r["type"],
            }
            out.append({
                "distance": distance,
                "concept": concept,
                "relations": relations,
                "neighbor_concepts": neighbor_concepts,
            })
        return out
