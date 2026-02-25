"""
Neo4j graph store for KCR data.
Nodes: Concept with id (KCR concept id), name, description, type, domain, attributes (JSON).
Edges: RELATES with relationship_type (from KCR relation.relationship), optional weight/attributes.
"""
import json
from typing import Any, Dict, List, Optional

try:
    from neo4j import AsyncGraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


class Neo4jStore:
    """Neo4j-backed graph store matching evidence-gathering-agent DataRepository contract."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        if not NEO4J_AVAILABLE:
            raise ImportError("neo4j driver not available. Install with: pip install neo4j")
        self._uri = uri
        self._user = user
        self._password = password
        self._driver = None

    async def connect(self) -> None:
        self._driver = AsyncGraphDatabase.driver(
            self._uri,
            auth=(self._user, self._password),
        )
        async with self._driver.session() as session:
            await session.run("RETURN 1")

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()
            self._driver = None

    # ---- KCR loader ----

    async def load_kcr(self, data: Dict[str, Any]) -> Dict[str, int]:
        """
        Load KCR JSON (concepts + relations) into Neo4j.
        data: { "concepts": [...], "relations": [...] }
        Returns counts: { "concepts": N, "relations": M }.
        """
        concepts = data.get("concepts") or []
        relations = data.get("relations") or []
        created_nodes = 0
        created_rels = 0

        async with self._driver.session() as session:
            for c in concepts:
                cid = c.get("id")
                if not cid:
                    continue
                name = c.get("name") or ""
                description = c.get("description") or ""
                ctype = c.get("type") or "concept"
                attrs = c.get("attributes") or {}
                domain = (attrs.get("domain") or "").strip() or "default"
                attrs_json = json.dumps(attrs)

                await session.run(
                    """
                    MERGE (c:Concept {id: $id})
                    SET c.name = $name, c.description = $description, c.type = $type,
                        c.domain = $domain, c.attributes = $attributes
                    """,
                    id=cid,
                    name=name,
                    description=description,
                    type=ctype,
                    domain=domain,
                    attributes=attrs_json,
                )
                created_nodes += 1

            for r in relations:
                nids = r.get("node_ids") or []
                if len(nids) < 2:
                    continue
                rel_type = (r.get("relationship") or "RELATED_TO").strip() or "RELATED_TO"
                rid = r.get("id") or ""
                attrs = r.get("attributes") or {}
                attrs_json = json.dumps(attrs)

                await session.run(
                    """
                    MATCH (a:Concept {id: $from_id})
                    MATCH (b:Concept {id: $to_id})
                    MERGE (a)-[r:RELATES]->(b)
                    SET r.relationship_type = $rel_type, r.relation_id = $rid, r.attributes = $attributes
                    """,
                    from_id=nids[0],
                    to_id=nids[1],
                    rel_type=rel_type,
                    rid=rid,
                    attributes=attrs_json,
                )
                created_rels += 1

        return {"concepts": created_nodes, "relations": created_rels}

    async def clear(self) -> None:
        async with self._driver.session() as session:
            await session.run("MATCH ()-[r:RELATES]->() DELETE r")
            await session.run("MATCH (c:Concept) DELETE c")

    # ---- Evidence-agent contract ----

    async def get_concepts_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """Return list of { id, name, type, description } for each concept id."""
        if not ids:
            return []
        out: List[Dict[str, Any]] = []
        async with self._driver.session() as session:
            result = await session.run(
                "MATCH (c:Concept) WHERE c.id IN $ids RETURN c.id AS id, c.name AS name, c.type AS type, c.description AS description",
                ids=ids,
            )
            async for record in result:
                out.append({
                    "id": record["id"],
                    "name": record["name"] or "",
                    "type": record["type"] or "concept",
                    "description": record["description"] or "",
                })
        return out

    async def get_concepts_by_name(self, name: str) -> List[Dict[str, Any]]:
        """Return list of concepts whose name equals the given string (case-sensitive)."""
        out: List[Dict[str, Any]] = []
        async with self._driver.session() as session:
            result = await session.run(
                "MATCH (c:Concept) WHERE c.name = $name RETURN c.id AS id, c.name AS name, c.type AS type, c.description AS description",
                name=name,
            )
            async for record in result:
                out.append({
                    "id": record["id"],
                    "name": record["name"] or "",
                    "type": record["type"] or "concept",
                    "description": record["description"] or "",
                })
        return out

    async def neighbors(self, concept_id: str) -> Dict[str, Any]:
        """
        Return { records: [ { node, relationships, neighbors } ] }.
        One record: the concept, its outgoing/incoming relations (with node_ids, relationship), and neighbor concepts.
        """
        records: List[Dict[str, Any]] = []
        async with self._driver.session() as session:
            # Get node; then optional edges so we return one row even when no relationships
            result = await session.run(
                """
                MATCH (c:Concept {id: $id})
                OPTIONAL MATCH (c)-[r:RELATES]-(other:Concept)
                WITH c, r, other
                WITH c,
                     collect(DISTINCT CASE WHEN r IS NOT NULL AND other IS NOT NULL THEN {
                         id: r.relation_id,
                         node_ids: CASE WHEN startNode(r) = c THEN [c.id, other.id] ELSE [other.id, c.id] END,
                         relationship: r.relationship_type,
                         attributes: r.attributes
                     } END) AS rels_raw,
                     collect(DISTINCT CASE WHEN other IS NOT NULL THEN { id: other.id, name: other.name, description: other.description, type: other.type } END) AS neighbors_raw
                RETURN c, rels_raw, neighbors_raw
                """,
                id=concept_id,
            )
            async for record in result:
                c = record["c"]
                if not c:
                    continue
                rels_raw = record["rels_raw"] or []
                rels = []
                for r in rels_raw:
                    if r and r.get("relationship"):
                        attrs = {}
                        if r.get("attributes"):
                            try:
                                attrs = json.loads(r["attributes"]) if isinstance(r["attributes"], str) else (r["attributes"] or {})
                            except Exception:
                                pass
                        rels.append({
                            "id": r.get("id"),
                            "node_ids": r.get("node_ids") or [],
                            "relationship": r.get("relationship"),
                            "attributes": attrs,
                        })
                neighbors_raw = record["neighbors_raw"] or []
                neighbors = []
                seen_ids: set = set()
                for n in neighbors_raw:
                    if n and n.get("id") and n["id"] not in seen_ids:
                        seen_ids.add(n["id"])
                        neighbors.append({
                            "id": n["id"],
                            "name": n.get("name") or "",
                            "description": n.get("description") or "",
                            "type": n.get("type") or "concept",
                        })
                node = {
                    "id": c["id"],
                    "name": c.get("name") or "",
                    "description": c.get("description") or "",
                    "type": c.get("type") or "concept",
                }
                records.append({"node": node, "relationships": rels, "neighbors": neighbors})
        return {"records": records}

    async def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 3,
        limit: int = 10,
        relations: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Return { status, paths } where each path has node_ids, edges (from_id, to_id, relation, from_name, to_name), path_length, symbolic.
        """
        paths: List[Dict[str, Any]] = []
        depth = min(max(1, max_depth), 5)
        try:
            async with self._driver.session() as session:
                params: Dict[str, Any] = {
                    "source_id": source_id,
                    "target_id": target_id,
                    "limit": limit,
                }
                rel_filter = ""
                if relations:
                    rel_filter = " AND all(rel in relationships(path) WHERE rel.relationship_type IN $rel_types)"
                    params["rel_types"] = relations

                query = f"""
                MATCH path = (a:Concept {{id: $source_id}})-[r:RELATES*1..{depth}]-(b:Concept {{id: $target_id}})
                WHERE a <> b {rel_filter}
                WITH path, relationships(path) AS rels, nodes(path) AS nodes
                WITH path, rels, nodes,
                     [n IN nodes | n.id] AS node_ids,
                     [rel IN rels | rel.relationship_type] AS rel_types
                RETURN node_ids, rel_types, length(path) AS path_length
                ORDER BY path_length ASC
                LIMIT $limit
                """
                result = await session.run(query, params)
                async for record in result:
                    node_ids = record["node_ids"] or []
                    rel_types = record["rel_types"] or []
                    path_length = record["path_length"] or 0
                    edges = []
                    for i, rel_type in enumerate(rel_types):
                        from_id = node_ids[i] if i < len(node_ids) else ""
                        to_id = node_ids[i + 1] if i + 1 < len(node_ids) else ""
                        edges.append({
                            "from_id": from_id,
                            "to_id": to_id,
                            "relation": rel_type,
                            "from_name": None,
                            "to_name": None,
                        })
                    # Build symbolic: "id1 -REL-> id2 ; id2 -REL2-> id3"
                    parts = []
                    for e in edges:
                        parts.append(f"{e['from_id']} -{e['relation']}-> {e['to_id']}")
                    symbolic = " ; ".join(parts) if parts else ""
                    paths.append({
                        "node_ids": node_ids,
                        "edges": edges,
                        "path_length": path_length,
                        "symbolic": symbolic,
                    })
        except Exception:
            return {"status": "error", "paths": []}
        return {"status": "success", "paths": paths}
