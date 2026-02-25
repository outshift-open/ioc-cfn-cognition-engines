"""API routes aligned with evidence-gathering-agent DataRepository contract."""
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

from .schemas import (
    GraphPathsRequest,
    GraphPathsResponse,
    TkfPath,
    TkfPathEdge,
    NeighborsResponse,
    ConceptsByIdsRequest,
    ConceptsByIdsResponse,
    Concept,
    SemanticSimilarRequest,
)


def get_store():
    from ..main import get_store as _get
    return _get()


def get_semantic():
    from ..main import get_semantic as _get
    return _get()


router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "healthy"}


@router.post("/graph/paths", response_model=GraphPathsResponse)
async def graph_paths(req: GraphPathsRequest, store=Depends(get_store)):
    if store is None:
        raise HTTPException(status_code=503, detail="Graph store not available")
    result = await store.find_paths(
        source_id=req.source_id,
        target_id=req.target_id,
        max_depth=req.max_depth,
        limit=req.limit,
        relations=req.relations,
    )
    status = result.get("status", "success")
    paths_raw = result.get("paths", [])
    paths = []
    for p in paths_raw:
        edges = [
            TkfPathEdge(
                from_id=e.get("from_id", ""),
                to_id=e.get("to_id", ""),
                relation=e.get("relation", "RELATED_TO"),
                from_name=e.get("from_name"),
                to_name=e.get("to_name"),
            )
            for e in (p.get("edges") or [])
        ]
        paths.append(TkfPath(
            node_ids=p.get("node_ids"),
            edges=edges,
            path_length=p.get("path_length"),
            symbolic=p.get("symbolic", ""),
        ))
    return GraphPathsResponse(status=status, paths=paths)


@router.get("/graph/neighbors/by_name", response_model=NeighborsResponse)
async def graph_neighbors_by_name(name: str, store=Depends(get_store)):
    """Get concept and its one-hop neighbours by concept name only (no ID required)."""
    if store is None:
        raise HTTPException(status_code=503, detail="Graph store not available")
    rows = await store.get_concepts_by_name(name)
    if not rows:
        return NeighborsResponse(records=[])
    concept_id = rows[0].get("id")
    if not concept_id:
        return NeighborsResponse(records=[])
    result = await store.neighbors(concept_id)
    records = result.get("records", [])
    return NeighborsResponse(records=records)


@router.get("/graph/neighbors/{concept_id}", response_model=NeighborsResponse)
async def graph_neighbors(concept_id: str, store=Depends(get_store)):
    if store is None:
        raise HTTPException(status_code=503, detail="Graph store not available")
    # If the client requested /neighbors/by_name but this route was matched (wrong order), reject so client doesn't get empty records
    if concept_id == "by_name":
        raise HTTPException(
            status_code=404,
            detail="Route /graph/neighbors/by_name was matched as concept_id. Restart tkf-data-layer so /graph/neighbors/by_name is registered first.",
        )
    result = await store.neighbors(concept_id)
    return NeighborsResponse(records=result.get("records", []))


@router.get("/graph/concepts/by_name", response_model=ConceptsByIdsResponse)
async def graph_concepts_by_name(name: str, store=Depends(get_store)):
    """Get concepts by exact name (e.g. ?name=multi_agent_system)."""
    if store is None:
        raise HTTPException(status_code=503, detail="Graph store not available")
    rows = await store.get_concepts_by_name(name)
    concepts = [
        Concept(
            id=r.get("id", ""),
            name=r.get("name", ""),
            type=r.get("type", "concept"),
            description=r.get("description", ""),
        )
        for r in rows
    ]
    return ConceptsByIdsResponse(concepts=concepts)


@router.post("/graph/concepts/by_ids", response_model=ConceptsByIdsResponse)
async def graph_concepts_by_ids(req: ConceptsByIdsRequest, store=Depends(get_store)):
    if store is None:
        raise HTTPException(status_code=503, detail="Graph store not available")
    rows = await store.get_concepts_by_ids(req.ids)
    concepts = [
        Concept(
            id=r.get("id", ""),
            name=r.get("name", ""),
            type=r.get("type", "concept"),
            description=r.get("description", ""),
        )
        for r in rows
    ]
    return ConceptsByIdsResponse(concepts=concepts)


@router.post("/semantic/similar")
async def semantic_similar(req: SemanticSimilarRequest, semantic=Depends(get_semantic)):
    """Search by query vector; returns list of { distance, concept, relations, neighbor_concepts }."""
    if semantic is None:
        raise HTTPException(status_code=503, detail="Semantic search not available")
    items = await semantic.search_similar_with_neighbors(
        query_vec=req.query_vector,
        k=req.k,
    )
    return {"results": items}


# ---- Admin ----

@router.post("/admin/load-kcr")
async def admin_load_kcr(payload: Dict[str, Any], store=Depends(get_store)):
    """Load KCR JSON (concepts + relations) into Neo4j. Body: { concepts: [...], relations: [...] }."""
    if store is None:
        raise HTTPException(status_code=503, detail="Graph store not available")
    counts = await store.load_kcr(payload)
    return {"status": "ok", "loaded": counts}


@router.post("/admin/clear")
async def admin_clear(store=Depends(get_store)):
    """Clear all concepts and relations from Neo4j."""
    if store is None:
        raise HTTPException(status_code=503, detail="Graph store not available")
    await store.clear()
    return {"status": "ok", "message": "Graph cleared"}


@router.post("/admin/generate-embeddings")
async def admin_generate_embeddings(store=Depends(get_store), semantic=Depends(get_semantic)):
    """Generate embeddings for all concepts that don't have one."""
    if store is None or semantic is None:
        raise HTTPException(status_code=503, detail="Store or semantic not available")
    driver = store._driver
    if not driver:
        raise HTTPException(status_code=503, detail="Neo4j not connected")
    indexed = 0
    errors = []
    # Don't use WHERE c.embedding IS NULL (triggers Neo4j warning when property doesn't exist yet)
    async with driver.session() as session:
        result = await session.run(
            "MATCH (c:Concept) RETURN c.id AS id, c.name AS name, c.description AS description, c.embedding AS embedding"
        )
        async for record in result:
            if record.get("embedding") is not None:
                continue  # already has embedding
            cid = record["id"]
            name = record["name"] or ""
            desc = record["description"] or ""
            try:
                ok = await semantic.index_concept(cid, name, desc)
                if ok:
                    indexed += 1
            except Exception as e:
                errors.append(f"{cid}: {e!s}")
    out = {"status": "ok", "indexed": indexed}
    if errors:
        out["errors"] = errors[:10]
    return out
