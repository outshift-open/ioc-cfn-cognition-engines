from typing import List, Dict, Any, Optional, Tuple, Set
import asyncio
from dataclasses import dataclass, field

import numpy as np
from dotenv import load_dotenv, find_dotenv

from ..api.schemas import ReasonerCognitionRequest, TKFKnowledgeRecord

from .embeddings import EmbeddingManager
from .llm_clients import EvidenceJudge, EvidenceRanker, get_llm_call_count
from .utiles import (
    select_by_relative_top,
    PathFormatter,
    GraphSession,
    mmr_select_indices,
)


load_dotenv(find_dotenv())


class SingleEntityConfig:
    def __init__(
        self,
        top_k_similar: int = 2,
        select_k_per_hop: int = 3,
        max_depth: int = 3,
        llm_temperature: float = 0.1,
    ):
        self.top_k_similar = top_k_similar
        self.select_k_per_hop = select_k_per_hop
        self.max_depth = max_depth
        self.llm_temperature = llm_temperature


class ConceptRepository:
    """
    Adapter over the data repository (mock or HTTP tkf-data-layer).
    For every entity we use vector similarity in FAISS to get top-k similar concepts; the graph
    is then searched by concept name only (get_concepts_by_name, neighbors_by_name).
    - If repo has search_similar_with_neighbors (vector DB): use it with query_vec for top-k + neighbors.
    - Else if cache_client is set: FAISS search by query_vec; cache returns concept names in 'text';
      then for each name call neighbors_by_name(name) to get that concept and its one-hop neighbours (by name only).
    - Else: return empty (no vector DB, no cache).
    """

    def __init__(self, repo, cache_client=None, use_cache_for_similar: bool = False):
        self.repo = repo
        self.cache_client = cache_client
        self.use_cache_for_similar = use_cache_for_similar

    async def similar_with_neighbors_async(
        self, query_vec: List[float], k: int, entity_text: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """One entry point: return top-k similar concepts with their one-hop neighbours (from vector DB or FAISS + graph by name)."""
        use_cache_first = self.cache_client and self.use_cache_for_similar
        if not use_cache_first:
            search_fn = getattr(self.repo, "search_similar_with_neighbors", None)
            if callable(search_fn):
                return await asyncio.to_thread(search_fn, query_vec, k)
            return []
        if self.cache_client:
            # Prefer text search when entity name is available: cache embeds text server-side,
            # avoiding vector dimension mismatch with the agent's embedding model.
            if entity_text and str(entity_text).strip():
                cache_results = await asyncio.to_thread(
                    self.cache_client.search_by_text, str(entity_text).strip(), k
                )
            else:
                cache_results = await asyncio.to_thread(self.cache_client.search, query_vec, k)
            if not cache_results:
                print("[SingleEntity][Cache] Cache returned 0 results (check cache is loaded and CACHE_VECTOR_DIMENSION if using vector).")
                return []
            print(f"[SingleEntity][Cache] Cache returned {len(cache_results)} results (entity_text={entity_text!r}).")
            concept_names: List[str] = []
            score_by_name: Dict[str, float] = {}
            for r in cache_results:
                name = (r or {}).get("text")
                if name is not None:
                    name = str(name).strip()
                    if name and name not in score_by_name:
                        concept_names.append(name)
                        score_by_name[name] = float((r or {}).get("score", 0.0))
            if not concept_names:
                print("[SingleEntity][Cache] Cache results had no usable 'text' field.")
                return []
            out: List[Dict[str, Any]] = []
            for name in concept_names[:k]:
                neighbors_result = await self.repo.neighbors_by_name(name)
                records = (neighbors_result or {}).get("records") or []
                if not records:
                    print(f"[SingleEntity][Cache] Graph returned no records for concept name={name!r} (check tkf-data-layer has this concept).")
                    continue
                rec = records[0]
                concept = rec.get("node") or {}
                relations = [
                    {"id": rel.get("id"), "node_ids": list(rel.get("node_ids", [])), "relationship": rel.get("relationship"), "attributes": rel.get("attributes")}
                    for rel in rec.get("relationships") or [] if rel.get("relationship")
                ]
                neighbor_concepts = [n for n in rec.get("neighbors") or [] if n and n.get("id")]
                out.append({
                    "distance": score_by_name.get(name, 0.0),
                    "concept": {"id": concept.get("id", ""), "name": concept.get("name", ""), "description": concept.get("description", ""), "type": concept.get("type", "concept")},
                    "relations": relations,
                    "neighbor_concepts": neighbor_concepts,
                })
            return out
        return []

    async def relations_for_async(self, concept_id: str):
        # Normalize neighbors result into relation entries with node_ids/relationship
        result = await self.repo.neighbors(concept_id)
        rels: List[Dict[str, Any]] = []
        for rec in (result or {}).get("records", []) or []:
            for rel in rec.get("relationships", []) or []:
                rels.append(
                    {
                        "id": rel.get("id"),
                        "node_ids": list(rel.get("node_ids", [])),
                        "relationship": rel.get("relationship") or rel.get("relation"),
                        "attributes": rel.get("attributes"),
                    }
                )
        return rels

    async def concepts_by_ids_async(self, concept_ids: List[str]):
        return await self.repo.get_concepts_by_ids(concept_ids)


@dataclass
class LaneState:
    anchor_id: str
    anchor_name: str
    graph: GraphSession
    selected_structured: List[List[Dict[str, Any]]] = field(default_factory=list)
    selected_nl: List[str] = field(default_factory=list)
    frontier_paths: List[List[Dict[str, Any]]] = field(default_factory=list)
    seen_path_keys: Set[Tuple[Any, ...]] = field(default_factory=set)
    last_candidates_structured: Optional[List[List[Dict[str, Any]]]] = None
    last_candidates_symbolic: Optional[List[str]] = None
    last_reason: Optional[str] = None
    sufficient: bool = False


def path_key(path: List[Dict[str, Any]]) -> Tuple:
    """
    Stable identity for a path based on segment kinds and ids/relationships.
    """
    out: List[Any] = []
    for seg in path:
        if seg.get("kind") == "concept":
            out.append((seg["kind"], (seg.get("value") or {}).get("id")))
        elif seg.get("kind") == "relation":
            out.append((seg["kind"], (seg.get("value") or {}).get("relationship")))
    return tuple(out)


def last_edge(
    path: List[Dict[str, Any]],
    name_fn,
    rel_fn,
) -> Optional[Dict[str, str]]:
    """
    Return the last edge (from, relation, to) in a structured path, or None.
    """
    for i in range(len(path) - 2, -1, -1):
        if path[i].get("kind") == "relation" and i - 1 >= 0 and i + 1 < len(path):
            prev_c = path[i - 1].get("value") or {}
            next_c = path[i + 1].get("value") or {}
            return {
                "from": name_fn(prev_c),
                "relation": rel_fn(path[i].get("value") or {}),
                "to": name_fn(next_c),
            }
    return None


def _expand_paths_one_hop(paths: List[List[Dict[str, Any]]], graph: GraphSession) -> List[List[Dict[str, Any]]]:
    """
    Expand each structured path by one hop from its tail concept (simple BFS frontier expansion).
    """
    next_paths: List[List[Dict[str, Any]]] = []
    for path in paths or []:
        if not path or path[-1].get("kind") != "concept":
            continue
        tail_meta = path[-1]["value"] or {}
        tail_id = tail_meta.get("id")
        if not tail_id:
            continue
        for rel, nei in graph._neighbors_for(tail_id):
            nid = (nei or {}).get("id")
            if not nid:
                continue
            visited_ids = {seg["value"].get("id") for seg in path if seg.get("kind") == "concept"}
            if nid in visited_ids:
                continue
            extended = list(path)
            extended.append({"kind": "relation", "value": rel})
            extended.append({"kind": "concept", "value": nei})
            next_paths.append(extended)
    return next_paths


class SingleEntityEvidenceEngine:
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        repo: ConceptRepository,
        path_formatter: PathFormatter,
        judge: EvidenceJudge,
        ranker: EvidenceRanker,
        config: Optional[SingleEntityConfig] = None,
    ):
        self.embedding_manager = embedding_manager
        self.repo = repo
        self.path_formatter = path_formatter
        self.judge = judge
        self.ranker = ranker
        self.config = config or SingleEntityConfig()

    def _entity_to_query_vec(self, entity: Dict[str, Any]) -> List[float]:
        text = f"{entity.get('description') or ''}{entity.get('name') or ''}"
        chunks = self.embedding_manager.preprocess_text(text)
        vectors = self.embedding_manager.generate_embeddings(chunks)
        arr = np.array(vectors, dtype=np.float32)
        if arr.ndim == 2:
            vec = np.mean(arr, axis=0).tolist()
        else:
            vec = arr.flatten().tolist()
        return vec

    async def gather(self, request: ReasonerCognitionRequest, entity: Dict[str, Any], extra_context: Optional[str] = None) -> TKFKnowledgeRecord:
        # Snapshot LLM call count to compute per-gather delta
        llm_calls_before = get_llm_call_count()
        query_vec = self._entity_to_query_vec(entity)
        print("[SingleEntity] Starting similar concept retrieval for entity:", entity.get("name") or "(unnamed)")
        entity_name = (entity.get("name") or "").strip()
        enriched = await self.repo.similar_with_neighbors_async(
            query_vec, k=self.config.top_k_similar, entity_text=entity_name or None
        )
        print(f"[SingleEntity] Retrieved {len(enriched)} similar concepts.")

        # Initialize trace structure
        def _name_for(meta: Dict[str, Any]) -> str:
            n = (meta or {}).get("name")
            return (n.strip() if isinstance(n, str) else "") or (meta or {}).get("id") or "unknown"

        def _rel_label(rel: Dict[str, Any]) -> str:
            return str((rel or {}).get("relationship") or (rel or {}).get("relation") or "").strip() or "related_to"

        trace: Dict[str, Any] = {
            "extracted_entity": _name_for(entity),
            "tope_similar_concepts": [],
            "iterations": [],
            "lanes_count": 0,
            "sufficient": False,
            "winning": None,
            "request_decomposition": ((getattr(request, "meta", {}) or {}) or {}).get("request_decomposition"),
            "pass_on_context": (extra_context or ""),
        }
        # Populate top_similar using enriched relations and neighbor concepts
        try:
            for item in enriched or []:
                concept = (item or {}).get("concept") or {}
                anchor_id = concept.get("id")
                anchor_name = _name_for(concept)
                neighbor_list: List[Dict[str, Any]] = []
                # Map id->name from neighbor_concepts
                id_to_name: Dict[str, str] = {
                    (nc or {}).get("id"): _name_for(nc) for nc in ((item or {}).get("neighbor_concepts") or [])
                }
                # Also include anchor itself
                if anchor_id and anchor_name:
                    id_to_name[anchor_id] = anchor_name
                seen_edge: Set[Tuple[str, str, str]] = set()
                for rel in (item or {}).get("relations") or []:
                    node_ids = (rel or {}).get("node_ids") or []
                    for nid in node_ids:
                        if not anchor_id or not nid or nid == anchor_id:
                            continue
                        tup = (anchor_id, nid, _rel_label(rel))
                        if tup in seen_edge:
                            continue
                        seen_edge.add(tup)
                        neighbor_list.append({"to": id_to_name.get(nid) or nid, "relation": _rel_label(rel)})
                trace["tope_similar_concepts"].append({"anchor_concept": anchor_name, "firs_neighbour": neighbor_list})
        except Exception:
            pass
        # Build a separate GraphSession per similar concept (anchor)
        lanes: List[LaneState] = []
        for idx, item in enumerate(enriched or []):
            concept = (item or {}).get("concept") or {}
            anchor_id = concept.get("id")
            if not anchor_id:
                continue
            g = GraphSession()
            g.ingest_enriched_results([item])
            lanes.append(LaneState(anchor_id=anchor_id, anchor_name=_name_for(concept), graph=g))
        if not lanes:
            print("[SingleEntity] No anchors available; returning empty evidence.")
            content = {
                "evidence": {
                    "entity": entity,
                    "status": "insufficient",
                    "summary": {
                        "supporting_paths": 0,
                        "unique_concepts": 0,
                    },
                    "paths": [],
                    "metadata": {
                        "retrieval_mode": "single_entity",
                        "pruning_applied": True,
                        "llm_assisted": True,
                    },
                },
                "trace": trace,
            }
            try:
                trace["llm_calls"] = max(0, get_llm_call_count() - llm_calls_before)
            except Exception:
                trace["llm_calls"] = 0
            return TKFKnowledgeRecord(type="json", content=content)

        trace["lanes_count"] = len(lanes)
        winning_lane_index: Optional[int] = None

        # Hop-wise parallel exploration across lanes
        for hop in range(1, self.config.max_depth + 1):

            # First, try sufficiency selection on each lane (async per-lane), restricting candidates
            # to only the paths that start from the previously selected outer nodes (frontier).
            async def select_lane(idx: int, lane: LaneState):
                anchor_id = lane.anchor_id
                graph = lane.graph
                if hop == 1:
                    candidates_structured = graph.build_paths_from(anchor_id, hop=1)
                else:
                    frontier_paths = lane.frontier_paths or []
                    candidates_structured = _expand_paths_one_hop(frontier_paths, graph) if frontier_paths else []
                # Use compact symbolic representation for LLM selection (preserve path identity, low tokens)
                candidates_symbolic = self.path_formatter.to_symbolic_paths(candidates_structured)
                question_text = (request.payload.intent or "")
                if extra_context:
                    question_text = f"{question_text}\n\nPrior evidence:\n{extra_context}"
                chosen_idx, sufficient, reason = await self.judge.async_select_paths_and_check_sufficiency(
                    question=question_text,
                    candidate_paths=candidates_symbolic,
                    select_k=self.config.select_k_per_hop,
                )
                return idx, candidates_structured, candidates_symbolic, chosen_idx, sufficient, reason

            # Create tasks explicitly so we can cancel the rest if any lane is sufficient
            selection_tasks = [asyncio.create_task(select_lane(i, lane)) for i, lane in enumerate(lanes)]
            for fut in asyncio.as_completed(selection_tasks):
                try:
                    lane_idx, candidates_structured, candidates_symbolic, chosen_idx, sufficient, reason = await fut
                except asyncio.CancelledError:
                    continue
                lane = lanes[lane_idx]
                # Persist the candidates for ranking stage to avoid recomputation
                lane.last_candidates_structured = candidates_structured
                lane.last_candidates_symbolic = candidates_symbolic
                lane.last_reason = reason
                try:
                    _ = chosen_idx, candidates_symbolic  # for trace/debug if needed
                except Exception as e:
                    print("[SingleEntity][WARN][select-log]", e)
                # Append to trace for this hop (record the outermost edge of each selected path)
                try:
                    # Prepare judge reason only (do not record judge-selected paths)
                    reason_text = reason
                    try:
                        # keep original reason text as-is
                        _ = reason_text  # no-op to satisfy linter in try
                    except Exception:
                        pass
                    # Do not add a judge-only iteration entry; ranker entry will include the judge reason
                except Exception as e:
                    print("[SingleEntity][WARN][trace-select]", e)
                # Initialize per-lane seen set for deduplication
                if lane.seen_path_keys is None:
                    lane.seen_path_keys = set()
                # Do not persist judge selections to lane; only ranker selections populate selected_structured/NL
                if sufficient and winning_lane_index is None:
                    lane.sufficient = True
                    winning_lane_index = lane_idx
                    trace["sufficient"] = True
                    trace["winning"] = {"anchor_concept": lane.anchor_name, "reason_for_sufficiency": reason_text}
                    # Cancel remaining selection tasks
                    for t in selection_tasks:
                        if not t.done():
                            t.cancel()
                    # Drain cancellations to avoid warnings
                    for t in selection_tasks:
                        try:
                            await t
                        except asyncio.CancelledError:
                            pass
                    # Exit the as_completed loop after draining, ranker stage will run next
                    break

            # If none sufficient, rank candidates per-lane and apply relative top-25% filtering
            async def rank_and_expand_lane(idx: int, lane: LaneState):
                anchor_id = lane.anchor_id
                graph = lane.graph
                # Use the same candidate set as selection stage; if missing, compute with the same restriction
                candidates_structured = lane.last_candidates_structured
                candidates_symbolic = lane.last_candidates_symbolic
                if candidates_structured is None or candidates_symbolic is None:
                    if hop == 1:
                        candidates_structured = graph.build_paths_from(anchor_id, hop=1)
                    else:
                        frontier_paths = lane.frontier_paths or []
                        candidates_structured = _expand_paths_one_hop(frontier_paths, graph) if frontier_paths else []
                    candidates_symbolic = self.path_formatter.to_symbolic_paths(candidates_structured)
                if not candidates_structured:
                    return
                question_text = (request.payload.intent or "")
                if extra_context:
                    question_text = f"{question_text}\n\nPrior evidence:\n{extra_context}"
                scores = await self.ranker.async_rank_paths(question=question_text, candidate_paths_repr=candidates_symbolic)
                # MMR-based selection to promote diversity among similarly scored paths
                try:
                    chosen_idx = mmr_select_indices(
                        scores=scores,
                        candidate_texts=candidates_symbolic,
                        query_text=request.payload.intent or "",
                        embedding_manager=self.embedding_manager,
                        k=self.config.select_k_per_hop,
                        alpha=0.7,
                        lam=0.7,
                    )
                except Exception:
                    # Fallback to relative-top selection if embeddings/MMR fail
                    chosen_idx = select_by_relative_top(scores, relative_gap=0.25, max_k=self.config.select_k_per_hop)
                # Append selected (ranking stage) to trace as well (outermost edge)
                try:
                    # Carry forward the judge reason from the selection stage for this lane if available.
                    carried_reason = (lane.last_reason or "").strip() or "No judge reason provided"
                    # Build detailed scored candidates: index, score, and path string
                    scored_detail = []
                    try:
                        for k_i, v_sc in (scores or {}).items():
                            try:
                                ii = int(k_i)
                                if 0 <= ii < len(candidates_symbolic):
                                    scored_detail.append(
                                        {"index": ii, "score": float(v_sc), "path": candidates_symbolic[ii]}
                                    )
                            except Exception:
                                continue
                    except Exception:
                        scored_detail = []
                    hop_entry = {
                        "iteration": hop,
                        "anchor_concept": lane.anchor_name,
                        "selected": [],
                        # For transparency: raw LLM rank scores and final MMR-chosen indices
                        "ranker_scored_paths": scored_detail,
                        "mmr_selected_indices": list(chosen_idx),
                        "selected_paths_symbolic": [
                            candidates_symbolic[i] for i in chosen_idx if 0 <= i < len(candidates_symbolic)
                        ],
                        "judge_reason_for_sufficiency": carried_reason,
                    }
                    seen_hop_edges: Set[Tuple[str, str, str]] = set()
                    for i in chosen_idx:
                        if 0 <= i < len(candidates_structured):
                            path = candidates_structured[i]
                            edge = last_edge(path, _name_for, _rel_label)
                            if edge:
                                key = (edge["from"], edge["to"], edge["relation"])
                                if key not in seen_hop_edges:
                                    seen_hop_edges.add(key)
                                    hop_entry["selected"].append(edge)
                    if hop_entry["selected"]:
                        trace["iterations"].append(hop_entry)
                except Exception as e:
                    print("[SingleEntity][WARN][trace-rank]", e)
                # Accumulate selections and prepare expansion frontier
                outer_node_ids: Set[str] = set()
                for i in chosen_idx:
                    if 0 <= i < len(candidates_structured):
                        path = candidates_structured[i]
                        k = path_key(path)
                        if k not in lane.seen_path_keys:
                            lane.seen_path_keys.add(k)
                            lane.selected_structured.append(path)
                            # Lazily render NL only for selected paths
                            nl_text = self.path_formatter.to_natural_language([path])[0] if path else ""
                            lane.selected_nl.append(nl_text)
                        if path and path[-1].get("kind") == "concept":
                            last_meta = path[-1].get("value") or {}
                            oid = last_meta.get("id")
                            if oid:
                                outer_node_ids.add(oid)
                # Update lane frontier for next hop: selected paths from this hop
                lane.frontier_paths = [
                    candidates_structured[i] for i in chosen_idx if 0 <= i < len(candidates_structured)
                ]
                if not outer_node_ids:
                    return
                # Fetch relations for each outer node concurrently
                rel_tasks = [self.repo.relations_for_async(oid) for oid in outer_node_ids]
                rel_results = await asyncio.gather(*rel_tasks, return_exceptions=True)
                all_relations: List[Dict[str, Any]] = []
                needed_concept_ids: Set[str] = set()
                for rels in rel_results:
                    if isinstance(rels, Exception):
                        continue
                    for rel in rels or []:
                        all_relations.append(rel)
                        for nid in rel.get("node_ids", []) or []:
                            if nid and nid not in graph.nodes:
                                needed_concept_ids.add(nid)
                neighbor_metas: List[Dict[str, Any]] = []
                if needed_concept_ids:
                    try:
                        neighbor_metas = await self.repo.concepts_by_ids_async(list(needed_concept_ids))
                    except Exception:
                        neighbor_metas = []
                graph.add_relations_and_nodes(all_relations, neighbor_metas)

            # Run ranker stage: if a winning lane exists, rank only that lane; else rank all lanes
            if winning_lane_index is not None:
                await rank_and_expand_lane(winning_lane_index, lanes[winning_lane_index])
                # After ranking the winning lane for this hop, stop further expansion
                break
            else:
                await asyncio.gather(*(rank_and_expand_lane(i, lane) for i, lane in enumerate(lanes)))

        # Compose the output: prefer winning lane if any, else aggregate all
        if winning_lane_index is not None:
            wl = lanes[winning_lane_index]
            selected_structured = wl.selected_structured
            sufficient = True
        else:
            # No judge-declared sufficiency: do not surface any evidence paths
            # (only final judge selections should appear in evidence)
            selected_structured = []
            sufficient = False

        # Build evidence.paths (symbolic) from ranker-selected structured paths
        evidence_paths: List[Dict[str, Any]] = []
        global_concept_ids: Set[str] = set()
        # Collect detailed concepts and relations used across all paths
        details_concepts: Dict[str, Dict[str, Any]] = {}
        details_relations: List[Dict[str, Any]] = []
        for idx, path in enumerate(selected_structured or []):
            # Symbolic string for the path
            try:
                symbolic = self.path_formatter.to_symbolic_paths([path])[0]
            except Exception:
                symbolic = ""
            # Collect details for this path
            path_concept_ids: Set[str] = set()
            for seg in path:
                if seg.get("kind") == "concept":
                    c = seg.get("value") or {}
                    cid = c.get("id")
                    if cid:
                        global_concept_ids.add(cid)
                        path_concept_ids.add(cid)
                        if cid not in details_concepts:
                            details_concepts[cid] = {
                                "concept_id": cid,
                                "name": _name_for(c),
                                "description": c.get("description"),
                                "type": c.get("type"),
                                "attributes": c.get("attributes"),
                            }
                elif seg.get("kind") == "relation":
                    r = seg.get("value") or {}
                    # capture the relation object as-is with minimal normalization
                    details_relations.append(
                        {
                            "id": r.get("id"),
                            "relationship": r.get("relationship") or r.get("relation"),
                            "node_ids": list(r.get("node_ids") or []),
                            "attributes": r.get("attributes"),
                        }
                    )
            evidence_paths.append({"path_id": f"p{idx+1}", "symbolic": symbolic})

        # When insufficient and no paths selected, include top similar concept with its relations and first neighbors
        if not evidence_paths:
            try:
                top = (enriched or [None])[0] or {}
                top_concept = top.get("concept") or {}
                neighbor_concepts = top.get("neighbor_concepts") or []
                relations = top.get("relations") or []
                # Add concept + neighbors to details
                for c in [top_concept, *neighbor_concepts]:
                    if not isinstance(c, dict):
                        continue
                    cid = c.get("id")
                    if not cid:
                        continue
                    global_concept_ids.add(cid)
                    if cid not in details_concepts:
                        details_concepts[cid] = {
                            "concept_id": cid,
                            "name": _name_for(c),
                            "description": c.get("description"),
                            "type": c.get("type"),
                            "attributes": c.get("attributes"),
                        }
                # Add relations as-is (normalized)
                for r in relations:
                    if not isinstance(r, dict):
                        continue
                    details_relations.append(
                        {
                            "id": r.get("id"),
                            "relationship": r.get("relationship") or r.get("relation"),
                            "node_ids": list(r.get("node_ids") or []),
                            "attributes": r.get("attributes"),
                        }
                    )
            except Exception:
                pass

        content = {
            "evidence": {
                "entity": entity,
                "status": "sufficient" if sufficient else "insufficient",
                "summary": {
                    "supporting_paths": len(evidence_paths),
                    "unique_concepts": len(global_concept_ids),
                },
                "paths": evidence_paths,
                "details": {
                    "concepts": list(details_concepts.values()),
                    "relations": details_relations,
                },
                "metadata": {
                    "retrieval_mode": "single_entity",
                    "pruning_applied": True,
                    "llm_assisted": True,
                },
            },
            "trace": trace,
        }
        # Add per-gather LLM call count to the trace
        try:
            trace["llm_calls"] = max(0, get_llm_call_count() - llm_calls_before)
        except Exception:
            trace["llm_calls"] = 0
        # Print the trace to terminal for visibility
        try:
            import json as _json

        except Exception:
            pass
        return TKFKnowledgeRecord(type="json", content=content)
