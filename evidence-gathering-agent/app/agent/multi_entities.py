from typing import List, Dict, Any, Optional, Tuple, Set
import asyncio
from dataclasses import dataclass

import numpy as np

from ..api.schemas import ReasonerCognitionRequest, TKFKnowledgeRecord

from .embeddings import EmbeddingManager
from .llm_clients import EvidenceJudge, EvidenceRanker, get_llm_call_count
from .utiles import mmr_select_indices


def _name_for(meta: Dict[str, Any]) -> str:
    n = (meta or {}).get("name")
    return (n.strip() if isinstance(n, str) else "") or (meta or {}).get("id") or "unknown"


def _rel_label(rel: Dict[str, Any]) -> str:
    return str((rel or {}).get("relationship") or (rel or {}).get("relation") or "").strip() or "related_to"


class MultiEntityConfig:
    def __init__(
        self,
        top_k_candidates: int = 2,
        max_depth: int = 4,
        pre_rank_limit: int = 20,
        mmr_top_k: int = 5,
        concurrency_limit: int = 3,
    ):
        self.top_k_candidates = top_k_candidates
        self.max_depth = max_depth
        self.pre_rank_limit = pre_rank_limit
        self.mmr_top_k = mmr_top_k
        self.concurrency_limit = concurrency_limit


@dataclass(frozen=True)
class Pair:
    source_id: str
    target_id: str
    source_name: str
    target_name: str


class MultiEntityEvidenceEngine:
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        data_layer,
        judge: EvidenceJudge,
        ranker: EvidenceRanker,
        config: Optional[MultiEntityConfig] = None,
        concept_repo=None,
    ):
        self.embedding_manager = embedding_manager
        self.data_layer = data_layer
        self.judge = judge
        self.ranker = ranker
        self.config = config or MultiEntityConfig()
        # When set, used for similar-concept retrieval (vector DB or cache+graph fallback); else data_layer.search_similar_with_neighbors
        self.concept_repo = concept_repo

    def _entity_to_query_vec(self, entity: Dict[str, Any]) -> List[float]:
        text = f"{entity.get('description') or ''}{entity.get('name') or ''}"
        chunks = self.embedding_manager.preprocess_text(text)
        vectors = self.embedding_manager.generate_embeddings(chunks)
        if isinstance(vectors, list) and vectors and isinstance(vectors[0], list):
            vec = np.mean(np.array(vectors, dtype=np.float32), axis=0).tolist()
        else:
            vec = np.array(vectors, dtype=np.float32).tolist()
        return vec

    async def _top_k_candidates(self, entity: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        query_vec = self._entity_to_query_vec(entity)
        entity_name = (entity.get("name") or "").strip() or None
        enriched = await (
            self.concept_repo.similar_with_neighbors_async(query_vec, k, entity_text=entity_name)
            if self.concept_repo
            else asyncio.to_thread(self.data_layer.search_similar_with_neighbors, query_vec, k)
        )
        out: List[Dict[str, Any]] = []
        for it in enriched or []:
            if (it or {}).get("concept", {}).get("id"):
                out.append(it)
            if len(out) >= k:
                break
        return out

    def _build_pairs(self, src: List[Dict[str, Any]], tgt: List[Dict[str, Any]]) -> List[Pair]:
        pairs: List[Pair] = []
        seen: Set[Tuple[str, str]] = set()
        # S1 with each of T1, T2
        for s in src[: self.config.top_k_candidates]:
            sc = (s or {}).get("concept") or {}
            s_id = sc.get("id")
            s_name = _name_for(sc)
            if not s_id:
                continue
            for t in tgt[: self.config.top_k_candidates]:
                tc = (t or {}).get("concept") or {}
                t_id = tc.get("id")
                t_name = _name_for(tc)
                if not t_id:
                    continue
                key = (s_id, t_id)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(Pair(source_id=s_id, target_id=t_id, source_name=s_name, target_name=t_name))
        # T1 with each of S1, S2 (adds reverse combos that may not have appeared)
        for t in tgt[: self.config.top_k_candidates]:
            tc = (t or {}).get("concept") or {}
            t_id = tc.get("id")
            t_name = _name_for(tc)
            if not t_id:
                continue
            for s in src[: self.config.top_k_candidates]:
                sc = (s or {}).get("concept") or {}
                s_id = sc.get("id")
                s_name = _name_for(sc)
                if not s_id:
                    continue
                key = (t_id, s_id)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(Pair(source_id=t_id, target_id=s_id, source_name=t_name, target_name=s_name))
        return pairs

    def _paths_call(self, source_id: str, target_id: str) -> List[Dict[str, Any]]:
        payload = {
            "source_id": source_id,
            "target_id": target_id,
            "max_depth": self.config.max_depth,
            "limit": self.config.pre_rank_limit,
        }
        url = self.data_layer.tkf_data_logic_svc_url + "/paths"
        try:
            resp = self.data_layer.post_to_data_logic_svc(url, payload)
            if resp is not None and getattr(resp, "status_code", None) == 200:
                j = resp.json()
                paths = j.get("paths") if j else []
                if not paths:
                    print(f"[MultiEntity] Paths API returned 200 but no paths: source_id={source_id!r}, target_id={target_id!r}, status={j.get('status') if j else None}")
                return paths or []
            print(f"[MultiEntity] Paths API non-200: source_id={source_id!r}, target_id={target_id!r}, status_code={getattr(resp, 'status_code', None)}")
        except Exception as e:
            print(f"[MultiEntity] Paths API error: source_id={source_id!r}, target_id={target_id!r}, error={e}")
        return []

    async def _paths_call_async(self, source_id: str, target_id: str) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self._paths_call, source_id, target_id)

    async def gather(
        self,
        request: ReasonerCognitionRequest,
        entities: Dict[str, Any],
        extra_context: Optional[str] = None,
    ) -> TKFKnowledgeRecord:
        # Extract names from entities mapping
        e1 = {"name": entities.get("source") or ""}
        e2 = {"name": entities.get("target") or ""}
        llm_calls_before = get_llm_call_count()

        # Top-k for both entities
        k = self.config.top_k_candidates
        src_top, tgt_top = await asyncio.gather(self._top_k_candidates(e1, k), self._top_k_candidates(e2, k))

        # Trace: top similar concepts for each entity
        def _similar_trace(items: List[Dict[str, Any]]) -> Dict[str, Any]:
            out_neighbors: List[Dict[str, str]] = []
            try:
                top = (items or [None])[0] or {}
                concept = (top or {}).get("concept") or {}
                anchor_name = _name_for(concept)
                seen_edge: Set[Tuple[str, str, str]] = set()
                id_to_name: Dict[str, str] = { (concept or {}).get("id"): anchor_name }
                for nc in (top or {}).get("neighbor_concepts") or []:
                    cid = (nc or {}).get("id")
                    if cid:
                        id_to_name[cid] = _name_for(nc)
                for rel in (top or {}).get("relations") or []:
                    node_ids = (rel or {}).get("node_ids") or []
                    for nid in node_ids:
                        if not nid or nid == (concept or {}).get("id"):
                            continue
                        edge = ( (concept or {}).get("id"), nid, _rel_label(rel) )
                        if edge in seen_edge:
                            continue
                        seen_edge.add(edge)
                        out_neighbors.append({"to": id_to_name.get(nid) or nid, "relation": _rel_label(rel)})
                return {"anchor_concept": anchor_name, "firs_neighbour": out_neighbors}
            except Exception:
                return {"anchor_concept": _name_for((items or [{}])[0].get("concept") if items else {}), "firs_neighbour": []}

        trace: Dict[str, Any] = {
            "extracted_entities": [e1.get("name"), e2.get("name")],
            "tope_similar_concepts": [_similar_trace(src_top), _similar_trace(tgt_top)],
            "iterations": [],
            "lanes_count": 0,
            "sufficient": False,
            "winning": None,
            "request_decomposition": ((getattr(request, "meta", {}) or {}) or {}).get("request_decomposition"),
            "pass_on_context": (extra_context or ""),
        }

        pairs = self._build_pairs(src_top, tgt_top)
        trace["lanes_count"] = len(pairs)

        # Per-pair pipeline
        sem = asyncio.Semaphore(self.config.concurrency_limit)

        async def process_pair(pair: Pair):
            async with sem:
                # Step 1: fetch paths from data-logic
                paths = await self._paths_call_async(pair.source_id, pair.target_id)
                # Build name mapping for all involved ids
                concept_ids_local: Set[str] = set()
                for p in paths or []:
                    for ed in (p or {}).get("edges") or []:
                        fid = (ed or {}).get("from_id")
                        tid = (ed or {}).get("to_id")
                        if fid:
                            concept_ids_local.add(fid)
                        if tid:
                            concept_ids_local.add(tid)
                id_to_name_local: Dict[str, str] = {}
                try:
                    metas_local = await self.data_layer.get_concepts_by_ids(list(concept_ids_local))
                    for c in metas_local or []:
                        cid = (c or {}).get("id")
                        if cid:
                            id_to_name_local[cid] = _name_for(c)
                except Exception:
                    pass
                # candidates for LLM are name-based symbolic strings
                candidates_symbolic = []
                for p in paths or []:
                    hops: List[str] = []
                    for ed in (p or {}).get("edges") or []:
                        fid = (ed or {}).get("from_id"); tid = (ed or {}).get("to_id"); rel = (ed or {}).get("relation")
                        frn = id_to_name_local.get(fid) or fid
                        ton = id_to_name_local.get(tid) or tid
                        if frn and ton and rel:
                            hops.append(f"{frn} -{rel}-> {ton}")
                    candidates_symbolic.append(" ; ".join(hops))
                # Filter out empties
                candidates_symbolic = [s for s in candidates_symbolic if s]
                # Step 2: rank with LLM, then MMR
                if not candidates_symbolic:
                    return {
                        "pair": pair,
                        "paths": [],
                        "selected_indices": [],
                        "scores": {},
                        "candidates_symbolic": [],
                        "sufficient": False,
                        "reason": "no_candidate_paths",
                    }
                question_text = (request.payload.intent or "")
                if extra_context:
                    question_text = f"{question_text}\n\nPrior evidence:\n{extra_context}"
                scores = await self.ranker.async_rank_paths(question=question_text, candidate_paths_repr=candidates_symbolic)
                try:
                    chosen_idx = mmr_select_indices(
                        scores=scores,
                        candidate_texts=candidates_symbolic,
                        query_text=request.payload.intent or "",
                        embedding_manager=self.embedding_manager,
                        k=self.config.mmr_top_k,
                        alpha=0.7,
                        lam=0.7,
                    )
                except Exception:
                    # simple fallback: top by score
                    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
                    chosen_idx = [int(i) for i, _ in ordered[: self.config.mmr_top_k]]
                # Step 3: judge sufficiency on selected paths
                chosen_symbolic = [candidates_symbolic[i] for i in chosen_idx if 0 <= i < len(candidates_symbolic)]
                _, sufficient, reason = await self.judge.async_select_paths_and_check_sufficiency(
                    question=question_text, candidate_paths=chosen_symbolic, select_k=len(chosen_symbolic) or 1
                )
                return {
                    "pair": pair,
                    "paths": paths,
                    "selected_indices": chosen_idx,
                    "scores": scores,
                    "candidates_symbolic": candidates_symbolic,
                    "sufficient": bool(sufficient),
                    "reason": reason,
                }

        results = await asyncio.gather(*(process_pair(p) for p in pairs))

        # Decide winning lane
        winning = next((r for r in results if r.get("sufficient")), None)
        if winning:
            trace["sufficient"] = True
            trace["winning"] = {
                "source": winning["pair"].source_name,
                "target": winning["pair"].target_name,
                "reason_for_sufficiency": winning.get("reason"),
            }

        # Add iterations trace per pair
        for hop_idx, r in enumerate(results, start=1):
            pair: Pair = r["pair"]
            # Use the name-based candidates that were ranked/judged
            candidates_symbolic = list(r.get("candidates_symbolic") or [])
            # detail scored paths
            scored_detail = []
            try:
                for k_i, v_sc in (r.get("scores") or {}).items():
                    i = int(k_i)
                    if 0 <= i < len(candidates_symbolic):
                        scored_detail.append({"index": i, "score": float(v_sc), "path": candidates_symbolic[i]})
            except Exception:
                scored_detail = []
            trace["iterations"].append(
                {
                    "iteration": hop_idx,
                    "anchor_concept": f"{pair.source_name} -> {pair.target_name}",
                    "selected": [],
                    "ranker_scored_paths": scored_detail,
                    "mmr_selected_indices": list(r.get("selected_indices") or []),
                    "selected_paths_symbolic": [
                        candidates_symbolic[i] for i in (r.get("selected_indices") or []) if 0 <= i < len(candidates_symbolic)
                    ],
                    "judge_reason_for_sufficiency": r.get("reason"),
                    "sufficient": bool(r.get("sufficient")),
                }
            )

        # Build evidence based on winning or best-scored (fallback).
        # resolved_concepts must be fetched in async context (await get_concepts_by_ids) and passed in.
        def build_evidence_from(
            r: Dict[str, Any],
            resolved_concepts: Optional[List[Dict[str, Any]]] = None,
        ) -> Tuple[List[Dict[str, Any]], Set[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
            selected_indices: List[int] = list(r.get("selected_indices") or [])
            paths: List[Dict[str, Any]] = list(r.get("paths") or [])
            selected_paths: List[Dict[str, Any]] = [paths[i] for i in selected_indices if 0 <= i < len(paths)]
            # symbolic list for UI
            evidence_paths: List[Dict[str, Any]] = []
            concept_ids: Set[str] = set()
            # Collect involved concept ids first
            for p in selected_paths or []:
                for ed in p.get("edges") or []:
                    fid = (ed or {}).get("from_id")
                    tid = (ed or {}).get("to_id")
                    if fid:
                        concept_ids.add(fid)
                    if tid:
                        concept_ids.add(tid)
            # use pre-resolved concepts (fetched with await get_concepts_by_ids in caller)
            details_concepts: List[Dict[str, Any]] = []
            id_to_name: Dict[str, str] = {}
            for c in resolved_concepts or []:
                cid = (c or {}).get("id")
                if not cid:
                    continue
                id_to_name[cid] = _name_for(c)
                details_concepts.append(
                    {
                        "concept_id": cid,
                        "name": _name_for(c),
                        "type": c.get("type"),
                        "description": c.get("description"),
                    }
                )
            # relations reconstructed from paths
            details_relations: List[Dict[str, Any]] = []
            for idx, p in enumerate(selected_paths or []):
                # Build symbolic using names (fallback to id)
                hops: List[str] = []
                for ed in p.get("edges") or []:
                    fid = (ed or {}).get("from_id")
                    tid = (ed or {}).get("to_id")
                    rel = (ed or {}).get("relation")
                    frn = id_to_name.get(fid) or fid
                    ton = id_to_name.get(tid) or tid
                    if frn and ton and rel:
                        hops.append(f"{frn} -{rel}-> {ton}")
                    details_relations.append(
                        {
                            "id": None,
                            "relationship": (ed or {}).get("relation"),
                            "node_ids": [fid, tid],
                            "attributes": None,
                        }
                    )
                evidence_paths.append({"path_id": f"p{idx+1}", "symbolic": " ; ".join(hops)})
            return evidence_paths, concept_ids, details_relations + [], details_concepts  # relations and concepts

        def _collect_concept_ids(r: Dict[str, Any]) -> Set[str]:
            ids: Set[str] = set()
            paths = list(r.get("paths") or [])
            selected_indices = list(r.get("selected_indices") or [])
            for i in selected_indices:
                if 0 <= i < len(paths):
                    for ed in (paths[i] or {}).get("edges") or []:
                        fid = (ed or {}).get("from_id")
                        tid = (ed or {}).get("to_id")
                        if fid:
                            ids.add(fid)
                        if tid:
                            ids.add(tid)
            return ids

        if winning:
            concept_ids_resolved = _collect_concept_ids(winning)
            metas_resolved = await self.data_layer.get_concepts_by_ids(list(concept_ids_resolved))
            evidence_paths, global_ids, details_relations, details_concepts = build_evidence_from(
                winning, resolved_concepts=metas_resolved
            )
            status = "sufficient"
        else:
            # pick the pair with most selected indices or fallback to first with any paths
            candidate = max(results, key=lambda r: len(r.get("selected_indices") or []), default=None)
            if candidate and candidate.get("selected_indices"):
                concept_ids_resolved = _collect_concept_ids(candidate)
                metas_resolved = await self.data_layer.get_concepts_by_ids(list(concept_ids_resolved))
                evidence_paths, global_ids, details_relations, details_concepts = build_evidence_from(
                    candidate, resolved_concepts=metas_resolved
                )
            else:
                evidence_paths, global_ids, details_relations, details_concepts = [], set(), [], []
            status = "insufficient"

        content = {
            "evidence": {
                "entity": {
                    "source": {"name": e1.get("name")},
                    "target": {"name": e2.get("name")},
                },
                "status": status,
                "summary": {
                    "supporting_paths": len(evidence_paths),
                    "unique_concepts": len(global_ids),
                },
                "paths": evidence_paths,
                "details": {
                    # Concepts without embeddings/metadata (strict view)
                    "concepts": details_concepts,
                    "relations": details_relations,
                },
                "metadata": {
                    "retrieval_mode": "multi_entity",
                    "pruning_applied": True,
                    "llm_assisted": True,
                },
            },
            "trace": trace,
        }

        try:
            content["trace"]["llm_calls"] = max(0, get_llm_call_count() - llm_calls_before)  # type: ignore[index]
        except Exception:
            content["trace"]["llm_calls"] = 0  # type: ignore[index]

        return TKFKnowledgeRecord(type="json", content=content)

