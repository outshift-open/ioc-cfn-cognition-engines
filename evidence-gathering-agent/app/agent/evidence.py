# app/evidence/evidence.py
import uuid
from typing import List, Dict, Any
import numpy as np

from .embeddings import EmbeddingManager
from ..api.schemas import (
    ReasonerCognitionRequest,
    ReasonerCognitionResponse,
    TKFKnowledgeRecord,
    Header
)
from .single_entity import (
    SingleEntityEvidenceEngine,
    SingleEntityConfig,
    ConceptRepository,
)
from .multi_entities import MultiEntityEvidenceEngine, MultiEntityConfig
from .utiles import PathFormatter
from .llm_clients import EvidenceJudge, EvidenceRanker, QueryDecomposer, EntityExtractor as LLMEntityExtractor

embedding_manager = EmbeddingManager()

# Local thin wrapper for entity extraction (migrated from entity_extractor.py)
def extract_entities(request: ReasonerCognitionRequest) -> List[Dict]:
    """
    Thin wrapper to use the LLM-based entity extractor client with a system prompt.
    """
    client = LLMEntityExtractor(temperature=0.2)
    return client.extract_entities_from_request(request)


async def process_evidence(request: ReasonerCognitionRequest, repo_adapter=None) -> ReasonerCognitionResponse:
    response_id = request.request_id

    response_header = Header(
        workspace_id=request.header.workspace_id,
        mas_id=request.header.mas_id,
        agent_id=request.header.agent_id
    )

    print("[Evidence] Starting entity extraction via LLM (or fallback).")
    entities = extract_entities(request)  # 1) Azure LLM (or fallback)
    print(f"[Evidence] Extracted {len(entities)} entities.")
    if not entities:
        return ReasonerCognitionResponse(
            header=response_header,
            response_id=response_id,
            records=[],
            metadata={"source": "evidence.process_evidence", "note": "no_entities"},
        )

    # Run SingleEntityEvidenceEngine sequentially per entity (concat outputs).
    judge = EvidenceJudge()
    ranker = EvidenceRanker()
    # 1b) Request decomposition via LLM; attach to request.meta for trace surfacing
    try:
        decomposer = QueryDecomposer()
        ent_names = []
        try:
            ent_names = [str(e.get("name")).strip() for e in (entities or []) if isinstance(e, dict) and e.get("name")]
        except Exception:
            ent_names = []
        decomposition = await decomposer.async_decompose(request.payload.intent or "", ent_names)
    except Exception:
        decomposition = []

    try:
        request.payload.metadata = (request.payload.metadata or {})  # type: ignore[attr-defined]
        request.payload.metadata["request_decomposition"] = decomposition  # type: ignore[index]
    except Exception:
        pass

    path_formatter = PathFormatter()
    repo = ConceptRepository(repo_adapter)
    config = SingleEntityConfig(top_k_similar=3, select_k_per_hop=3, max_depth=4)

    records_out: List[TKFKnowledgeRecord] = []
    subquery_results: List[Dict[str, Any]] = []
    prior_paths: List[str] = []

    items = decomposition if decomposition else [{"index": 1, "sentence": request.payload.intent or "", "entities": ent_names[:1]}]
    for item in items:
        sent = str(item.get("sentence") or "").strip()
        ents = item.get("entities") or []
        extra_context = "\n".join(prior_paths[-8:]) if prior_paths else ""
        if len(ents) == 1:
            engine = SingleEntityEvidenceEngine(
                embedding_manager=embedding_manager,
                repo=repo,
                path_formatter=path_formatter,
                judge=judge,
                ranker=ranker,
                config=config,
            )
            ent_dict = {"name": ents[0]}
            print(f"[Evidence] Subquery (single): {sent} | entity={ents[0]}")
            rec = await engine.gather(request, ent_dict, extra_context=extra_context)
            records_out.append(rec)
            try:
                paths = (rec.content or {}).get("evidence", {}).get("paths", [])  # type: ignore[dict-item]
                syms = [p.get("symbolic") for p in paths if isinstance(p, dict) and p.get("symbolic")]
            except Exception:
                syms = []
            prior_paths.extend(syms or [])
            subquery_results.append({"sentence": sent, "entities": ents, "paths_symbolic": syms, "status": (rec.content or {}).get("evidence", {}).get("status")})  # type: ignore[dict-item]
        elif len(ents) >= 2:
            print(f"[Evidence] Subquery (multi-entity): {sent} | entities={ents[:2]}")
            me_engine = MultiEntityEvidenceEngine(
                embedding_manager=embedding_manager,
                data_layer=repo_adapter,
                judge=judge,
                ranker=ranker,
                config=MultiEntityConfig(top_k_candidates=2, max_depth=4, pre_rank_limit=20, mmr_top_k=5, concurrency_limit=3),
            )
            pair_entities = {"source": ents[0], "target": ents[1]}
            rec = await me_engine.gather(request, pair_entities, extra_context=extra_context)
            records_out.append(rec)
            try:
                paths = (rec.content or {}).get("evidence", {}).get("paths", [])  # type: ignore[dict-item]
                syms = [p.get("symbolic") for p in paths if isinstance(p, dict) and p.get("symbolic")]
            except Exception:
                syms = []
            prior_paths.extend(syms or [])
            subquery_results.append({"sentence": sent, "entities": ents[:2], "paths_symbolic": syms, "status": (rec.content or {}).get("evidence", {}).get("status")})  # type: ignore[dict-item]
        else:
            subquery_results.append({"sentence": sent, "entities": [], "paths_symbolic": [], "status": "no_entities"})

    return ReasonerCognitionResponse(
        header=response_header,
        response_id=response_id,
        records=records_out,
        metadata={
            "source": "evidence.process_evidence",
            "mode": "decomposed_subqueries",
            "lanes": len(items),
            "returned": len(records_out),
            "request_decomposition": decomposition,
            "subquery_results": subquery_results,
        },
    )
