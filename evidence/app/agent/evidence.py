# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

# app/evidence/evidence.py
import logging
import uuid
from typing import List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

from .embeddings import EmbeddingManager
from ..api.schemas import (
    ReasonerCognitionRequest,
    ReasonerCognitionResponse,
    KnowledgeRecord,
    Header
)
from .single_entity import (
    SingleEntityEvidenceEngine,
    SingleEntityConfig,
    ConceptRepository,
)
from .multi_entities import MultiEntityEvidenceEngine, MultiEntityConfig
from .utiles import PathFormatter
from .llm_clients import EvidenceJudge, EvidenceRanker, ResponseGenerator, QueryDecomposer, EntityExtractor as LLMEntityExtractor

embedding_manager = EmbeddingManager()

# Local thin wrapper for entity extraction (migrated from entity_extractor.py)
def extract_entities(request: ReasonerCognitionRequest) -> List[Dict]:
    """
    Thin wrapper to use the LLM-based entity extractor client with a system prompt.
    """
    client = LLMEntityExtractor(temperature=0.2)
    return client.extract_entities_from_request(request)


async def process_evidence(
    request: ReasonerCognitionRequest,
    repo_adapter=None,
    cache_layer=None,
) -> ReasonerCognitionResponse:
    response_id = request.request_id

    response_header = Header(
        workspace_id=request.header.workspace_id,
        mas_id=request.header.mas_id,
        agent_id=request.header.agent_id
    )

    logger.info("[Evidence] Starting entity extraction via LLM.")
    entities = LLMEntityExtractor(temperature=0).extract_entities_from_request(request)
    logger.info("[Evidence] Extracted %d entities.", len(entities))
    if not entities:
        return ReasonerCognitionResponse(
            header=response_header,
            response_id=response_id,
            records=[],
            metadata={"source": "evidence.process_evidence", "note": "no_entities"},
        )

    try:
        ent_names = [str(e.get("name")).strip() for e in (entities or []) if isinstance(e, dict) and e.get("name")]
    except Exception:
        ent_names = []
    n_entities = len(ent_names)
    intent = request.payload.intent or ""

    # Decomposition only when 3+ entities; 1 → single-entity, 2 → multi-entity directly
    decomposition: List[Dict[str, Any]] = []
    if n_entities >= 3:
        try:
            decomposer = QueryDecomposer()
            decomposition = await decomposer.async_decompose(intent, ent_names)
        except Exception:
            decomposition = []

    if n_entities == 1:
        items = [{"index": 1, "sentence": intent, "entities": ent_names}]
        mode = "single_entity"
    elif n_entities == 2:
        items = [{"index": 1, "sentence": intent, "entities": ent_names}]
        mode = "multi_entity"
    else:
        # 3+ entities: use decomposition or fallback to first two
        items = decomposition if decomposition else [{"index": 1, "sentence": intent, "entities": ent_names[:2]}]
        mode = "decomposed"

    try:
        request.payload.metadata = (request.payload.metadata or {})  # type: ignore[attr-defined]
        request.payload.metadata["request_decomposition"] = decomposition  # type: ignore[index]
    except Exception:
        pass

    judge = EvidenceJudge()
    ranker = EvidenceRanker()
    response_generator = ResponseGenerator(temperature=0.2)

    path_formatter = PathFormatter()
    # Similar concepts: in-process cache_layer only (unified gateway sets app.state.cache_layer).
    repo = ConceptRepository(repo_adapter, cache_layer=cache_layer)
    config = SingleEntityConfig(top_k_similar=1, select_k_per_hop=3, max_depth=4)

    records_out: List[KnowledgeRecord] = []
    subquery_results: List[Dict[str, Any]] = []
    prior_paths: List[str] = []

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
                response_generator=response_generator,
            )
            ent_dict = {"name": ents[0]}
            rec = await engine.gather(request, ent_dict, extra_context=extra_context)
            # The format of rec is a dictionary with the following keys:
            # - content: a dictionary with the following keys:
            #   - evidence: a dictionary with the following keys:
            #     - paths: a list of dictionaries with the following keys:
            #       - symbolic: a string
            #     - status: a string
            # - status: a string
            # - message: a string
            # - trace: a dictionary with the following keys:
            #   - extracted_entity: a string
            #   - tope_similar_concepts: a list of dictionaries with the following keys: ...
   
            records_out.append(rec)
            try:
                paths = (rec.content or {}).get("evidence", {}).get("paths", [])  # type: ignore[dict-item]
                syms = [p.get("symbolic") for p in paths if isinstance(p, dict) and p.get("symbolic")]
            except Exception:
                syms = []
            prior_paths.extend(syms or [])
            subquery_results.append({"sentence": sent, "entities": ents, "paths_symbolic": syms, "status": (rec.content or {}).get("evidence", {}).get("status")})  # type: ignore[dict-item]
        elif len(ents) >= 2:
            me_engine = MultiEntityEvidenceEngine(
                embedding_manager=embedding_manager,
                data_layer=repo_adapter,
                judge=judge,
                ranker=ranker,
                config=MultiEntityConfig(top_k_candidates=1, max_depth=4, pre_rank_limit=20, mmr_top_k=5, concurrency_limit=3),
                concept_repo=repo,
                response_generator=response_generator,
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
            "mode": mode,
            "lanes": len(items),
            "returned": len(records_out),
            "request_decomposition": decomposition,
            "subquery_results": subquery_results,
        },
    )
