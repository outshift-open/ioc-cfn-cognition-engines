"""
Unit tests for extraction services and KnowledgeProcessor.
"""

import pytest

from app.agent.service import ConceptRelationshipExtractionService
from app.agent.knowledge_processor import (
    KnowledgeProcessor,
    EmbeddingManager,
    cosine_similarity,
    FASTEMBED_AVAILABLE,
)


# ---------------------------------------------------------------------------
# ConceptRelationshipExtractionService
# ---------------------------------------------------------------------------


class TestConceptRelationshipExtractionService:
    """Tests for ConceptRelationshipExtractionService (heuristic / no-LLM mode)."""

    def test_generate_id_deterministic(self, concept_relationship_service):
        id1 = concept_relationship_service._generate_id("test_string")
        id2 = concept_relationship_service._generate_id("test_string")
        assert id1 == id2

    def test_filter_spans_keeps_client_and_server(self, concept_relationship_service):
        records = [
            {"SpanKind": "Client", "SpanId": "1"},
            {"SpanKind": "Server", "SpanId": "2"},
            {"SpanKind": "Internal", "SpanId": "3"},
            {"SpanKind": "Producer", "SpanId": "4"},
        ]
        filtered = concept_relationship_service._filter_spans(records)
        assert len(filtered) == 2
        kinds = {r["SpanKind"] for r in filtered}
        assert kinds == {"Client", "Server"}

    def test_extract_important_fields_agent_id(self, concept_relationship_service):
        records = [
            {
                "ServiceName": "svc",
                "SpanAttributes": {
                    "agent_id": "my_agent",
                    "gen_ai.request.model": "gpt-4o",
                },
            }
        ]
        fields = concept_relationship_service._extract_important_fields(records)
        assert len(fields) == 1
        assert fields[0]["agent_id"] == "my_agent"
        assert fields[0]["model"] == "gpt-4o"

    def test_heuristic_extract_produces_concepts_and_relationships(
        self, concept_relationship_service
    ):
        compact = [
            {
                "ServiceName": "svc_a",
                "agent_id": "agent_x",
                "model": "gpt-4o",
                "user_prompt": "Hello?",
            }
        ]
        result = concept_relationship_service._heuristic_extract(compact)
        assert len(result["concepts"]) > 0
        concept_names = [c["name"] for c in result["concepts"]]
        assert "agent_x" in concept_names
        assert "svc_a" in concept_names
        assert "gpt-4o" in concept_names

    def test_full_pipeline_returns_expected_structure(
        self, concept_relationship_service, sample_otel_records
    ):
        result = concept_relationship_service.extract_concepts_and_relationships(
            sample_otel_records
        )
        assert "knowledge_cognition_request_id" in result
        assert "concepts" in result
        assert "relations" in result
        assert "descriptor" in result
        assert "meta" in result
        assert result["meta"]["records_processed"] > 0

    def test_custom_request_id_is_echoed(
        self, concept_relationship_service, sample_otel_records
    ):
        result = concept_relationship_service.extract_concepts_and_relationships(
            sample_otel_records, request_id="cr-custom-id"
        )
        assert result["knowledge_cognition_request_id"] == "cr-custom-id"

    def test_format_descriptor_is_used(
        self, concept_relationship_service, sample_otel_records
    ):
        result = concept_relationship_service.extract_concepts_and_relationships(
            sample_otel_records, format_descriptor="observe-sdk-otel"
        )
        assert result["descriptor"] == "observe-sdk-otel"

    def test_default_descriptor(
        self, concept_relationship_service, sample_otel_records
    ):
        result = concept_relationship_service.extract_concepts_and_relationships(
            sample_otel_records
        )
        assert result["descriptor"] == "concept relationship extraction"

    def test_empty_records(self, concept_relationship_service):
        result = concept_relationship_service.extract_concepts_and_relationships([])
        assert result["concepts"] == []
        assert result["relations"] == []
        assert result["meta"]["records_processed"] == 0

    def test_empty_records_with_custom_id(self, concept_relationship_service):
        result = concept_relationship_service.extract_concepts_and_relationships(
            [], request_id="empty-cr"
        )
        assert result["knowledge_cognition_request_id"] == "empty-cr"

    def test_concepts_have_ids(self, concept_relationship_service, sample_otel_records):
        result = concept_relationship_service.extract_concepts_and_relationships(
            sample_otel_records
        )
        for concept in result["concepts"]:
            assert "id" in concept
            assert len(concept["id"]) == 32

    def test_relations_have_node_ids(
        self, concept_relationship_service, sample_otel_records
    ):
        result = concept_relationship_service.extract_concepts_and_relationships(
            sample_otel_records
        )
        for relation in result["relations"]:
            assert "node_ids" in relation
            assert len(relation["node_ids"]) == 2


# ---------------------------------------------------------------------------
# KnowledgeProcessor
# ---------------------------------------------------------------------------


class TestKnowledgeProcessor:
    """Tests for KnowledgeProcessor."""

    def test_process_adds_dedup_meta(
        self, knowledge_processor_with_dedup, sample_extraction_result
    ):
        result = knowledge_processor_with_dedup.process(sample_extraction_result)
        assert "dedup_enabled" in result["meta"]
        assert result["meta"]["dedup_enabled"] is True
        assert "concepts_deduped" in result["meta"]
        assert "relations_deduped" in result["meta"]

    def test_name_based_dedup_removes_duplicates(self, knowledge_processor_with_dedup):
        extraction_result = {
            "concepts": [
                {
                    "id": "1",
                    "name": "agent_a",
                    "description": "Agent A",
                    "attributes": {"concept_type": "agent"},
                },
                {
                    "id": "2",
                    "name": "agent_a",
                    "description": "Agent A copy",
                    "attributes": {"concept_type": "agent"},
                },
                {
                    "id": "3",
                    "name": "agent_b",
                    "description": "Agent B",
                    "attributes": {"concept_type": "agent"},
                },
            ],
            "relations": [],
            "meta": {
                "records_processed": 0,
                "concepts_extracted": 3,
                "relations_extracted": 0,
            },
        }
        result = knowledge_processor_with_dedup.process(extraction_result)
        assert len(result["concepts"]) == 2
        assert result["meta"]["concepts_deduped"] == 1

    def test_no_dedup_preserves_all(self, knowledge_processor_no_dedup):
        extraction_result = {
            "concepts": [
                {
                    "id": "1",
                    "name": "agent_a",
                    "description": "Agent A",
                    "attributes": {"concept_type": "agent"},
                },
                {
                    "id": "2",
                    "name": "agent_a",
                    "description": "Agent A copy",
                    "attributes": {"concept_type": "agent"},
                },
            ],
            "relations": [],
            "meta": {
                "records_processed": 0,
                "concepts_extracted": 2,
                "relations_extracted": 0,
            },
        }
        result = knowledge_processor_no_dedup.process(extraction_result)
        assert len(result["concepts"]) == 2
        assert result["meta"]["dedup_enabled"] is False

    def test_relation_dedup_removes_duplicates(self, knowledge_processor_with_dedup):
        extraction_result = {
            "concepts": [
                {"id": "1", "name": "agent_a", "attributes": {"concept_type": "agent"}},
                {"id": "2", "name": "llm_a", "attributes": {"concept_type": "llm"}},
            ],
            "relations": [
                {
                    "id": "r1",
                    "node_ids": ["1", "2"],
                    "relationship": "SENDS_PROMPT_TO",
                    "attributes": {},
                },
                {
                    "id": "r2",
                    "node_ids": ["1", "2"],
                    "relationship": "SENDS_PROMPT_TO",
                    "attributes": {},
                },
                {
                    "id": "r3",
                    "node_ids": ["1", "2"],
                    "relationship": "INVOKES_TOOL",
                    "attributes": {},
                },
            ],
            "meta": {
                "records_processed": 0,
                "concepts_extracted": 2,
                "relations_extracted": 3,
            },
        }
        result = knowledge_processor_with_dedup.process(extraction_result)
        assert len(result["relations"]) == 2
        assert result["meta"]["relations_deduped"] == 1

    def test_relation_dedup_filters_invalid_nodes(self, knowledge_processor_with_dedup):
        extraction_result = {
            "concepts": [
                {"id": "1", "name": "agent_a", "attributes": {"concept_type": "agent"}},
            ],
            "relations": [
                {
                    "id": "r1",
                    "node_ids": ["1", "999"],
                    "relationship": "SENDS_PROMPT_TO",
                    "attributes": {},
                },
            ],
            "meta": {
                "records_processed": 0,
                "concepts_extracted": 1,
                "relations_extracted": 1,
            },
        }
        result = knowledge_processor_with_dedup.process(extraction_result)
        assert len(result["relations"]) == 0


# ---------------------------------------------------------------------------
# EmbeddingManager
# ---------------------------------------------------------------------------


class TestEmbeddingManager:
    """Tests for EmbeddingManager."""

    def test_embedding_manager_init(self):
        manager = EmbeddingManager()
        if FASTEMBED_AVAILABLE:
            assert manager.model is not None
        else:
            assert manager.model is None

    def test_generate_embedding_empty_text(self):
        manager = EmbeddingManager()
        result = manager.generate_embedding("")
        assert result is None

    @pytest.mark.skipif(
        not FASTEMBED_AVAILABLE,
        reason="fastembed not available",
    )
    def test_generate_embedding_returns_array(self):
        import numpy as np

        manager = EmbeddingManager()
        result = manager.generate_embedding("test text")
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Cosine Similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    """Tests for cosine similarity function."""

    def test_identical_vectors(self):
        import numpy as np

        vec = np.array([1.0, 2.0, 3.0])
        similarity = cosine_similarity(vec, vec)
        assert abs(similarity - 1.0) < 0.0001

    def test_orthogonal_vectors(self):
        import numpy as np

        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity) < 0.0001

    def test_opposite_vectors(self):
        import numpy as np

        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([-1.0, 0.0, 0.0])
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity + 1.0) < 0.0001

    def test_zero_vector(self):
        import numpy as np

        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([0.0, 0.0, 0.0])
        similarity = cosine_similarity(vec1, vec2)
        assert similarity == 0.0


# ---------------------------------------------------------------------------
# Semantic Deduplication
# ---------------------------------------------------------------------------


class TestSemanticDeduplication:
    """Tests for semantic deduplication with embeddings."""

    @pytest.mark.skipif(
        not FASTEMBED_AVAILABLE,
        reason="fastembed not available",
    )
    def test_semantic_dedup_similar_concepts(self):
        processor = KnowledgeProcessor(
            enable_embeddings=True,
            enable_dedup=True,
            similarity_threshold=0.9,
        )
        extraction_result = {
            "concepts": [
                {
                    "id": "1",
                    "name": "weather_agent",
                    "description": "Agent for weather lookup",
                    "attributes": {"concept_type": "agent"},
                },
                {
                    "id": "2",
                    "name": "weather_agent_v2",
                    "description": "Agent for weather lookup version 2",
                    "attributes": {"concept_type": "agent"},
                },
                {
                    "id": "3",
                    "name": "payment_agent",
                    "description": "Agent for payment processing",
                    "attributes": {"concept_type": "agent"},
                },
            ],
            "relations": [],
            "meta": {
                "records_processed": 0,
                "concepts_extracted": 3,
                "relations_extracted": 0,
            },
        }
        result = processor.process(extraction_result)
        for concept in result["concepts"]:
            assert "embedding" in concept.get("attributes", {})

    @pytest.mark.skipif(
        not FASTEMBED_AVAILABLE,
        reason="fastembed not available",
    )
    def test_embedding_format(self):
        processor = KnowledgeProcessor(
            enable_embeddings=True,
            enable_dedup=False,
            similarity_threshold=0.95,
        )
        extraction_result = {
            "concepts": [
                {
                    "id": "1",
                    "name": "test_agent",
                    "description": "Test agent",
                    "attributes": {"concept_type": "agent"},
                },
            ],
            "relations": [],
            "meta": {
                "records_processed": 0,
                "concepts_extracted": 1,
                "relations_extracted": 0,
            },
        }
        result = processor.process(extraction_result)
        embedding = result["concepts"][0]["attributes"].get("embedding")
        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) == 1
        assert isinstance(embedding[0], list)
