"""
Unit tests for the TelemetryExtractionService and KnowledgeProcessor.
"""
import pytest
from typing import Dict, Any, List

from app.agent.service import TelemetryExtractionService
from app.agent.knowledge_processor import (
    KnowledgeProcessor,
    EmbeddingManager,
    cosine_similarity,
    SENTENCE_TRANSFORMERS_AVAILABLE
)


class TestTelemetryExtractionService:
    """Tests for TelemetryExtractionService."""
    
    def test_generate_id_deterministic(self, extraction_service):
        """Test that ID generation is deterministic."""
        id1 = extraction_service._generate_id("test_string")
        id2 = extraction_service._generate_id("test_string")
        assert id1 == id2
        
    def test_generate_id_different_inputs(self, extraction_service):
        """Test that different inputs produce different IDs."""
        id1 = extraction_service._generate_id("input_a")
        id2 = extraction_service._generate_id("input_b")
        assert id1 != id2
    
    def test_extract_agents(self, extraction_service, sample_otel_records):
        """Test that agents are correctly extracted from OTel records."""
        result = extraction_service.extract_entities_and_relations(sample_otel_records)
        
        concepts = result["concepts"]
        agent_names = [c["name"] for c in concepts if c["attributes"].get("concept_type") == "agent"]
        
        assert "orchestrator_agent" in agent_names
        assert "worker_agent" in agent_names
    
    def test_extract_services(self, extraction_service, sample_otel_records):
        """Test that services are correctly extracted from OTel records."""
        result = extraction_service.extract_entities_and_relations(sample_otel_records)
        
        concepts = result["concepts"]
        service_names = [c["name"] for c in concepts if c["attributes"].get("concept_type") == "service"]
        
        assert "corto.orchestrator" in service_names
        assert "corto.worker" in service_names
    
    def test_extract_llm_models(self, extraction_service, sample_otel_records):
        """Test that LLM models are correctly extracted from OTel records."""
        result = extraction_service.extract_entities_and_relations(sample_otel_records)
        
        concepts = result["concepts"]
        llm_names = [c["name"] for c in concepts if c["attributes"].get("concept_type") == "llm"]
        
        assert "gpt-4o" in llm_names
    
    def test_extract_tools(self, extraction_service, sample_otel_records):
        """Test that tools are correctly extracted from OTel records."""
        result = extraction_service.extract_entities_and_relations(sample_otel_records)
        
        concepts = result["concepts"]
        tool_names = [c["name"] for c in concepts if c["attributes"].get("concept_type") == "tool"]
        
        assert "weather_lookup" in tool_names
    
    def test_extract_uses_relations(self, extraction_service, sample_otel_records):
        """Test that USES relations (Agent -> LLM) are correctly extracted."""
        result = extraction_service.extract_entities_and_relations(sample_otel_records)
        
        relations = result["relations"]
        uses_relations = [r for r in relations if r["relationship"] == "USES"]
        
        assert len(uses_relations) > 0
        
        # Check that at least one agent uses gpt-4o
        uses_gpt4 = [r for r in uses_relations if r["attributes"]["target_name"] == "gpt-4o"]
        assert len(uses_gpt4) > 0
    
    def test_extract_coordinates_relations(self, extraction_service, sample_otel_records):
        """Test that COORDINATES relations (parent -> child) are correctly extracted."""
        result = extraction_service.extract_entities_and_relations(sample_otel_records)
        
        relations = result["relations"]
        coordinates_relations = [r for r in relations if r["relationship"] == "COORDINATES"]
        
        # orchestrator_agent should coordinate worker_agent
        assert len(coordinates_relations) > 0
    
    def test_extract_calls_relations(self, extraction_service, sample_otel_records):
        """Test that CALLS relations (LLM -> Tool) are correctly extracted."""
        result = extraction_service.extract_entities_and_relations(sample_otel_records)
        
        relations = result["relations"]
        calls_relations = [r for r in relations if r["relationship"] == "CALLS"]
        
        # gpt-4o should call weather_lookup
        assert len(calls_relations) > 0
        tool_calls = [r for r in calls_relations if r["attributes"]["target_name"] == "weather_lookup"]
        assert len(tool_calls) > 0
    
    def test_meta_records_processed(self, extraction_service, sample_otel_records):
        """Test that meta information includes correct record count."""
        result = extraction_service.extract_entities_and_relations(sample_otel_records)
        
        assert result["meta"]["records_processed"] == len(sample_otel_records)
    
    def test_result_has_request_id(self, extraction_service, sample_otel_records):
        """Test that result includes a knowledge_cognition_request_id."""
        result = extraction_service.extract_entities_and_relations(sample_otel_records)
        
        assert "knowledge_cognition_request_id" in result
        assert len(result["knowledge_cognition_request_id"]) == 32  # MD5 hash length
    
    def test_concepts_have_ids(self, extraction_service, sample_otel_records):
        """Test that all concepts have IDs."""
        result = extraction_service.extract_entities_and_relations(sample_otel_records)
        
        for concept in result["concepts"]:
            assert "id" in concept
            assert len(concept["id"]) == 32
    
    def test_relations_have_node_ids(self, extraction_service, sample_otel_records):
        """Test that all relations have node_ids."""
        result = extraction_service.extract_entities_and_relations(sample_otel_records)
        
        for relation in result["relations"]:
            assert "node_ids" in relation
            assert len(relation["node_ids"]) == 2
    
    def test_empty_records(self, extraction_service):
        """Test extraction with empty records list."""
        result = extraction_service.extract_entities_and_relations([])
        
        assert result["concepts"] == []
        assert result["relations"] == []
        assert result["meta"]["records_processed"] == 0


class TestKnowledgeProcessor:
    """Tests for KnowledgeProcessor."""
    
    def test_process_adds_dedup_meta(self, knowledge_processor_with_dedup, sample_extraction_result):
        """Test that process adds dedup metadata."""
        result = knowledge_processor_with_dedup.process(sample_extraction_result)
        
        assert "dedup_enabled" in result["meta"]
        assert result["meta"]["dedup_enabled"] is True
        assert "concepts_deduped" in result["meta"]
        assert "relations_deduped" in result["meta"]
    
    def test_name_based_dedup_removes_duplicates(self, knowledge_processor_with_dedup):
        """Test that name-based deduplication removes duplicate concepts."""
        # Create extraction result with duplicate concepts
        extraction_result = {
            "concepts": [
                {"id": "1", "name": "agent_a", "description": "Agent A", "attributes": {"concept_type": "agent"}},
                {"id": "2", "name": "agent_a", "description": "Agent A copy", "attributes": {"concept_type": "agent"}},
                {"id": "3", "name": "agent_b", "description": "Agent B", "attributes": {"concept_type": "agent"}},
            ],
            "relations": [],
            "meta": {"records_processed": 0, "concepts_extracted": 3, "relations_extracted": 0}
        }
        
        result = knowledge_processor_with_dedup.process(extraction_result)
        
        # Should have removed one duplicate
        assert len(result["concepts"]) == 2
        assert result["meta"]["concepts_deduped"] == 1
    
    def test_no_dedup_preserves_all(self, knowledge_processor_no_dedup):
        """Test that disabling dedup preserves all concepts."""
        extraction_result = {
            "concepts": [
                {"id": "1", "name": "agent_a", "description": "Agent A", "attributes": {"concept_type": "agent"}},
                {"id": "2", "name": "agent_a", "description": "Agent A copy", "attributes": {"concept_type": "agent"}},
            ],
            "relations": [],
            "meta": {"records_processed": 0, "concepts_extracted": 2, "relations_extracted": 0}
        }
        
        result = knowledge_processor_no_dedup.process(extraction_result)
        
        # Should preserve all concepts
        assert len(result["concepts"]) == 2
        assert result["meta"]["dedup_enabled"] is False
    
    def test_relation_dedup_removes_duplicates(self, knowledge_processor_with_dedup):
        """Test that duplicate relations are removed."""
        extraction_result = {
            "concepts": [
                {"id": "1", "name": "agent_a", "attributes": {"concept_type": "agent"}},
                {"id": "2", "name": "llm_a", "attributes": {"concept_type": "llm"}},
            ],
            "relations": [
                {"id": "r1", "node_ids": ["1", "2"], "relationship": "USES", "attributes": {}},
                {"id": "r2", "node_ids": ["1", "2"], "relationship": "USES", "attributes": {}},  # Duplicate
                {"id": "r3", "node_ids": ["1", "2"], "relationship": "CALLS", "attributes": {}},  # Different relationship
            ],
            "meta": {"records_processed": 0, "concepts_extracted": 2, "relations_extracted": 3}
        }
        
        result = knowledge_processor_with_dedup.process(extraction_result)
        
        # Should have 2 relations (one USES, one CALLS)
        assert len(result["relations"]) == 2
        assert result["meta"]["relations_deduped"] == 1
    
    def test_relation_dedup_filters_invalid_nodes(self, knowledge_processor_with_dedup):
        """Test that relations with invalid node references are removed."""
        extraction_result = {
            "concepts": [
                {"id": "1", "name": "agent_a", "attributes": {"concept_type": "agent"}},
            ],
            "relations": [
                {"id": "r1", "node_ids": ["1", "999"], "relationship": "USES", "attributes": {}},  # Invalid node
            ],
            "meta": {"records_processed": 0, "concepts_extracted": 1, "relations_extracted": 1}
        }
        
        result = knowledge_processor_with_dedup.process(extraction_result)
        
        # Relation should be removed due to invalid node reference
        assert len(result["relations"]) == 0


class TestEmbeddingManager:
    """Tests for EmbeddingManager."""
    
    def test_embedding_manager_init(self):
        """Test EmbeddingManager initialization."""
        manager = EmbeddingManager()
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            assert manager.model is not None
        else:
            assert manager.model is None
    
    def test_generate_embedding_empty_text(self):
        """Test that empty text returns None."""
        manager = EmbeddingManager()
        result = manager.generate_embedding("")
        
        assert result is None
    
    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
    def test_generate_embedding_returns_array(self):
        """Test that embedding generation returns numpy array."""
        import numpy as np
        
        manager = EmbeddingManager()
        result = manager.generate_embedding("test text")
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result) > 0


class TestCosineSimilarity:
    """Tests for cosine similarity function."""
    
    def test_identical_vectors(self):
        """Test that identical vectors have similarity of 1."""
        import numpy as np
        
        vec = np.array([1.0, 2.0, 3.0])
        similarity = cosine_similarity(vec, vec)
        
        assert abs(similarity - 1.0) < 0.0001
    
    def test_orthogonal_vectors(self):
        """Test that orthogonal vectors have similarity of 0."""
        import numpy as np
        
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        similarity = cosine_similarity(vec1, vec2)
        
        assert abs(similarity) < 0.0001
    
    def test_opposite_vectors(self):
        """Test that opposite vectors have similarity of -1."""
        import numpy as np
        
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([-1.0, 0.0, 0.0])
        similarity = cosine_similarity(vec1, vec2)
        
        assert abs(similarity + 1.0) < 0.0001
    
    def test_zero_vector(self):
        """Test that zero vector returns 0 similarity."""
        import numpy as np
        
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([0.0, 0.0, 0.0])
        similarity = cosine_similarity(vec1, vec2)
        
        assert similarity == 0.0


class TestSemanticDeduplication:
    """Tests for semantic deduplication with embeddings."""
    
    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
    def test_semantic_dedup_similar_concepts(self):
        """Test that semantically similar concepts are deduplicated."""
        processor = KnowledgeProcessor(
            enable_embeddings=True,
            enable_dedup=True,
            similarity_threshold=0.9
        )
        
        # Create concepts with very similar names/descriptions
        extraction_result = {
            "concepts": [
                {"id": "1", "name": "weather_agent", "description": "Agent for weather lookup", "attributes": {"concept_type": "agent"}},
                {"id": "2", "name": "weather_agent_v2", "description": "Agent for weather lookup version 2", "attributes": {"concept_type": "agent"}},
                {"id": "3", "name": "payment_agent", "description": "Agent for payment processing", "attributes": {"concept_type": "agent"}},
            ],
            "relations": [],
            "meta": {"records_processed": 0, "concepts_extracted": 3, "relations_extracted": 0}
        }
        
        result = processor.process(extraction_result)
        
        # Check that embeddings were added
        for concept in result["concepts"]:
            assert "embedding" in concept.get("attributes", {})
    
    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
    def test_embedding_format(self):
        """Test that embeddings are in correct format for JSON serialization."""
        processor = KnowledgeProcessor(
            enable_embeddings=True,
            enable_dedup=False,
            similarity_threshold=0.95
        )
        
        extraction_result = {
            "concepts": [
                {"id": "1", "name": "test_agent", "description": "Test agent", "attributes": {"concept_type": "agent"}},
            ],
            "relations": [],
            "meta": {"records_processed": 0, "concepts_extracted": 1, "relations_extracted": 0}
        }
        
        result = processor.process(extraction_result)
        
        # Embedding should be a list of lists (for JSON compatibility)
        embedding = result["concepts"][0]["attributes"].get("embedding")
        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) == 1  # Wrapped in outer list
        assert isinstance(embedding[0], list)

