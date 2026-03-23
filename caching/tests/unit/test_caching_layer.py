# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the CachingLayer FAISS helper."""
import numpy as np
import pytest

from caching.app.agent.caching_layer import CachingLayer


class TestCachingLayer:
    def test_creates_faiss_index(self):
        layer = CachingLayer(vector_dimension=8)
        assert layer.index.d == 8
        assert layer.describe()["metric"] == "l2"

    def test_invalid_dimension_raises(self):
        with pytest.raises(ValueError):
            CachingLayer(vector_dimension=0)

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError):
            CachingLayer(vector_dimension=4, metric="cosine")

    def test_store_knowledge_adds_vectors(self):
        layer = CachingLayer(vector_dimension=6)
        result = layer.store_knowledge("hello faiss")

        assert result["id"] == 0
        assert layer.describe()["ntotal"] == 1

    def test_custom_embed_function_must_match_dimension(self):
        def bad_embed(_: str) -> np.ndarray:
            return np.array([1.0, 2.0], dtype=np.float32)

        layer = CachingLayer(vector_dimension=4, embed_fn=bad_embed)

        with pytest.raises(ValueError):
            layer.store_knowledge("text")

    def test_store_with_vector_only(self):
        layer = CachingLayer(vector_dimension=3)
        manual_vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        result = layer.store_knowledge(vector=manual_vec)

        assert result["id"] == 0
        assert layer.describe()["ntotal"] == 1

    def test_store_without_payload_raises(self):
        layer = CachingLayer(vector_dimension=3)

        with pytest.raises(ValueError):
            layer.store_knowledge()

    def test_search_similar_returns_expected_match(self):
        layer = CachingLayer(vector_dimension=3)
        layer.store_knowledge(vector=np.array([1.0, 0.0, 0.0], dtype=np.float32))
        layer.store_knowledge(vector=np.array([0.0, 1.0, 0.0], dtype=np.float32))

        query = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        results = layer.search_similar(vector=query, k=1)

        assert len(results) == 1
        assert results[0]["id"] == 0

    def test_search_requires_input(self):
        layer = CachingLayer(vector_dimension=3)
        layer.store_knowledge("payload")

        with pytest.raises(ValueError):
            layer.search_similar(k=1)

    def test_search_empty_index_returns_empty(self):
        layer = CachingLayer(vector_dimension=3)

        assert layer.search_similar(text="query", k=2) == []

    def test_metadata_round_trips_through_store_and_search(self):
        layer = CachingLayer(vector_dimension=3)
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        layer.store_knowledge(
            text="agent_x | handles routing",
            vector=vec,
            metadata={"concept_id": "abc123"},
        )

        results = layer.search_similar(vector=vec, k=1)

        assert len(results) == 1
        assert results[0]["text"] == "agent_x | handles routing"
        assert results[0]["concept_id"] == "abc123"

    def test_search_without_metadata_returns_empty_text(self):
        layer = CachingLayer(vector_dimension=3)
        vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        layer.store_knowledge(vector=vec)

        results = layer.search_similar(vector=vec, k=1)

        assert len(results) == 1
        assert results[0]["text"] == ""
        assert "concept_id" not in results[0]

    def test_multiple_metadata_fields_returned(self):
        layer = CachingLayer(vector_dimension=3)
        vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        layer.store_knowledge(
            text="my concept",
            vector=vec,
            metadata={"concept_id": "def456", "concept_type": "agent"},
        )

        results = layer.search_similar(vector=vec, k=1)

        assert results[0]["concept_id"] == "def456"
        assert results[0]["concept_type"] == "agent"
        assert results[0]["text"] == "my concept"
