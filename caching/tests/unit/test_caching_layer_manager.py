# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the CachingLayerManager."""
import numpy as np
import pytest

from caching.app.agent.caching_layer_manager import CachingLayerManager


class TestCachingLayerManager:
    def test_create_layer(self):
        manager = CachingLayerManager()
        layer = manager.create_layer("layer1", vector_dimension=8)

        assert layer is not None
        assert layer.vector_dimension == 8
        assert manager.layer_exists("layer1")

    def test_create_duplicate_layer_raises(self):
        manager = CachingLayerManager()
        manager.create_layer("layer1")

        with pytest.raises(ValueError, match="already exists"):
            manager.create_layer("layer1")

    def test_get_layer(self):
        manager = CachingLayerManager()
        created = manager.create_layer("layer1")
        retrieved = manager.get_layer("layer1")

        assert created is retrieved

    def test_get_nonexistent_layer_returns_none(self):
        manager = CachingLayerManager()
        assert manager.get_layer("nonexistent") is None

    def test_remove_layer(self):
        manager = CachingLayerManager()
        manager.create_layer("layer1")

        assert manager.remove_layer("layer1") is True
        assert manager.layer_exists("layer1") is False

    def test_remove_nonexistent_layer_returns_false(self):
        manager = CachingLayerManager()
        assert manager.remove_layer("nonexistent") is False

    def test_list_layers(self):
        manager = CachingLayerManager()
        manager.create_layer("layer1")
        manager.create_layer("layer2")
        manager.create_layer("layer3")

        layers = manager.list_layers()
        assert set(layers) == {"layer1", "layer2", "layer3"}

    def test_layer_exists(self):
        manager = CachingLayerManager()
        manager.create_layer("layer1")

        assert manager.layer_exists("layer1") is True
        assert manager.layer_exists("layer2") is False

    def test_complete_isolation_between_layers(self):
        """Verify that operations on one layer don't affect another."""
        manager = CachingLayerManager()

        # Create two layers with same dimension
        layer1 = manager.create_layer("user_123", vector_dimension=4)
        layer2 = manager.create_layer("session_456", vector_dimension=4)

        # Store different data in each layer
        vec1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        result1 = layer1.store_knowledge(text="user data", vector=vec1)
        result2 = layer1.store_knowledge(text="more user data", vector=vec2)

        result3 = layer2.store_knowledge(text="session data", vector=vec1)

        # Verify IDs are isolated (both start from 0)
        assert result1["id"] == 0
        assert result2["id"] == 1
        assert result3["id"] == 0  # Layer2 has its own ID counter

        # Verify counts are isolated
        assert layer1.describe()["ntotal"] == 2
        assert layer2.describe()["ntotal"] == 1

        # Verify searches are isolated
        results1 = layer1.search_similar(vector=vec1, k=5)
        results2 = layer2.search_similar(vector=vec1, k=5)

        assert len(results1) == 2
        assert len(results2) == 1
        assert results1[0]["text"] == "user data"
        assert results2[0]["text"] == "session data"

    def test_isolation_with_metadata(self):
        """Verify metadata isolation between layers."""
        manager = CachingLayerManager()

        layer1 = manager.create_layer("layer1", vector_dimension=3)
        layer2 = manager.create_layer("layer2", vector_dimension=3)

        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        layer1.store_knowledge(
            text="concept A",
            vector=vec,
            metadata={"concept_id": "A123", "source": "layer1"}
        )

        layer2.store_knowledge(
            text="concept B",
            vector=vec,
            metadata={"concept_id": "B456", "source": "layer2"}
        )

        results1 = layer1.search_similar(vector=vec, k=1)
        results2 = layer2.search_similar(vector=vec, k=1)

        assert results1[0]["concept_id"] == "A123"
        assert results1[0]["source"] == "layer1"

        assert results2[0]["concept_id"] == "B456"
        assert results2[0]["source"] == "layer2"

    def test_isolation_after_removing_layer(self):
        """Verify that removing one layer doesn't affect others."""
        manager = CachingLayerManager()

        layer1 = manager.create_layer("layer1", vector_dimension=3)
        layer2 = manager.create_layer("layer2", vector_dimension=3)

        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        layer1.store_knowledge(text="data1", vector=vec)
        layer2.store_knowledge(text="data2", vector=vec)

        # Remove layer1
        manager.remove_layer("layer1")

        # layer2 should still work
        assert layer2.describe()["ntotal"] == 1
        results = layer2.search_similar(vector=vec, k=1)
        assert results[0]["text"] == "data2"

    def test_multiple_layers_different_dimensions(self):
        """Verify layers can have different vector dimensions."""
        manager = CachingLayerManager()

        layer1 = manager.create_layer("layer1", vector_dimension=3)
        layer2 = manager.create_layer("layer2", vector_dimension=5)

        vec3 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec5 = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        layer1.store_knowledge(vector=vec3)
        layer2.store_knowledge(vector=vec5)

        assert layer1.vector_dimension == 3
        assert layer2.vector_dimension == 5
        assert layer1.describe()["ntotal"] == 1
        assert layer2.describe()["ntotal"] == 1
