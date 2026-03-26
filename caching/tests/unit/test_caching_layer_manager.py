# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the CachingLayerManager."""
import numpy as np
import pytest

from caching.app.agent.caching_layer_manager import CachingLayerManager


class TestCachingLayerManager:
    def test_create_cache(self):
        manager = CachingLayerManager()
        cache = manager.create_cache("cache1", vector_dimension=8)

        assert cache is not None
        assert cache.vector_dimension == 8
        assert manager.cache_exists("cache1")

    def test_create_duplicate_cache_raises(self):
        manager = CachingLayerManager()
        manager.create_cache("cache1")

        with pytest.raises(ValueError, match="already exists"):
            manager.create_cache("cache1")

    def test_get_cache(self):
        manager = CachingLayerManager()
        created = manager.create_cache("cache1")
        retrieved = manager.get_cache("cache1")

        assert created is retrieved

    def test_get_nonexistent_cache_returns_none(self):
        manager = CachingLayerManager()
        assert manager.get_cache("nonexistent") is None

    def test_remove_cache(self):
        manager = CachingLayerManager()
        manager.create_cache("cache1")

        assert manager.remove_cache("cache1") is True
        assert manager.cache_exists("cache1") is False

    def test_remove_nonexistent_cache_returns_false(self):
        manager = CachingLayerManager()
        assert manager.remove_cache("nonexistent") is False

    def test_list_cache_ids(self):
        manager = CachingLayerManager()
        manager.create_cache("cache1")
        manager.create_cache("cache2")
        manager.create_cache("cache3")

        cache_ids = manager.list_cache_ids()
        assert set(cache_ids) == {"cache1", "cache2", "cache3"}

    def test_cache_exists(self):
        manager = CachingLayerManager()
        manager.create_cache("cache1")

        assert manager.cache_exists("cache1") is True
        assert manager.cache_exists("cache2") is False

    def test_complete_isolation_between_caches(self):
        """Verify that operations on one cache don't affect another."""
        manager = CachingLayerManager()

        # Create two caches with same dimension
        cache1 = manager.create_cache("user_123", vector_dimension=4)
        cache2 = manager.create_cache("session_456", vector_dimension=4)

        # Store different data in each cache
        vec1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        result1 = cache1.store_knowledge(text="user data", vector=vec1)
        result2 = cache1.store_knowledge(text="more user data", vector=vec2)

        result3 = cache2.store_knowledge(text="session data", vector=vec1)

        # Verify IDs are isolated (both start from 0)
        assert result1["id"] == 0
        assert result2["id"] == 1
        assert result3["id"] == 0  # cache2 has its own ID counter

        # Verify counts are isolated
        assert cache1.describe()["ntotal"] == 2
        assert cache2.describe()["ntotal"] == 1

        # Verify searches are isolated
        results1 = cache1.search_similar(vector=vec1, k=5)
        results2 = cache2.search_similar(vector=vec1, k=5)

        assert len(results1) == 2
        assert len(results2) == 1
        assert results1[0]["text"] == "user data"
        assert results2[0]["text"] == "session data"

    def test_isolation_with_metadata(self):
        """Verify metadata isolation between caches."""
        manager = CachingLayerManager()

        cache1 = manager.create_cache("cache1", vector_dimension=3)
        cache2 = manager.create_cache("cache2", vector_dimension=3)

        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        cache1.store_knowledge(
            text="concept A",
            vector=vec,
            metadata={"concept_id": "A123", "source": "cache1"}
        )

        cache2.store_knowledge(
            text="concept B",
            vector=vec,
            metadata={"concept_id": "B456", "source": "cache2"}
        )

        results1 = cache1.search_similar(vector=vec, k=1)
        results2 = cache2.search_similar(vector=vec, k=1)

        assert results1[0]["concept_id"] == "A123"
        assert results1[0]["source"] == "cache1"

        assert results2[0]["concept_id"] == "B456"
        assert results2[0]["source"] == "cache2"

    def test_isolation_after_removing_cache(self):
        """Verify that removing one cache doesn't affect others."""
        manager = CachingLayerManager()

        cache1 = manager.create_cache("cache1", vector_dimension=3)
        cache2 = manager.create_cache("cache2", vector_dimension=3)

        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        cache1.store_knowledge(text="data1", vector=vec)
        cache2.store_knowledge(text="data2", vector=vec)

        # Remove cache1
        manager.remove_cache("cache1")

        # cache2 should still work
        assert cache2.describe()["ntotal"] == 1
        results = cache2.search_similar(vector=vec, k=1)
        assert results[0]["text"] == "data2"

    def test_multiple_caches_different_dimensions(self):
        """Verify caches can have different vector dimensions."""
        manager = CachingLayerManager()

        cache1 = manager.create_cache("cache1", vector_dimension=3)
        cache2 = manager.create_cache("cache2", vector_dimension=5)

        vec3 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec5 = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        cache1.store_knowledge(vector=vec3)
        cache2.store_knowledge(vector=vec5)

        assert cache1.vector_dimension == 3
        assert cache2.vector_dimension == 5
        assert cache1.describe()["ntotal"] == 1
        assert cache2.describe()["ntotal"] == 1
