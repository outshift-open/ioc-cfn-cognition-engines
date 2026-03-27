# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Manager for multiple isolated caching layers."""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from .caching_layer import CachingLayer


class CachingLayerManager:
    """Manages multiple isolated FAISS-backed caches by unique identifiers.

    Each cache maintains its own FAISS index and storage, ensuring complete
    isolation between different caching contexts (e.g., per-user, per-session).
    """

    def __init__(self) -> None:
        self._caches: Dict[str, CachingLayer] = {}

    def create_cache(
        self,
        cache_id: str,
        vector_dimension: int = 1536,
        metric: str = "l2",
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
    ) -> CachingLayer:
        """Create and register a new isolated cache.

        Args:
            cache_id: Unique identifier for this cache.
            vector_dimension: Dimensionality of vectors (default: 1536).
            metric: Distance metric ("l2" or "ip").
            embed_fn: Optional custom embedding function.

        Returns:
            The newly created CachingLayer instance.

        Raises:
            ValueError: If cache_id already exists.
        """
        if cache_id in self._caches:
            raise ValueError(f"Cache with ID '{cache_id}' already exists")

        cache = CachingLayer(vector_dimension, metric, embed_fn)
        self._caches[cache_id] = cache
        return cache

    def get_cache(self, cache_id: str) -> Optional[CachingLayer]:
        """Retrieve a cache by its ID.

        Args:
            cache_id: The unique identifier of the cache.

        Returns:
            The CachingLayer instance, or None if not found.
        """
        return self._caches.get(cache_id)

    def remove_cache(self, cache_id: str) -> bool:
        """Remove a cache by its ID.

        Args:
            cache_id: The unique identifier of the cache to remove.

        Returns:
            True if the cache was removed, False if it didn't exist.
        """
        return self._caches.pop(cache_id, None) is not None

    def list_cache_ids(self) -> list[str]:
        """List all registered cache IDs.

        Returns:
            A list of all cache IDs currently managed.
        """
        return list(self._caches.keys())

    def cache_exists(self, cache_id: str) -> bool:
        """Check if a cache with the given ID exists.

        Args:
            cache_id: The unique identifier to check.

        Returns:
            True if the cache exists, False otherwise.
        """
        return cache_id in self._caches
