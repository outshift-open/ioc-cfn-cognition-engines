# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Manager for multiple isolated caching layers."""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from .caching_layer import CachingLayer


class CachingLayerManager:
    """Manages multiple isolated caching layers by ID.

    Each layer maintains its own FAISS index and cache storage, ensuring
    complete isolation between different caching contexts.
    """

    def __init__(self) -> None:
        self.layers: Dict[str, CachingLayer] = {}

    def create_layer(
        self,
        layer_id: str,
        vector_dimension: int = 1536,
        metric: str = "l2",
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
    ) -> CachingLayer:
        """Create and register a new caching layer.

        Args:
            layer_id: Unique identifier for this caching layer.
            vector_dimension: Dimensionality of vectors (default: 1536).
            metric: Distance metric ("l2" or "ip").
            embed_fn: Optional custom embedding function.

        Returns:
            The newly created CachingLayer instance.

        Raises:
            ValueError: If layer_id already exists.
        """
        if layer_id in self.layers:
            raise ValueError(f"Layer with ID '{layer_id}' already exists")

        layer = CachingLayer(vector_dimension, metric, embed_fn)
        self.layers[layer_id] = layer
        return layer

    def get_layer(self, layer_id: str) -> Optional[CachingLayer]:
        """Retrieve a caching layer by ID.

        Args:
            layer_id: The unique identifier of the layer.

        Returns:
            The CachingLayer instance, or None if not found.
        """
        return self.layers.get(layer_id)

    def remove_layer(self, layer_id: str) -> bool:
        """Remove a caching layer.

        Args:
            layer_id: The unique identifier of the layer to remove.

        Returns:
            True if the layer was removed, False if it didn't exist.
        """
        return self.layers.pop(layer_id, None) is not None

    def list_layers(self) -> list[str]:
        """List all registered layer IDs.

        Returns:
            A list of all layer IDs currently managed.
        """
        return list(self.layers.keys())

    def layer_exists(self, layer_id: str) -> bool:
        """Check if a layer with the given ID exists.

        Args:
            layer_id: The unique identifier to check.

        Returns:
            True if the layer exists, False otherwise.
        """
        return layer_id in self.layers
