# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Concept Vector Store – In-process FAISS storage using the caching-layer library.

Imports the ``CachingLayer`` class from the sibling caching-layer service
and uses its methods directly (store_knowledge, search_similar, describe)
to persist extracted concepts in an in-memory FAISS index.

Uses ``importlib.util`` to load the module without namespace collisions
(both services have a top-level ``app`` package).
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load CachingLayer: prefer package import (Docker PYTHONPATH=/app), else file path (repo)
# ---------------------------------------------------------------------------

def _load_caching_layer_class():
    """Load CachingLayer from caching package (Docker) or sibling dir (repo)."""
    try:
        from caching.app.agent.caching_layer import CachingLayer as _Cls
        return _Cls
    except ImportError:
        pass
    base = Path(__file__).resolve().parent.parent.parent.parent
    for name in ("caching",):
        path = base / name / "app" / "agent" / "caching_layer.py"
        if path.exists():
            spec = importlib.util.spec_from_file_location(
                "caching_layer_ext", str(path)
            )
            if spec is None or spec.loader is None:
                raise ImportError("Failed to create module spec for CachingLayer")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.CachingLayer
    raise ImportError(
        f"CachingLayer not found: tried package 'caching' and paths under {base!s}"
    )


CachingLayer = _load_caching_layer_class()


class ConceptVectorStore:
    """
    In-process FAISS store for extracted concepts.

    Wraps ``CachingLayer`` directly – no HTTP overhead.  Each concept is
    indexed by its embedding and its textual payload (name | description)
    is stored alongside for retrieval.

    When ``cache_layer`` is provided (unified app), that shared instance
    is used; otherwise a new CachingLayer is created (standalone).

    Methods used from CachingLayer:
        * ``store_knowledge(text, vector)``
        * ``search_similar(text, vector, k)``
        * ``describe()``
    """

    def __init__(
        self,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        vector_dimension: int = 384,
        metric: str = "l2",
        cache_layer: Optional[Any] = None,
    ) -> None:
        if cache_layer is not None:
            self._cache = cache_layer
        else:
            self._cache = CachingLayer(
                vector_dimension=vector_dimension,
                metric=metric,
                embed_fn=embed_fn,
            )

    def store_concepts(self, concepts: List[Dict[str, Any]]) -> None:
        """
        Store every concept that carries an embedding into the FAISS index
        via ``CachingLayer.store_knowledge``.

        Concepts without embeddings are silently skipped.
        """
        stored = 0
        skipped = 0

        for concept in concepts:
            embedding_data = concept.get("attributes", {}).get("embedding")
            if not embedding_data or not embedding_data[0]:
                skipped += 1
                continue

            try:
                vector = np.array(embedding_data[0], dtype=np.float32)
                # Store "name | description" so FAISS has both; cache returns id (int) and text for evidence.
                name = (concept.get("name") or "").strip()
                description = (concept.get("description") or "").strip()
                if not name:
                    skipped += 1
                    continue
                text = f"{name} | {description}" if description else name
                self._cache.store_knowledge(
                    text=text,
                    vector=vector,
                    metadata={"concept_id": concept.get("id", "")},
                )
                stored += 1
            except Exception:
                logger.exception(
                    "Failed to store concept %s in FAISS",
                    concept.get("id"),
                )
                skipped += 1

        logger.info(
            "FAISS store complete: stored=%d, skipped=%d, index_total=%d",
            stored,
            skipped,
            self._cache.describe()["ntotal"],
        )

    def search_similar(
        self,
        text: Optional[str] = None,
        vector: Optional[np.ndarray] = None,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find the *k* most similar concepts via ``CachingLayer.search_similar``."""
        return self._cache.search_similar(text=text, vector=vector, k=k)

    def describe(self) -> Dict[str, Any]:
        """Return FAISS index metadata via ``CachingLayer.describe``."""
        return self._cache.describe()
