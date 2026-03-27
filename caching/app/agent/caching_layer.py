# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""High-level caching layer backed by a FAISS index."""
from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, List, Optional

import faiss
import numpy as np


class CachingLayer:
    """Thin wrapper that owns a FAISS index for future vector cache use."""

    def __init__(
        self,
        vector_dimension: int = 1536,
        metric: str = "l2",
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
    ) -> None:
        if vector_dimension <= 0:
            raise ValueError("vector_dimension must be positive")

        self.vector_dimension = vector_dimension
        self.metric = metric.lower()
        self.index = self._initialize_index()
        self.embed_fn = embed_fn or self._default_embed
        self._payload_store: Dict[int, Dict[str, Any]] = {}
        self._next_id = 0

    def _initialize_index(self) -> faiss.Index:
        if self.metric == "l2":
            return faiss.IndexFlatL2(self.vector_dimension)
        if self.metric in {"ip", "inner_product"}:
            return faiss.IndexFlatIP(self.vector_dimension)
        raise ValueError(f"Unsupported FAISS metric: {self.metric}")

    def describe(self) -> Dict[str, Any]:
        """Return metadata about the managed FAISS index."""
        return {
            "dimension": self.vector_dimension,
            "metric": self.metric,
            "ntotal": int(self.index.ntotal),
        }

    def store_knowledge(
        self,
        text: Optional[str] = None,
        vector: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Persist either raw text (auto-embedded) or a caller-provided vector.

        ``metadata``, when supplied, is stored alongside the text and returned
        verbatim by :meth:`search_similar`.
        """
        if vector is None:
            if not text:
                raise ValueError("Either text or vector must be provided")
            vector = self._embed_text(text)
        else:
            vector = self._normalize_vector(vector)

        payload: Dict[str, Any] = {"text": text or ""}
        if metadata:
            payload.update(metadata)

        self.index.add(vector)
        entry_id = self._next_id
        self._payload_store[entry_id] = payload
        self._next_id += 1

        return {
            "id": entry_id,
            "ntotal": int(self.index.ntotal),
        }

    def search_similar(
        self,
        *,
        text: Optional[str] = None,
        vector: Optional[np.ndarray] = None,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find the top-k nearest payloads for the provided query."""
        if k <= 0:
            raise ValueError("k must be positive")
        if self.index.ntotal == 0:
            return []

        if vector is None:
            if not text:
                raise ValueError("Either text or vector must be provided")
            vector = self._embed_text(text)
        else:
            vector = self._normalize_vector(vector)

        limit = min(k, self.index.ntotal)
        distances, indices = self.index.search(vector, limit)
        results: List[Dict[str, Any]] = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < 0:
                continue
            stored = self._payload_store.get(int(idx), {})
            entry: Dict[str, Any] = {
                "id": int(idx),
                "score": float(distance),
                "text": stored.get("text", ""),
            }
            entry.update({k: v for k, v in stored.items() if k != "text"})
            results.append(entry)
        return results

    def _embed_text(self, text: str) -> np.ndarray:
        vector = self.embed_fn(text)
        return self._normalize_vector(vector)

    def _normalize_vector(self, vector: Any) -> np.ndarray:
        """Ensure the incoming vector has shape (1, d) and correct dimension."""
        arr = np.asarray(vector, dtype="float32")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError("Vector must be 1D or 2D before normalization")
        if arr.shape[1] != self.vector_dimension:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {self.vector_dimension}, received {arr.shape[1]}"
            )
        return arr

    def _default_embed(self, text: str) -> np.ndarray:
        """Deterministic hash-based embedding to keep dependencies light."""
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big", signed=False)
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal(self.vector_dimension, dtype=np.float32)

        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
