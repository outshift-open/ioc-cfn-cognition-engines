# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Async top-k retrieval against rag_cache_layer (same search_similar contract as graph cache_layer)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from caching.app.agent import CachingLayer

logger = logging.getLogger(__name__)


def _meta_scalar(raw: Any) -> str:
    """RAG rows are expected to carry timestamp and domain (often ``{\"value\": ...}``)."""
    if raw is None:
        return ""
    if isinstance(raw, dict):
        if raw.get("value") is not None:
            return str(raw["value"]).strip()
        return ""
    return str(raw).strip()


def _attach_display_line(row: Dict[str, Any], index: int) -> None:
    st = _meta_scalar(row.get("timestamp"))
    dm = _meta_scalar(row.get("domain"))
    txt = str(row.get("text") or "").strip()
    parts: List[str] = [p for p in (st, dm) if p]
    parts.append(txt)
    row["display_line"] = f"[{index}] " + ", ".join(parts)


def _normalize_hit(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not row or not isinstance(row, dict):
        return None
    text = row.get("text")
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    out = {k: v for k, v in row.items()}
    out["text"] = text
    try:
        out["score"] = float(row.get("score", 0.0))
    except (TypeError, ValueError):
        out["score"] = 0.0
    return out


async def retrieve_rag_top_k(
    rag_layer: CachingLayer,
    intent: str,
    top_k: int,
    timeout_seconds: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Run vector similarity on rag_layer.search_similar(text=..., k=...) in a thread.
    Each hit includes display_line: ``[n] timestamp, domain, chunk_text``.
    """
    if rag_layer is None or not (intent or "").strip():
        return []

    def _search() -> List[Dict[str, Any]]:
        try:
            raw = rag_layer.search_similar(text=str(intent).strip(), k=top_k)
        except Exception as e:
            logger.warning("[RAG] search_similar failed: %s", e)
            return []
        out: List[Dict[str, Any]] = []
        for i, r in enumerate(raw or [], start=1):
            n = _normalize_hit(r if isinstance(r, dict) else {})
            if n:
                _attach_display_line(n, i)
                out.append(n)
        logger.debug("[RAG] retrieval result: %s", out)
        return out

    try:
        if timeout_seconds is not None and timeout_seconds > 0:
            return await asyncio.wait_for(asyncio.to_thread(_search), timeout=timeout_seconds)
        return await asyncio.to_thread(_search)
    except asyncio.TimeoutError:
        logger.warning("[RAG] retrieval timed out after %ss", timeout_seconds)
        return []
    except Exception as e:
        logger.warning("[RAG] retrieval error: %s", e)
        return []
