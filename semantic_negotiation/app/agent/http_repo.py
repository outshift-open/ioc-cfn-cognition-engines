# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
HTTP helpers for the cognition-fabric **shared-memories query** route.

Callers build the ``intent`` string (e.g. in :mod:`app.agent.options_generation` or
:mod:`app.agent.intent_discovery`) and invoke :func:`post_shared_memories_query` once
per request. No multi-issue loop lives here.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional
from urllib.parse import quote

import httpx


class SharedMemoryNotFoundError(Exception):
    """Raised when the fabric shared-memories query returns 404 (no cache for MAS)."""


def issue_labels_from_negotiable_entities(negotiable_entities: Optional[list[Any]]) -> list[str]:
    """Normalize negotiable entities to issue strings (``term`` attribute when present)."""
    if not negotiable_entities:
        return []
    if isinstance(negotiable_entities, (list, tuple, set)):
        out: list[str] = []
        for x in negotiable_entities:
            label = getattr(x, "term", x)
            s = str(label).strip()
            if s:
                out.append(s)
        return out
    s = str(negotiable_entities).strip()
    return [s] if s else []


def shared_memories_query_path(workspace_id: str, mas_id: str) -> str:
    """URL path (no origin) for POST shared-memories query."""
    wid = quote(workspace_id.strip(), safe="")
    mid = quote(mas_id.strip(), safe="")
    return f"/api/workspaces/{wid}/multi-agentic-systems/{mid}/shared-memories/query"


def post_shared_memories_query(
    client: httpx.Client,
    path: str,
    intent: str,
) -> dict[str, Any]:
    """
    Perform exactly one POST to *path* with the given ``intent``.

    Returns the parsed JSON body (typically ``message``, ``response_id``).
    Raises :exc:`SharedMemoryNotFoundError` on HTTP 404.
    """
    payload: dict[str, Any] = {
        "intent": intent,
        "request_id": str(uuid.uuid4()),
    }
    r = client.post(path, json=payload)
    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            raise SharedMemoryNotFoundError(
                "No shared memory for this workspace/mas (HTTP 404)"
            ) from exc
        raise
    return r.json()
