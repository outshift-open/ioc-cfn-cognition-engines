# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Semantic negotiation pipeline — top-level orchestrator.

This module is the single entry point for the full semantic negotiation flow.
It wires together the three components in order:

1. :class:`~app.agent.intent_discovery.IntentDiscovery`
   Takes a shared context and returns the list of negotiable issues.

2. :class:`~app.agent.options_generation.OptionsGeneration`
   Takes each issue, the context, the participating agents and their memories,
   and produces a list of candidate options per issue.

3. :class:`~app.agent.negotiation_model.NegotiationModel`
   Takes the issues, options, and participant preferences and runs a NegMAS
   SAO negotiation, returning a :class:`~app.agent.negotiation_model.NegotiationResult`.

Callers that need the resolved issue space for envelopes (e.g. SSTP
``semantic_context``) should use :meth:`SemanticNegotiationPipeline.run`, which
returns :class:`SemanticPipelineRun` with ``issues`` and ``options_per_issue``.
HTTP initiate builds :class:`~app.agent.negotiation_model.NegotiationParticipant`
lists only *after* options exist (server-side prefs are derived per option list),
so the API uses ``run(..., after_options=...)`` rather than pre-filling
``self.participants``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from app.agent.intent_discovery import IntentDiscovery
from app.agent.negotiation_model import (
    NegotiationModel,
    NegotiationOutcome,
    NegotiationParticipant,
    NegotiationResult,
)
from app.agent.options_generation import OptionsGeneration

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SemanticPipelineRun:
    """Bundle returned by :meth:`SemanticNegotiationPipeline.run`.

    Holds the negotiation outcome plus the exact issue list and option map used,
    so HTTP or tracing layers can populate SSTP ``semantic_context`` without
    re-invoking intent discovery or options generation.
    """

    result: NegotiationResult
    issues: List[str]
    options_per_issue: Dict[str, List[str]]
    participants: List[NegotiationParticipant]


__all__ = [
    "SemanticNegotiationPipeline",
    "SemanticPipelineRun",
    # Re-exported for convenience
    "NegotiationModel",
    "NegotiationParticipant",
    "NegotiationOutcome",
    "NegotiationResult",
    "IntentDiscovery",
    "OptionsGeneration",
]


class SemanticNegotiationPipeline:
    """Orchestrates the full three-component semantic negotiation flow.

    Usage (once all components are implemented)::

        pipeline = SemanticNegotiationPipeline(
            context=shared_context,
            agents=agent_list,
            memories=agent_memories,
            participants=negotiation_participants,
        )
        pr = pipeline.run(content_text="…", session_id="sess-1")
        result = pr.result

    Args:
        context: Shared interaction context forwarded to components 1 and 2.
        agents: Agent descriptors forwarded to component 2.
        memories: Per-agent memory objects forwarded to component 2.
        participants: Negotiation participants forwarded to component 3.
        n_steps: Maximum SAO rounds for the negotiation (component 3).
    """

    def __init__(
        self,
        context: Any = None,
        agents: Optional[List[Any]] = None,
        memories: Optional[Dict[str, Any]] = None,
        participants: Optional[List[NegotiationParticipant]] = None,
        n_steps: int = 100,
    ) -> None:
        self.context = context
        self.agents = agents or []
        self.memories = memories or {}
        self.participants = participants or []
        self.n_steps = n_steps

        # Instantiate the three components (LLM comes from get_llm_provider() unless overridden)
        self._intent_discovery = IntentDiscovery()
        self._options_generation = OptionsGeneration()
        self._negotiation_model = NegotiationModel(n_steps=self.n_steps, strategy=None)  # reads NEGOTIATOR_STRATEGY from env

    def run(
        self,
        content_text: Optional[str] = None,
        issues: Optional[List[str]] = None,
        options_per_issue: Optional[Dict[str, List[str]]] = None,
        participants: Optional[List[NegotiationParticipant]] = None,
        session_id: str = "unknown",
        *,
        after_options: Optional[
            Callable[[List[str], Dict[str, List[str]]], List[NegotiationParticipant]]
        ] = None,
    ) -> SemanticPipelineRun:
        """Execute the full pipeline end-to-end.

        Components 1 and 2 run only when *issues* / *options_per_issue* are
        omitted.  Component 3 always runs.

        Args:
            content_text: Mission text for intent discovery and options generation.
            issues: When set, skips component 1.
            options_per_issue: When set, skips component 2.
            participants: When set, used as negotiators (``after_options`` ignored).
            session_id: Passed to :meth:`~app.agent.negotiation_model.NegotiationModel.run`
                (callbacks and session locking).
            after_options: If ``participants`` is ``None``, called as
                ``after_options(issues, options_per_issue)`` to build negotiators
                after the option space exists (e.g. HTTP layer with mixed callback /
                server-side strategies). Otherwise falls back to ``self.participants``.

        Returns:
            :class:`SemanticPipelineRun` with the negotiation result and the
            resolved issue space / participant list.
        """
        # Component 1 — intent discovery (skipped when caller supplies issues)
        if issues is None:
            logger.info(
                "[%s] Component 1 — IntentDiscovery (content_text=%r)",
                session_id,
                (content_text or "")[:80],
            )
            idr = self._intent_discovery.discover(sentence=content_text or "")
            issues = [
                x.strip()
                for x in idr.negotiable_entities
                if isinstance(x, str) and x.strip()
            ]
            if not issues:
                raise ValueError(
                    "Intent discovery returned no negotiable issues. "
                    "Typical causes: LLM errors, blocked/empty API key, or model output that is not "
                    "JSON with `negotiable_entities` (either `[{\"term\": ...}]` or a list of strings). "
                    "See negotiation server logs for Component 1."
                )
            logger.info("[%s] Discovered issues: %s", session_id, issues)

        # Component 2 — options generation (skipped when caller supplies options)
        if options_per_issue is None:
            logger.info("[%s] Component 2 — OptionsGeneration", session_id)
            options_per_issue = self._options_generation.generate_options(
                issues,
                content_text or "",
                None,
            )
            logger.info("[%s] Generated options: %s", session_id, options_per_issue)

        # Resolve negotiators: explicit list wins; else optional factory (HTTP
        # initiate); else defaults from __init__.
        if participants is not None:
            resolved = participants
        elif after_options is not None:
            resolved = after_options(issues, options_per_issue)
        else:
            resolved = self.participants

        logger.info(
            "[%s] Component 3 — NegotiationModel (%d agents, n_steps=%s)",
            session_id,
            len(resolved),
            self.n_steps,
        )
        # session_id threads into callback SSTP messages and prevents concurrent
        # runs with the same id from colliding on in-process decision storage.
        result = self._negotiation_model.run(
            issues=issues,
            options_per_issue=options_per_issue,
            participants=resolved,
            session_id=session_id,
        )
        return SemanticPipelineRun(
            result=result,
            issues=issues,
            options_per_issue=options_per_issue,
            participants=resolved,
        )