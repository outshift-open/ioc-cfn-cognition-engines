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
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.agent.intent_discovery import IntentDiscovery
from app.agent.negotiation_model import (
    NegotiationModel,
    NegotiationOutcome,
    NegotiationParticipant,
    NegotiationResult,
)
from app.agent.options_generation import OptionsGeneration

__all__ = [
    "SemanticNegotiationPipeline",
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
        result = pipeline.run()

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

        # Instantiate the three components
        self._intent_discovery = IntentDiscovery(context=self.context)
        self._options_generation = OptionsGeneration(
            context=self.context,
            agents=self.agents,
            memories=self.memories,
        )
        self._negotiation_model = NegotiationModel(n_steps=self.n_steps, strategy=None)  # reads NEGOTIATOR_STRATEGY from env

    def run(
        self,
        content_text: Optional[str] = None,
        issues: Optional[List[str]] = None,
        options_per_issue: Optional[Dict[str, List[str]]] = None,
        participants: Optional[List[NegotiationParticipant]] = None,
    ) -> NegotiationResult:
        """Execute the full pipeline end-to-end.

        Components 1 and 2 are only invoked when their outputs are not
        pre-supplied by the caller.  Pass *content_text* to drive
        component 1 (IntentDiscovery); pass *issues* to bypass it.

        Args:
            content_text: Natural-language mission description forwarded to
                component 1.  Ignored when *issues* is provided.
            issues: Pre-supplied ordered list of issue identifiers.  If
                ``None`` component 1 is invoked with *content_text*.
            options_per_issue: Pre-supplied ``{issue_id: [option, ...]}``
                mapping.  If ``None`` component 2 is invoked.
            participants: Negotiation participants for component 3.  Falls
                back to ``self.participants`` when omitted.

        Returns:
            A :class:`~app.agent.negotiation_model.NegotiationResult`.
        """
        resolved_participants = participants if participants is not None else self.participants

        # Component 1 — intent discovery (skipped when caller supplies issues)
        if issues is None:
            issues = self._intent_discovery.discover(content_text=content_text)

        # Component 2 — options generation (skipped when caller supplies options)
        if options_per_issue is None:
            options_per_issue = self._options_generation.generate(issues)

        # Component 3 — negotiation model
        return self._negotiation_model.run(
            issues=issues,
            options_per_issue=options_per_issue,
            participants=resolved_participants,
        )