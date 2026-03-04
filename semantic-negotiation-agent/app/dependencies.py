"""
Dependency injection configuration for the Semantic Negotiation Agent.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from .agent.negotiation_model import NegotiationModel
from .agent.semantic_negotiation import SemanticNegotiationPipeline
from .config.settings import settings

logger = logging.getLogger(__name__)


@lru_cache()
def get_negotiation_model() -> NegotiationModel:
    """Return a singleton :class:`NegotiationModel` using the configured step budget."""
    logger.info(
        "Initialising NegotiationModel (n_steps=%d)", settings.negotiation_n_steps
    )
    return NegotiationModel(n_steps=settings.negotiation_n_steps)


@lru_cache()
def get_pipeline() -> SemanticNegotiationPipeline:
    """Return a singleton :class:`SemanticNegotiationPipeline`.

    The pipeline is the single entry point for all negotiation requests.  It
    wires together the three components in order:

    1. :class:`~app.agent.intent_discovery.IntentDiscovery`
    2. :class:`~app.agent.options_generation.OptionsGeneration`
    3. :class:`~app.agent.negotiation_model.NegotiationModel`

    When issues and options are pre-supplied by the caller (via the SSTP
    payload) components 1 and 2 are skipped automatically.
    """
    logger.info(
        "Initialising SemanticNegotiationPipeline (n_steps=%d)",
        settings.negotiation_n_steps,
    )
    return SemanticNegotiationPipeline(n_steps=settings.negotiation_n_steps)
