"""Options generation — component 2 of the semantic negotiation pipeline.

For each issue identified by component 1 (intent discovery), this component
generates candidate options that the negotiating agents could agree upon.
It has access to the original context as well as each agent and its memory.

Current implementation: stub returning two hardcoded options per issue.
Replace :meth:`OptionsGeneration.generate` with a real LLM/NLP-backed
implementation when ready.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Hardcoded fallback options used for any issue not in the predefined map.
_STUB_OPTIONS: Dict[str, List[str]] = {
    "budget":   ["minimal", "low", "medium", "high", "uncapped"],
    "timeline": ["express", "short", "standard", "extended", "long"],
    "scope":    ["core", "standard", "extended", "full"],
    "quality":  ["basic", "standard", "premium"],
}
_DEFAULT_OPTIONS: List[str] = ["option_a", "option_b", "option_c"]


class OptionsGeneration:
    """Generates candidate options per issue for all participating agents.

    Args:
        context: The shared interaction context (same object passed to
            :class:`~app.agent.intent_discovery.IntentDiscovery`).
        agents: Descriptions of the negotiating agents.
        memories: Per-agent memory objects keyed by agent id.
    """

    def __init__(
        self,
        context: Any = None,
        agents: List[Any] | None = None,
        memories: Dict[str, Any] | None = None,
    ) -> None:
        self.context = context
        self.agents = agents or []
        self.memories = memories or {}

    def generate(self, issues: List[str]) -> Dict[str, List[str]]:
        """Produce a list of candidate options for each supplied issue.

        .. note::
            **Stub implementation** — returns two hardcoded options per issue
            using a predefined map (``budget → [low, high]``,
            ``timeline → [short, long]``).  Unknown issues fall back to
            ``[option_a, option_b]``.
            Replace this method body with a real LLM/NLP-backed
            implementation when ready.

        Args:
            issues: Ordered list of issue ids returned by component 1.

        Returns:
            A mapping ``{issue_id: [option, ...]}`` covering every issue.
        """
        result = {
            issue: _STUB_OPTIONS.get(issue, _DEFAULT_OPTIONS)[:]
            for issue in issues
        }
        logger.debug("OptionsGeneration.generate() — stub returning %s", result)
        return result
