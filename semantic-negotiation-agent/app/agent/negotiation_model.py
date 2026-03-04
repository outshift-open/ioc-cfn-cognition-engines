"""Negotiation model — component 3 of the semantic negotiation pipeline.

This module implements the negotiation runner that takes issues (from component 1 –
intent discovery) and options per issue (from component 2 – options generation) and
runs a multi-issue bilateral or multilateral negotiation using the NegMAS SAO
(Stacked Alternating Offers) mechanism.

Components 1 and 2 are *not* implemented here; they are expected to produce the
inputs consumed by this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import importlib

from negmas import SAOMechanism, make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as UFun
import negmas.sao.negotiators as _sao_negotiators

_STRATEGY_MODULES = [
    "negmas.sao.negotiators",
    "negmas.sao.negotiators.controlled",
    "negmas.sao.negotiators.limited",
    "negmas.sao.negotiators.timebased",
]


def _resolve_strategy(name: str) -> type:
    """Resolve a NegMAS negotiator class by name.

    Searches ``negmas.sao.negotiators`` and its common sub-modules.
    Raises ``ValueError`` with available names when not found.
    """
    # Fast path — class is re-exported at the top-level package
    cls = getattr(_sao_negotiators, name, None)
    if cls is not None:
        return cls
    # Slow path — try sub-modules
    for mod_path in _STRATEGY_MODULES:
        try:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, name, None)
            if cls is not None:
                return cls
        except ImportError:
            continue
    available = sorted(
        n for n in dir(_sao_negotiators)
        if "Negotiator" in n or "Agent" in n
    )
    raise ValueError(
        f"Unknown negotiator strategy '{name}'. "
        f"Available: {available}"
    )


@dataclass
class NegotiationParticipant:
    """Represents one participant (agent) in a negotiation session.

    Attributes:
        id: Unique identifier for the participant.
        name: Human-readable name.
        preferences: Per-issue option utilities.
            Shape: ``{issue_id: {option_label: utility_value}}``.
            Utilities should be in ``[0.0, 1.0]`` for best results.
            Ignored when ``callback_url`` is set — the external agent owns its
            own utility function privately.
        issue_weights: Optional per-issue importance weights for the linear additive
            utility function. Defaults to equal weights when omitted.
            Ignored when ``callback_url`` is set.
        callback_url: Optional HTTP endpoint of the external agent that will
            make all propose/respond decisions for this participant.  When set,
            the server-side NegMAS negotiator strategy is bypassed and an
            :class:`~app.agent.callback_negotiator.SSTPCallbackNegotiator` is
            used instead, delegating every decision via ``SSTPNegotiateMessage``.
    """

    id: str
    name: str
    preferences: Dict[str, Dict[str, float]] = field(default_factory=dict)
    issue_weights: Optional[Dict[str, float]] = None
    callback_url: Optional[str] = None


@dataclass
class NegotiationOutcome:
    """Agreed value for a single issue."""

    issue_id: str
    chosen_option: str


@dataclass
class NegotiationResult:
    """Full result returned by :class:`NegotiationModel.run`.

    Attributes:
        agreement: Agreed outcomes per issue, or ``None`` if no agreement was reached.
        timedout: Whether the negotiation exhausted the step budget without agreement.
        broken: Whether a participant explicitly ended the negotiation.
        steps: Number of SAO rounds executed.
        history: Raw NegMAS extended trace as ``(step, negotiator_id, offer)`` tuples.
        raw_state: The final ``SAOState`` object for advanced inspection.
    """

    agreement: Optional[List[NegotiationOutcome]]
    timedout: bool
    broken: bool
    steps: int
    history: List[Tuple[int, str, Any]] = field(default_factory=list)
    raw_state: Any = field(default=None, repr=False)


class NegotiationModel:
    """Runs a multi-issue SAO negotiation via NegMAS.

    This is component 3 of the semantic negotiation pipeline. It consumes:

    * A list of issue identifiers produced by component 1 (intent discovery).
    * A mapping of options per issue produced by component 2 (options generation).
    * A list of :class:`NegotiationParticipant` objects describing each party.

    Internally it builds NegMAS ``Issue`` objects, constructs a
    :class:`~negmas.SAOMechanism`, attaches one negotiator per participant
    (class controlled by *strategy*) with a normalised
    ``LinearAdditiveUtilityFunction`` derived from that participant's stated
    preferences, then runs the mechanism and packages the outcome into a
    :class:`NegotiationResult`.

    Args:
        n_steps: Maximum number of SAO rounds before the negotiation times out.
        strategy: NegMAS negotiator class name (e.g. ``"ConcederTBNegotiator"``)
            or a class object directly.  Defaults to ``"BoulwareTBNegotiator"``
            unless overridden by the ``NEGOTIATOR_STRATEGY`` env variable.
    """

    def __init__(self, n_steps: int = 100, strategy: "str | type | None" = None) -> None:
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        self.n_steps = n_steps

        if strategy is None:
            from app.config.settings import settings
            strategy = settings.negotiator_strategy

        if isinstance(strategy, str):
            self._negotiator_cls = _resolve_strategy(strategy)
        else:
            self._negotiator_cls = strategy

        import logging
        logging.getLogger(__name__).info(
            "NegotiationModel — strategy: %s", self._negotiator_cls.__name__
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        issues: List[str],
        options_per_issue: Dict[str, List[str]],
        participants: List[NegotiationParticipant],
        session_id: str = "unknown",
    ) -> NegotiationResult:
        """Run a full SAO negotiation session and return the result.

        Args:
            issues: Ordered list of issue identifiers.
            options_per_issue: Mapping from each issue id to its candidate options.
            participants: The negotiating parties.  At least two are required.
            session_id: Negotiation session identifier threaded through to any
                :class:`~app.agent.callback_negotiator.SSTPCallbackNegotiator`
                instances so outgoing SSTP messages carry the correct session.

        Returns:
            A :class:`NegotiationResult` describing what was (or was not) agreed.

        Raises:
            ValueError: If fewer than two participants are supplied, if any issue
                is missing from ``options_per_issue``, or if an options list is empty.
        """
        self._validate_inputs(issues, options_per_issue, participants)

        negmas_issues = self._build_issues(issues, options_per_issue)
        mechanism = SAOMechanism(issues=negmas_issues, n_steps=self.n_steps)

        for participant in participants:
            if participant.callback_url:
                # Delegate all decisions to the external agent via SSTP callbacks.
                # No server-side utility function or strategy is needed.
                from app.agent.callback_negotiator import SSTPCallbackNegotiator
                negotiator = SSTPCallbackNegotiator(
                    name=participant.name,
                    callback_url=participant.callback_url,
                    participant_id=participant.id,
                    session_id=session_id,
                )
                mechanism.add(negotiator)
            else:
                ufun = self._build_ufun(participant, issues, options_per_issue, mechanism)
                mechanism.add(self._negotiator_cls(name=participant.name), ufun=ufun)

        state = mechanism.run()
        return self._build_result(state, issues, mechanism)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_inputs(
        self,
        issues: List[str],
        options_per_issue: Dict[str, List[str]],
        participants: List[NegotiationParticipant],
    ) -> None:
        if len(participants) < 2:
            raise ValueError("At least two participants are required for a negotiation")
        for issue_id in issues:
            if issue_id not in options_per_issue:
                raise ValueError(f"Issue '{issue_id}' has no entry in options_per_issue")
            if not options_per_issue[issue_id]:
                raise ValueError(f"Issue '{issue_id}' has an empty options list")

    def _build_issues(self, issues: List[str], options_per_issue: Dict[str, List[str]]):
        """Convert issue ids and option lists into NegMAS Issue objects."""
        return [make_issue(values=options_per_issue[issue_id], name=issue_id) for issue_id in issues]

    def _build_ufun(
        self,
        participant: NegotiationParticipant,
        issues: List[str],
        options_per_issue: Dict[str, List[str]],
        mechanism: SAOMechanism,
    ) -> UFun:
        """Build a normalised LinearAdditiveUtilityFunction for one participant.

        Options not listed in the participant's preferences receive a utility of
        ``0.0``. Issue weights default to uniform when ``issue_weights`` is None.
        """
        values: Dict[str, Dict[str, float]] = {}
        for issue_id in issues:
            issue_prefs = participant.preferences.get(issue_id, {})
            values[issue_id] = {opt: float(issue_prefs.get(opt, 0.0)) for opt in options_per_issue[issue_id]}

        n_issues = len(issues)
        weights: Dict[str, float] = {
            issue_id: (
                float(participant.issue_weights[issue_id])
                if participant.issue_weights and issue_id in participant.issue_weights
                else 1.0 / n_issues
            )
            for issue_id in issues
        }

        return UFun(values=values, weights=weights, outcome_space=mechanism.outcome_space).normalize()

    def _build_result(
        self,
        state: Any,
        issues: List[str],
        mechanism: SAOMechanism,
    ) -> NegotiationResult:
        """Translate a raw SAOState into a NegotiationResult."""
        agreement: Optional[List[NegotiationOutcome]] = None
        if state.agreement is not None:
            agreement = [
                NegotiationOutcome(issue_id=issue_id, chosen_option=str(state.agreement[idx]))
                for idx, issue_id in enumerate(issues)
            ]

        return NegotiationResult(
            agreement=agreement,
            timedout=state.timedout,
            broken=state.broken,
            steps=state.step,
            history=list(mechanism.extended_trace),
            raw_state=state,
        )
