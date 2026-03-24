# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Semantic negotiation pipeline - top-level orchestrator.

Wires together:

1. :class:`~app.agent.intent_discovery.IntentDiscovery` - extracts negotiable issues.
2. :class:`~app.agent.options_generation.OptionsGeneration` - generates options per issue.
3. :class:`~app.agent.batch_callback_runner.BatchCallbackRunner` - runs the turn-by-turn
   SAO negotiation, driven externally via ``/initiate`` + ``/decide``.

Callers that need the resolved issue space for envelopes (e.g. SSTP
``semantic_context``) should use :meth:`SemanticNegotiationPipeline.run`, which
returns :class:`SemanticPipelineRun` with ``issues`` and ``options_per_issue``.
HTTP initiate builds :class:`~app.agent.negotiation_model.NegotiationParticipant`
lists only *after* options exist (server-side prefs are derived per option list),
so the API uses ``run(..., after_options=...)`` rather than pre-filling
``self.participants``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from app.agent.intent_discovery import IntentDiscovery
from app.agent.negotiation_model import (
    NegotiationParticipant,
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
    "NegotiationParticipant",
    "IntentDiscovery",
    "OptionsGeneration",
]


class SemanticNegotiationPipeline:
    """Orchestrates the semantic negotiation flow.

    Provides three public methods that map to the three pipeline components:

    1. :meth:`discover_and_generate` - Components 1+2 (IntentDiscovery + OptionsGeneration).
    2. :meth:`start_negotiation` - Component 3 seed: creates a session and returns
       the first batch of agent messages.
    3. :meth:`step_negotiation` - Component 3 advance: applies one batch of replies
       and returns the next messages or the final result.

    Args:
        n_steps: Maximum SAO rounds for the negotiation.
    """

    def __init__(self, n_steps: int = 100) -> None:
        self.n_steps = n_steps
        self._intent_discovery = IntentDiscovery()
        self._options_generation = OptionsGeneration()
        # session_id → (runner, sess)
        self._sessions: Dict[str, tuple] = {}

    def start_negotiation(
        self,
        issues: List[str],
        options_per_issue: Dict[str, List[str]],
        participants: List[NegotiationParticipant],
        session_id: str,
        n_steps: int | None = None,
    ) -> tuple:
        """Seed a turn-by-turn SAO session (Component 3, step 0).

        Returns ``(runner, sess, first_round_messages)`` where *runner* and
        *sess* must be passed back to :meth:`step_negotiation` on every
        subsequent ``/decide`` call.
        """
        from app.agent.batch_callback_runner import BatchCallbackRunner

        runner = BatchCallbackRunner(
            n_steps=n_steps if n_steps is not None else self.n_steps
        )
        sess, first_round_messages = runner.start(
            issues=issues,
            options_per_issue=options_per_issue,
            participants=participants,
            session_id=session_id,
        )
        return runner, sess, first_round_messages

    def step_negotiation(
        self,
        runner: Any,
        sess: Any,
        agent_replies: List[Dict[str, Any]],
    ) -> tuple:
        """Advance the SAO by one batch of agent replies (Component 3, step N).

        Returns the same ``(status, next_messages, result)`` triple as
        :meth:`~app.agent.batch_callback_runner.BatchCallbackRunner.step`.
        """
        return runner.step(sess, agent_replies)

    def discover_and_generate(
        self,
        content_text: str,
    ) -> tuple[List[str], Dict[str, List[str]]]:
        """Run only Components 1 and 2 and return ``(issues, options_per_issue)``.

        Use this from the turn-by-turn ``/initiate`` endpoint to get issues and
        options without triggering the NegMAS negotiation (Component 3).
        """
        issues = self._intent_discovery.discover(sentence=content_text)
        if hasattr(issues, "negotiable_entities"):
            issues = issues.negotiable_entities
        options_per_issue = self._options_generation.generate_options(
            issues, content_text
        )
        return issues, options_per_issue

    def execute(
        self,
        session_id: str,
        *,
        n_steps: int | None = None,
        content_text: str = "",
        agents_raw: List[Dict[str, Any]] | None = None,
        initiate_message: Dict[str, Any] | None = None,
        agent_replies: List[Dict[str, Any]] | None = None,
        commit_message_id: str = "",
    ) -> Dict[str, Any]:
        """Unified entry point for both ``/initiate`` and ``/decide``.

        - **New session** (``session_id`` not yet registered): runs Components 1+2
          then seeds the SAO session via :meth:`start_negotiation`.  Requires
          *content_text* and *agents_raw*.
        - **Existing session** (``session_id`` already registered): advances the SAO
          by one batch via :meth:`step_negotiation`.  Requires *agent_replies*.
          Raises ``KeyError`` if the session is not found (caller should return 404).

        Returns a dict with ``status`` and either ``messages`` (ongoing) or
        ``result`` (terminal — ``agreed``, ``broken``, or ``timeout``).
        """
        if session_id not in self._sessions:
            # ── Initiate ──────────────────────────────────────────────
            if agents_raw is None:
                raise KeyError(session_id)
            effective_n = n_steps if n_steps is not None else self.n_steps
            issues, options_per_issue = self.discover_and_generate(content_text)
            participants = [
                NegotiationParticipant(id=a["id"], name=a["name"]) for a in agents_raw
            ]
            runner, sess, messages = self.start_negotiation(
                issues, options_per_issue, participants, session_id, n_steps=effective_n
            )
            if initiate_message:
                sess.sstp_message_trace.insert(0, initiate_message)
            self._sessions[session_id] = (runner, sess)
            return {
                "status": "initiated",
                "session_id": session_id,
                "issues": issues,
                "options_per_issue": options_per_issue,
                "n_steps": effective_n,
                "round": 1,
                "messages": messages,
            }

        # ── Decide ────────────────────────────────────────────────────
        runner, sess = self._sessions[session_id]
        status, next_messages, result = self.step_negotiation(
            runner, sess, agent_replies or []
        )
        if status == "ongoing":
            # Detect if the last outgoing message is already a commit envelope.
            # This can happen when the runner emits a commit as part of its final
            # propose/respond cycle before the terminal step is reached.
            if next_messages:
                last_msg = next_messages[-1]
                last_kind = (
                    last_msg.kind if hasattr(last_msg, "kind") else last_msg.get("kind")
                )
                if last_kind == "commit":
                    pass  # TODO validate later
            return {
                "status": "ongoing",
                "session_id": session_id,
                "round": sess.sao_step + 1,
                "messages": next_messages,
            }
        # Terminal — build commit envelope and clean up session.
        del self._sessions[session_id]
        participant_id_by_name = {p.name: p.id for p in sess.participants}
        commit = self.build_commit_envelope(
            result, sess.issues, participant_id_by_name, session_id, commit_message_id
        )
        return {
            "status": status,
            "session_id": session_id,
            "round": sess.sao_step,
            "result": result,
            "issues": sess.issues,
            "participant_id_by_name": participant_id_by_name,
            "final_result": commit,
        }

    def build_commit_envelope(
        self,
        result: Any,
        issues: List[str],
        participant_id_by_name: Dict[str, str],
        session_id: str,
        message_id: str,
    ) -> Any:
        """Build an ``SSTPCommitMessage`` from a terminal :class:`NegotiationResult`.

        Appends the serialised commit envelope to ``result.sstp_message_trace``
        so the complete dialogue history (initiate → all rounds → commit) is
        captured in one list.

        Returns the ``SSTPCommitMessage`` instance.
        """
        from datetime import datetime, timezone

        from protocol.sstp import SSTPCommitMessage
        from protocol.sstp._base import LogicalClock, Origin, PolicyLabels, Provenance
        from protocol.sstp.commit import NegotiateCommitSemanticContext

        from app.api.schemas import (
            AgentDecision,
            NegotiationOutcomeResponse,
            NegotiationTrace,
            RoundOffer,
        )

        # ── Rebuild negotiation trace ────────────────────────────────
        def _resolve_id(name: str) -> str:
            if name in participant_id_by_name:
                return participant_id_by_name[name]
            for pname, pid in participant_id_by_name.items():
                if name.startswith(pname):
                    return pid
            return name

        decisions_map: Dict[int, List[Any]] = getattr(result, "round_decisions", {})
        rounds: List[Any] = []
        for idx, (_, name, offer_tuple) in enumerate(result.history):
            offer: Dict[str, str] = {}
            if offer_tuple is not None:
                for i, issue_id in enumerate(issues):
                    offer[issue_id] = str(offer_tuple[i])
            round_num = idx + 1
            raw_decs = decisions_map.get(round_num, [])
            decisions = [
                AgentDecision(
                    participant_id=d["participant_id"],
                    action=d["action"],
                    offer=d.get("offer"),
                )
                for d in raw_decs
            ]
            rounds.append(
                RoundOffer(
                    round=round_num,
                    proposer_id=_resolve_id(name),
                    offer=offer,
                    decisions=decisions,
                )
            )

        final_agreement = None
        if result.agreement is not None:
            final_agreement = [
                NegotiationOutcomeResponse(
                    issue_id=o.issue_id, chosen_option=o.chosen_option
                )
                for o in result.agreement
            ]

        trace = NegotiationTrace(
            rounds=rounds,
            final_agreement=final_agreement,
            timedout=result.timedout,
            broken=result.broken,
            sstp_message_trace=result.sstp_message_trace or None,
        )

        total_rounds = len(rounds)
        _agreed = result.agreement is not None
        _status_str = (
            "agreed" if _agreed else ("broken" if result.broken else "timeout")
        )

        # ── Build SSTPCommitMessage ──────────────────────────────────
        commit = SSTPCommitMessage(
            kind="commit",
            message_id=message_id,
            dt_created=datetime.now(timezone.utc).isoformat(),
            origin=Origin(actor_id="negotiation-server", tenant_id=session_id),
            semantic_context=NegotiateCommitSemanticContext(
                session_id=session_id,
                final_agreement=(
                    [o.model_dump() for o in trace.final_agreement]
                    if trace.final_agreement
                    else None
                ),
            ),
            payload_hash="0" * 64,
            policy_labels=PolicyLabels(
                sensitivity="internal",
                propagation="restricted",
                retention_policy="default",
            ),
            provenance=Provenance(sources=[], transforms=[]),
            payload={
                "status": _status_str,
                "session_id": session_id,
                "total_rounds": total_rounds,
                "trace": trace.model_dump(mode="json"),
            },
            state_object_id=session_id,
            parent_ids=[message_id],
            logical_clock=LogicalClock(type="lamport", value=total_rounds),
            merge_strategy="add",
            confidence_score=1.0 if _agreed else 0.0,
            risk_score=0.0 if _agreed else 1.0,
            ttl_seconds=86400,
        )

        # Append the commit to the full message trace.
        if isinstance(getattr(result, "sstp_message_trace", None), list):
            result.sstp_message_trace.append(commit.model_dump(mode="json"))

        # ── Persist the flat trace to disk ───────────────────────────
        messages: list[dict] = result.sstp_message_trace or []
        if messages:
            workspace_root = Path(__file__).resolve().parents[3]
            out_dir = workspace_root / "neg_trace" / session_id
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "sstp_message_trace.json"
            try:
                with open(out_path, "w", encoding="utf-8") as fh:
                    json.dump(messages, fh, indent=2)
                logging.getLogger(__name__).info(
                    "[%s] SSTP trace written: %s (%d messages)",
                    session_id,
                    out_path,
                    len(messages),
                )
            except OSError as exc:
                logging.getLogger(__name__).warning(
                    "[%s] failed to write SSTP trace: %s", session_id, exc
                )

        return commit

    def release_session(self, session_id: str) -> None:
        """Remove a session from the internal store (e.g. on error clean-up)."""
        self._sessions.pop(session_id, None)
