# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Batch callback runner — drives SAO-style rounds via batched agent messages.

Architecture
------------
This runner collects **all** per-participant decision requests for a round into
a **single** batch:

    Forward:   List[SSTPNegotiateMessage]  (one message per participant)
    Return:    List[SSTPNegotiateMessage]  (same length)

Each message carries ``payload.participant_id`` so the receiving endpoint knows
which participant it is addressing.

Round semantics
---------------
**Round 1 (server-seeded):**
  The negotiation server picks a *random* initial offer (one option per issue
  chosen uniformly at random) and forwards it to **all** participants as
  ``action=respond``.  There are no shadow calls.  A participant that accepts
  immediately ends the negotiation.  The server also randomly selects which
  participant will be the proposer in round 2.

**Rounds 2 … N (alternating SAO):**
  - **Proposer** (rotates, starting from the server's random pick):
    receives ``action=propose``.  Returns a new offer via
    ``{ "action": "counter_offer", "offer": {...} }``.
  - **Responders** (all others): receive ``action=respond`` with
    ``current_offer`` = the standing offer from the previous round.
    Return ``{ "action": "accept" | "reject" }``.
  - ``is_shadow_call`` is **always** ``false`` — every message represents a
    real decision.

Evaluation after each call
--------------------------
1. Round 1: if any participant accepts the random offer → agreement.
2. Rounds 2+: if any responder accepts the standing offer → agreement.
3. If the proposer returns no valid offer → ``broken``.
4. If all rounds exhaust without agreement → ``timedout``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

_workspace_root = str(Path(__file__).resolve().parents[3])
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

from protocol.sstp import SSTPNegotiateMessage  # noqa: E402
from protocol.sstp._base import Origin, PolicyLabels, Provenance  # noqa: E402
from protocol.sstp.negotiate import NegotiateSemanticContext  # noqa: E402
from protocol.sstp.negmas_sao import SAOState  # noqa: E402

from .negotiation_model import (  # noqa: E402  (same package)
    NegotiationOutcome,
    NegotiationParticipant,
    NegotiationResult,
)

logger = logging.getLogger(__name__)


def store_decisions(
    key: str, replies: list
) -> None:  # noqa: U100 — kept for backward compat
    """No-op stub retained so existing imports don't break."""


def _purge_session_decisions(session_id: str) -> None:  # noqa: U100
    """No-op stub retained so negotiation_model.py's finally-block call doesn't break."""


# ---------------------------------------------------------------------------
# Session state — holds everything needed to continue a paused negotiation
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class NegotiationSession:
    """Mutable state snapshot for one in-progress turn-by-turn negotiation.

    Created by :meth:`BatchCallbackRunner.start()` and updated in place by
    :meth:`BatchCallbackRunner.step()`.

    ``phase`` tracks exactly what kind of reply the server is waiting for next:
    - ``"round1_respond"``  — all agents must accept/reject the server's seed offer.
    - ``"propose"``         — the designated proposer must return a counter-offer.
    - ``"respond"``         — all non-proposers must accept/reject the proposer's offer.
    """

    session_id: str
    issues: list[str]
    options_per_issue: dict[str, list[str]]
    participants: list[Any]  # list[NegotiationParticipant]
    n_steps: int
    # current SAO step index (1-based, advances after each full propose+respond pair)
    sao_step: int = 1
    standing_offer: dict[str, str] = dataclasses.field(default_factory=dict)
    standing_offer_proposer_id: str = "server"
    first_proposer_idx: int = 0
    negmas_history: list[tuple] = dataclasses.field(default_factory=list)
    round_decisions: dict[int, list[dict[str, Any]]] = dataclasses.field(
        default_factory=dict
    )
    sstp_message_trace: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    # Mission metadata — stored at initiation so it can be embedded in the commit.
    content_text: str = ""
    agents_negotiating: list[str] = dataclasses.field(default_factory=list)
    # Phase drives what step() does with the incoming replies.
    phase: str = "round1_respond"  # "round1_respond" | "propose" | "respond"
    # Filled by "propose" phase, consumed by "respond" phase.
    pending_new_offer: dict[str, str] = dataclasses.field(default_factory=dict)
    pending_proposer_id: str = ""
    pending_round_decs: list[dict[str, Any]] = dataclasses.field(default_factory=list)


# ---------------------------------------------------------------------------
# SSTP message helpers
# ---------------------------------------------------------------------------


def build_callback_message(
    payload: dict[str, Any],
    participant_id: str,
    session_id: str,
    sao_state: SAOState | None = None,
    issues: list[str] | None = None,
    options_per_issue: dict[str, list[str]] | None = None,
) -> SSTPNegotiateMessage:
    """Build a validated ``SSTPNegotiateMessage`` for one participant decision request.

    ``message_id`` is a deterministic UUID-5 derived from session, participant,
    and payload content — stable across retries of the same round state.
    ``sao_state`` carries the full SAO mechanism snapshot so the receiver has
    complete context about the current negotiation state.
    ``issues`` and ``options_per_issue`` are placed in ``semantic_context`` (not
    in ``payload``) so that the full negotiation space travels in the structured
    envelope rather than the free-form payload dict.
    """
    payload_str = json.dumps(payload, sort_keys=True)
    payload_hash = hashlib.sha256(payload_str.encode()).hexdigest()
    message_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{session_id}:{participant_id}:{payload_hash}",
        )
    )
    return SSTPNegotiateMessage(
        kind="negotiate",
        message_id=message_id,
        dt_created=datetime.now(timezone.utc).isoformat(),
        origin=Origin(actor_id="negotiation-server", tenant_id=session_id),
        semantic_context=NegotiateSemanticContext(
            session_id=session_id,
            sao_state=sao_state,
            issues=issues or [],
            options_per_issue=options_per_issue or {},
        ),
        payload_hash=payload_hash,
        policy_labels=PolicyLabels(
            sensitivity="internal",
            propagation="restricted",
            retention_policy="default",
        ),
        provenance=Provenance(sources=[], transforms=[]),
        payload=payload,
    )


def unwrap_reply(data: dict[str, Any]) -> dict[str, Any]:
    """Extract the inner payload from an SSTP-wrapped reply dict.

    If the reply is a full ``SSTPNegotiateMessage`` envelope (has a ``"payload"``
    key whose value is a dict), return that inner payload.  Otherwise return
    *data* unchanged (plain-JSON reply compatibility).
    """
    inner = data.get("payload")
    if isinstance(inner, dict):
        return inner
    return data


# ---------------------------------------------------------------------------
# Batch round runner
# ---------------------------------------------------------------------------


class BatchCallbackRunner:
    """SAO-style negotiation runner that drives rounds via batch HTTP calls.

    Each round dispatches a ``List[SSTPNegotiateMessage]`` (one per participant)
    to a shared agent endpoint and waits for the replies.

    Args:
        n_steps: Maximum rounds before the negotiation times out.
        timeout: HTTP request timeout in seconds (per batch call).
    """

    def __init__(self, n_steps: int = 100, timeout: float = 30.0) -> None:
        self.n_steps = n_steps
        self._timeout = timeout
        self._http = httpx.Client(timeout=timeout)
        # Maps "session_id:round_num:participant_id" → SHA-256 hex of outbound
        # sao_state so we can detect agent tampering on the echo.
        self._sao_state_checksums: dict[str, str] = {}

    def run(
        self,
        issues: list[str],
        options_per_issue: dict[str, list[str]],
        participants: list[NegotiationParticipant],
        session_id: str = "unknown",
        agent_url: str = "",
    ) -> NegotiationResult:
        """Execute the negotiation and return the result.

        Args:
            issues: Ordered list of issue identifiers.
            options_per_issue: ``{issue_id: [option, ...]}`` for every issue.
            participants: All negotiating parties.
            session_id: Correlation identifier threaded through every message.
            agent_url: Shared HTTP endpoint to POST batched messages to.

        Returns:
            :class:`~app.agent.negotiation_model.NegotiationResult`.
        """
        callback_url: str = agent_url

        # ── Round 1: server picks a random initial offer ──────────────
        standing_offer: dict[str, str] = {
            issue: random.choice(options_per_issue[issue]) for issue in issues
        }
        # The proposer id of the current standing offer (used to prevent an
        # agent from triggering agreement by accepting their own proposal).
        standing_offer_proposer_id: str = "server"
        # Randomly decide which participant proposes in round 2.
        first_proposer_idx: int = random.randrange(len(participants))
        logger.info(
            "[%s] server initial offer: %s  |  first proposer: %s",
            session_id,
            standing_offer,
            participants[first_proposer_idx].id,
        )

        # NegMAS-compatible history: (step, proposer_name, offer_tuple)
        negmas_history: list[tuple] = [
            (
                -1,
                "server",
                tuple(standing_offer[issue] for issue in issues),
            )
        ]
        # Per-round agent decisions: {round_num: [{participant_id, action, offer?}]}
        round_decisions: dict[int, list[dict[str, Any]]] = {}
        # Chronological log of every SSTPNegotiateMessage sent to or received from agents.
        sstp_message_trace: list[dict[str, Any]] = []

        for step in range(self.n_steps):
            round_num = step + 1

            # ── Round 1: all agents respond to server's random offer ──
            if step == 0:
                round1_state = SAOState(
                    step=0,
                    relative_time=0.0,
                    running=True,
                    started=True,
                    n_negotiators=len(participants),
                    current_offer=standing_offer,
                    current_proposer="server",
                )
                messages: list[SSTPNegotiateMessage] = []
                for p in participants:
                    payload: dict[str, Any] = {
                        "action": "respond",
                        "participant_id": p.id,
                        "round": round_num,
                        "n_steps": self.n_steps,
                        "can_counter_offer": False,
                        "allowed_actions": ["accept", "reject"],
                        "is_shadow_call": False,
                        "current_offer": standing_offer,
                        "proposer_id": "server",
                    }
                    messages.append(
                        build_callback_message(
                            payload,
                            p.id,
                            session_id,
                            sao_state=round1_state,
                            issues=issues,
                            options_per_issue=options_per_issue,
                        )
                    )

                # Record outbound messages before dispatching.
                sstp_message_trace.extend(m.model_dump(mode="json") for m in messages)
                self._store_sao_checksums(messages, session_id, round_num)
                replies_raw = self._post_batch(
                    callback_url, messages, session_id, round_num
                )
                if replies_raw is None:
                    return NegotiationResult(
                        agreement=None,
                        timedout=False,
                        broken=True,
                        steps=step,
                        history=negmas_history,
                        sstp_message_trace=sstp_message_trace,
                    )
                self._verify_sao_checksums(messages, replies_raw, session_id, round_num)
                # Record agent replies in the trace.
                sstp_message_trace.extend(replies_raw)
                replies = [unwrap_reply(r) for r in replies_raw]

                # Record every participant's round-1 decision.
                round_decisions[1] = [
                    {"participant_id": p.id, "action": r.get("action", "reject")}
                    for p, r in zip(participants, replies)
                ]

                if all(r.get("action") == "accept" for r in replies):
                    logger.info(
                        "[%s] round 1 — agreement on server initial offer: %s",
                        session_id,
                        standing_offer,
                    )
                    agreement = [
                        NegotiationOutcome(
                            issue_id=issue, chosen_option=standing_offer[issue]
                        )
                        for issue in issues
                    ]
                    return NegotiationResult(
                        agreement=agreement,
                        timedout=False,
                        broken=False,
                        steps=round_num,
                        history=negmas_history,
                        round_decisions=round_decisions,
                        sstp_message_trace=sstp_message_trace,
                    )
                continue  # proceed to alternating rounds

            # ── Rounds 2+: two-step SAO ────────────────────────────────
            # Step A — ask the proposer to make their offer.
            proposer_idx = (first_proposer_idx + (step - 1)) % len(participants)
            proposer = participants[proposer_idx]

            propose_state = SAOState(
                step=step,
                relative_time=step / self.n_steps,
                running=True,
                started=True,
                n_negotiators=len(participants),
                current_offer=standing_offer,
                current_proposer=standing_offer_proposer_id,
            )
            propose_msg = build_callback_message(
                {
                    "action": "propose",
                    "participant_id": proposer.id,
                    "round": round_num,
                    "n_steps": self.n_steps,
                    "can_counter_offer": True,
                    "allowed_actions": ["counter_offer"],
                    "is_shadow_call": False,
                },
                proposer.id,
                session_id,
                sao_state=propose_state,
                issues=issues,
                options_per_issue=options_per_issue,
            )
            # Record propose message before dispatching.
            sstp_message_trace.append(propose_msg.model_dump(mode="json"))
            self._store_sao_checksums([propose_msg], session_id, round_num)
            propose_replies = self._post_batch(
                callback_url, [propose_msg], session_id, round_num
            )
            if not propose_replies:
                return NegotiationResult(
                    agreement=None,
                    timedout=False,
                    broken=True,
                    steps=round_num,
                    history=negmas_history,
                    sstp_message_trace=sstp_message_trace,
                )
            self._verify_sao_checksums(
                [propose_msg], propose_replies, session_id, round_num
            )
            # Record proposer reply in the trace.
            sstp_message_trace.extend(propose_replies)
            propose_reply = unwrap_reply(propose_replies[0])
            offer_raw = propose_reply.get("offer")
            if not (
                isinstance(offer_raw, dict)
                and all(issue in offer_raw for issue in issues)
            ):
                logger.warning(
                    "[%s] round %d — proposer '%s' returned invalid/missing offer: %s",
                    session_id,
                    round_num,
                    proposer.id,
                    propose_reply,
                )
                return NegotiationResult(
                    agreement=None,
                    timedout=False,
                    broken=True,
                    steps=round_num,
                    history=negmas_history,
                    sstp_message_trace=sstp_message_trace,
                )
            new_offer: dict[str, str] = {
                issue: str(offer_raw[issue]) for issue in issues
            }

            logger.info(
                "[%s] round %d — proposer '%s' offers %s",
                session_id,
                round_num,
                proposer.id,
                new_offer,
            )
            standing_offer = new_offer
            standing_offer_proposer_id = proposer.id
            # Start building this round's decision list with the proposer's counter_offer.
            round_decs: list[dict[str, Any]] = [
                {
                    "participant_id": proposer.id,
                    "action": "counter_offer",
                    "offer": new_offer,
                }
            ]
            negmas_history.append(
                (
                    step,
                    proposer.name,
                    tuple(new_offer[issue] for issue in issues),
                )
            )

            # Step B — ask all non-proposers to respond to the fresh offer.
            responders = [p for p in participants if p.id != proposer.id]
            if not responders:
                continue  # only one participant; nothing to accept

            respond_state = SAOState(
                step=step,
                relative_time=step / self.n_steps,
                running=True,
                started=True,
                n_negotiators=len(participants),
                current_offer=new_offer,
                current_proposer=proposer.id,
            )
            respond_messages: list[SSTPNegotiateMessage] = []
            for p in responders:
                respond_messages.append(
                    build_callback_message(
                        {
                            "action": "respond",
                            "participant_id": p.id,
                            "round": round_num,
                            "n_steps": self.n_steps,
                            "can_counter_offer": False,
                            "allowed_actions": ["accept", "reject"],
                            "is_shadow_call": False,
                            "current_offer": new_offer,
                            "proposer_id": proposer.id,
                        },
                        p.id,
                        session_id,
                        sao_state=respond_state,
                        issues=issues,
                        options_per_issue=options_per_issue,
                    )
                )

            # Record respond messages before dispatching.
            sstp_message_trace.extend(
                m.model_dump(mode="json") for m in respond_messages
            )
            self._store_sao_checksums(respond_messages, session_id, round_num)
            respond_replies_raw = self._post_batch(
                callback_url, respond_messages, session_id, round_num
            )
            if respond_replies_raw is None:
                return NegotiationResult(
                    agreement=None,
                    timedout=False,
                    broken=True,
                    steps=round_num,
                    history=negmas_history,
                    sstp_message_trace=sstp_message_trace,
                )
            self._verify_sao_checksums(
                respond_messages, respond_replies_raw, session_id, round_num
            )
            # Record responder replies in the trace.
            sstp_message_trace.extend(respond_replies_raw)
            respond_replies = [unwrap_reply(r) for r in respond_replies_raw]

            # Complete this round's decision list with each responder's action.
            for p, r in zip(responders, respond_replies):
                round_decs.append(
                    {"participant_id": p.id, "action": r.get("action", "reject")}
                )
            round_decisions[round_num] = round_decs

            if all(r.get("action") == "accept" for r in respond_replies):
                logger.info(
                    "[%s] round %d — agreement on proposer '%s' offer: %s",
                    session_id,
                    round_num,
                    proposer.id,
                    new_offer,
                )
                agreement = [
                    NegotiationOutcome(issue_id=issue, chosen_option=new_offer[issue])
                    for issue in issues
                ]
                return NegotiationResult(
                    agreement=agreement,
                    timedout=False,
                    broken=False,
                    steps=round_num,
                    history=negmas_history,
                    round_decisions=round_decisions,
                    sstp_message_trace=sstp_message_trace,
                )

        # exhausted step budget
        return NegotiationResult(
            agreement=None,
            timedout=True,
            broken=False,
            steps=self.n_steps,
            history=negmas_history,
            round_decisions=round_decisions,
            sstp_message_trace=sstp_message_trace,
        )

    # ------------------------------------------------------------------
    # Turn-by-turn API (stateful stepper)
    # ------------------------------------------------------------------

    def start(
        self,
        issues: list[str],
        options_per_issue: dict[str, list[str]],
        participants: list[NegotiationParticipant],
        session_id: str = "unknown",
    ) -> tuple[NegotiationSession, list[dict[str, Any]]]:
        """Initialise a new negotiation session and return the first round's messages.

        Creates a :class:`NegotiationSession`, seeds the server's random initial
        offer, builds the round-1 ``respond`` batch, and returns:

        ``(session, first_round_messages)``

        where ``first_round_messages`` is a list of serialised
        ``SSTPNegotiateMessage`` dicts ready to be forwarded to the agents.
        The caller must persist *session* and pass it back to :meth:`step`.
        """
        standing_offer: dict[str, str] = {
            issue: random.choice(options_per_issue[issue]) for issue in issues
        }
        first_proposer_idx = random.randrange(len(participants))
        history: list[tuple] = [
            (-1, "server", tuple(standing_offer[issue] for issue in issues))
        ]
        logger.info(
            "[%s] start — server initial offer: %s  |  first proposer: %s",
            session_id,
            standing_offer,
            participants[first_proposer_idx].id,
        )

        sess = NegotiationSession(
            session_id=session_id,
            issues=issues,
            options_per_issue=options_per_issue,
            participants=participants,
            n_steps=self.n_steps,
            sao_step=1,
            standing_offer=standing_offer,
            standing_offer_proposer_id="server",
            first_proposer_idx=first_proposer_idx,
            negmas_history=history,
            phase="round1_respond",
        )

        # Build round-1 messages (all agents respond to server's random offer).
        round1_state = SAOState(
            step=0,
            relative_time=0.0,
            running=True,
            started=True,
            n_negotiators=len(participants),
            current_offer=standing_offer,
            current_proposer="server",
        )
        messages: list[SSTPNegotiateMessage] = []
        for p in participants:
            messages.append(
                build_callback_message(
                    {
                        "action": "respond",
                        "participant_id": p.id,
                        "round": 1,
                        "n_steps": self.n_steps,
                        "can_counter_offer": False,
                        "allowed_actions": ["accept", "reject"],
                        "is_shadow_call": False,
                        "current_offer": standing_offer,
                        "proposer_id": "server",
                    },
                    p.id,
                    session_id,
                    sao_state=round1_state,
                    issues=issues,
                    options_per_issue=options_per_issue,
                )
            )

        serialised = [m.model_dump(mode="json") for m in messages]
        sess.sstp_message_trace.extend(serialised)
        return sess, serialised

    def step(
        self,
        sess: NegotiationSession,
        agent_replies: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]] | None, NegotiationResult | None]:
        """Advance the negotiation by one batch of agent replies.

        Args:
            sess: The mutable session returned by :meth:`start` (or updated by
                  a previous :meth:`step` call).
            agent_replies: The list of raw reply dicts returned by the agents for
                           the messages dispatched in the previous call.

        Returns:
            A ``(status, next_messages, result)`` triple:

            - ``("ongoing", next_messages, None)`` — more rounds to go.
            - ``("agreed"|"broken"|"timeout", None, result)`` — negotiation done.
        """
        issues = sess.issues
        participants = sess.participants
        session_id = sess.session_id

        # Record agent replies in the trace.
        sess.sstp_message_trace.extend(agent_replies)
        replies = [unwrap_reply(r) for r in agent_replies]

        # ── Phase: round1_respond ─────────────────────────────────────
        if sess.phase == "round1_respond":
            sess.round_decisions[1] = [
                {"participant_id": p.id, "action": r.get("action", "reject")}
                for p, r in zip(participants, replies)
            ]
            if all(r.get("action") == "accept" for r in replies):
                result = NegotiationResult(
                    agreement=[
                        NegotiationOutcome(
                            issue_id=issue, chosen_option=sess.standing_offer[issue]
                        )
                        for issue in issues
                    ],
                    timedout=False,
                    broken=False,
                    steps=1,
                    history=sess.negmas_history,
                    round_decisions=sess.round_decisions,
                    sstp_message_trace=sess.sstp_message_trace,
                )
                return "agreed", None, result
            # Not agreed — dispatch first propose (sao_step already == 1).
            return self._dispatch_propose(sess)

        # ── Phase: propose ────────────────────────────────────────────
        if sess.phase == "propose":
            propose_reply = replies[0] if replies else {}
            offer_raw = propose_reply.get("offer")
            proposer_idx = (sess.first_proposer_idx + (sess.sao_step - 1)) % len(
                participants
            )
            proposer = participants[proposer_idx]

            if not (
                isinstance(offer_raw, dict)
                and all(issue in offer_raw for issue in issues)
            ):
                logger.warning(
                    "[%s] sao_step %d — proposer '%s' returned invalid offer: %s",
                    session_id,
                    sess.sao_step,
                    proposer.id,
                    propose_reply,
                )
                return (
                    "broken",
                    None,
                    NegotiationResult(
                        agreement=None,
                        timedout=False,
                        broken=True,
                        steps=sess.sao_step + 1,
                        history=sess.negmas_history,
                        round_decisions=sess.round_decisions,
                        sstp_message_trace=sess.sstp_message_trace,
                    ),
                )

            new_offer: dict[str, str] = {
                issue: str(offer_raw[issue]) for issue in issues
            }
            sess.standing_offer = new_offer
            sess.standing_offer_proposer_id = proposer.id
            sess.pending_new_offer = new_offer
            sess.pending_proposer_id = proposer.id
            sess.pending_round_decs = [
                {
                    "participant_id": proposer.id,
                    "action": "counter_offer",
                    "offer": new_offer,
                }
            ]
            sess.negmas_history.append(
                (
                    sess.sao_step,
                    proposer.name,
                    tuple(new_offer[issue] for issue in issues),
                )
            )
            return self._dispatch_respond(sess)

        # ── Phase: respond ────────────────────────────────────────────
        if sess.phase == "respond":
            responders = [p for p in participants if p.id != sess.pending_proposer_id]
            for p, r in zip(responders, replies):
                sess.pending_round_decs.append(
                    {"participant_id": p.id, "action": r.get("action", "reject")}
                )
            round_num = sess.sao_step + 1
            sess.round_decisions[round_num] = list(sess.pending_round_decs)
            sess.pending_round_decs = []

            if all(r.get("action") == "accept" for r in replies):
                result = NegotiationResult(
                    agreement=[
                        NegotiationOutcome(
                            issue_id=issue, chosen_option=sess.pending_new_offer[issue]  # type: ignore[index]
                        )
                        for issue in issues
                    ],
                    timedout=False,
                    broken=False,
                    steps=round_num,
                    history=sess.negmas_history,
                    round_decisions=sess.round_decisions,
                    sstp_message_trace=sess.sstp_message_trace,
                )
                return "agreed", None, result

            # Not agreed — advance to next propose step.
            sess.sao_step += 1
            if sess.sao_step >= sess.n_steps:
                return (
                    "timeout",
                    None,
                    NegotiationResult(
                        agreement=None,
                        timedout=True,
                        broken=False,
                        steps=sess.n_steps,
                        history=sess.negmas_history,
                        round_decisions=sess.round_decisions,
                        sstp_message_trace=sess.sstp_message_trace,
                    ),
                )
            return self._dispatch_propose(sess)

        # Should never reach here.
        raise RuntimeError(f"[{session_id}] Unknown phase: {sess.phase!r}")

    def _dispatch_propose(
        self, sess: NegotiationSession
    ) -> tuple[str, list[dict[str, Any]], None]:
        """Build a propose message for sao_step and return an 'ongoing' triple."""
        issues = sess.issues
        options_per_issue = sess.options_per_issue
        participants = sess.participants
        session_id = sess.session_id
        st = sess.sao_step
        round_num = st + 1
        proposer_idx = (sess.first_proposer_idx + (st - 1)) % len(participants)
        proposer = participants[proposer_idx]

        propose_state = SAOState(
            step=st,
            relative_time=st / sess.n_steps,
            running=True,
            started=True,
            n_negotiators=len(participants),
            current_offer=sess.standing_offer,
            current_proposer=sess.standing_offer_proposer_id,
        )
        propose_msg = build_callback_message(
            {
                "action": "propose",
                "participant_id": proposer.id,
                "round": round_num,
                "n_steps": sess.n_steps,
                "can_counter_offer": True,
                "allowed_actions": ["counter_offer"],
                "is_shadow_call": False,
            },
            proposer.id,
            session_id,
            sao_state=propose_state,
            issues=issues,
            options_per_issue=options_per_issue,
        )
        serialised = [propose_msg.model_dump(mode="json")]
        sess.sstp_message_trace.extend(serialised)
        sess.phase = "propose"
        logger.debug(
            "[%s] dispatching propose round %d to %s",
            session_id,
            round_num,
            proposer.id,
        )
        return "ongoing", serialised, None

    def _dispatch_respond(
        self, sess: NegotiationSession
    ) -> tuple[str, list[dict[str, Any]], None]:
        """Build respond messages for all non-proposers and return an 'ongoing' triple."""
        issues = sess.issues
        options_per_issue = sess.options_per_issue
        participants = sess.participants
        session_id = sess.session_id
        st = sess.sao_step
        round_num = st + 1
        new_offer = sess.pending_new_offer
        proposer_id = sess.pending_proposer_id

        respond_state = SAOState(
            step=st,
            relative_time=st / sess.n_steps,
            running=True,
            started=True,
            n_negotiators=len(participants),
            current_offer=new_offer,
            current_proposer=proposer_id,
        )
        responders = [p for p in participants if p.id != proposer_id]
        respond_messages: list[SSTPNegotiateMessage] = [
            build_callback_message(
                {
                    "action": "respond",
                    "participant_id": p.id,
                    "round": round_num,
                    "n_steps": sess.n_steps,
                    "can_counter_offer": False,
                    "allowed_actions": ["accept", "reject"],
                    "is_shadow_call": False,
                    "current_offer": new_offer,
                    "proposer_id": proposer_id,
                },
                p.id,
                session_id,
                sao_state=respond_state,
                issues=issues,
                options_per_issue=options_per_issue,
            )
            for p in responders
        ]
        serialised = [m.model_dump(mode="json") for m in respond_messages]
        sess.sstp_message_trace.extend(serialised)
        sess.phase = "respond"
        logger.debug(
            "[%s] dispatching respond round %d to %d agents",
            session_id,
            round_num,
            len(responders),
        )
        return "ongoing", serialised, None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _post_batch(
        self,
        callback_url: str,
        messages: list[SSTPNegotiateMessage],
        session_id: str,
        round_num: int,
    ) -> list[dict[str, Any]] | None:
        """POST ``messages`` to the agent's /decide endpoint and return the
        replies directly from the HTTP response body.

        The agent's /decide endpoint is a **blocking** call: it processes all
        messages synchronously and returns ``List[SSTPNegotiateMessage]`` as
        the response body.  No separate callback to /agents-decisions is needed.

        Returns the list of reply dicts (one per message) or None on error.
        """
        try:
            resp = self._http.post(
                callback_url,
                json=[m.model_dump(mode="json") for m in messages],
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            result = resp.json()
        except httpx.HTTPError as exc:
            logger.error(
                "[%s] round %d — /decide HTTP error: %s", session_id, round_num, exc
            )
            return None
        except Exception as exc:
            logger.error(
                "[%s] round %d — /decide unexpected error: %s",
                session_id,
                round_num,
                exc,
            )
            return None

        if not isinstance(result, list):
            logger.error(
                "[%s] round %d — /decide returned non-list body: %s",
                session_id,
                round_num,
                type(result),
            )
            return None

        if len(result) != len(messages):
            logger.error(
                "[%s] round %d — /decide returned %d replies (expected %d)",
                session_id,
                round_num,
                len(result),
                len(messages),
            )
            return None
        return result

    def _store_sao_checksums(
        self,
        messages: list[SSTPNegotiateMessage],
        session_id: str,
        round_num: int,
    ) -> None:
        """Compute and store SHA-256 checksums for every outbound ``sao_state``.

        Keys follow the pattern ``"{session_id}:{round_num}:{participant_id}"``
        so they can be looked up when the corresponding reply arrives.
        """
        for msg in messages:
            participant_id = (msg.payload or {}).get("participant_id", "unknown")
            ck_key = f"{session_id}:{round_num}:{participant_id}"
            sc = msg.semantic_context
            sao_state_dict = (
                sc.sao_state.model_dump(mode="json") if (sc and sc.sao_state) else None
            )
            if sao_state_dict is not None:
                self._sao_state_checksums[ck_key] = hashlib.sha256(
                    json.dumps(sao_state_dict, sort_keys=True).encode()
                ).hexdigest()

    def _verify_sao_checksums(
        self,
        messages: list[SSTPNegotiateMessage],
        replies_raw: list[dict[str, Any]],
        session_id: str,
        round_num: int,
    ) -> None:
        """Verify that each reply's echoed ``sao_state`` matches the stored checksum.

        Logs a WARNING if tampering is detected (checksum mismatch) or if the
        reply is missing the ``sao_state`` field that was sent.  Logs DEBUG on
        successful verification.
        """
        for msg, reply in zip(messages, replies_raw):
            participant_id = (msg.payload or {}).get("participant_id", "unknown")
            ck_key = f"{session_id}:{round_num}:{participant_id}"
            stored_ck = self._sao_state_checksums.get(ck_key)
            if stored_ck is None:
                continue  # nothing was stored (no sao_state was sent)

            sc_dict = reply.get("semantic_context") or {}
            echoed_sao_dict = (
                sc_dict.get("sao_state") if isinstance(sc_dict, dict) else None
            )
            if echoed_sao_dict is None:
                logger.warning(
                    "[%s] round %d — participant '%s' reply is missing sao_state "
                    "in semantic_context (expected checksum …%s)",
                    session_id,
                    round_num,
                    participant_id,
                    stored_ck[-8:],
                )
                continue

            echoed_ck = hashlib.sha256(
                json.dumps(echoed_sao_dict, sort_keys=True).encode()
            ).hexdigest()
            if echoed_ck != stored_ck:
                logger.warning(
                    "[%s] round %d — TAMPERED sao_state from participant '%s' "
                    "(expected=…%s  received=…%s)",
                    session_id,
                    round_num,
                    participant_id,
                    stored_ck[-8:],
                    echoed_ck[-8:],
                )
            else:
                logger.debug(
                    "[%s] round %d — sao_state integrity OK for participant '%s' (…%s)",
                    session_id,
                    round_num,
                    participant_id,
                    stored_ck[-8:],
                )
