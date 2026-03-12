"""Batch callback runner — drives SAO-style rounds via a shared callback URL.

Architecture
------------
Instead of calling each participant's callback independently (as
``SSTPCallbackNegotiator`` did inside NegMAS), this runner collects **all**
per-participant decision requests for a round into a **single** HTTP call:

    Request:   POST <callback_url>   body = List[SSTPNegotiateMessage]
    Response:                        List[SSTPNegotiateMessage]  (same length)

Each message in the request list carries ``payload.participant_id`` so the
receiving endpoint knows which participant it is addressing.

All participants must expose the **same** ``callback_url``.  The first
participant's URL is used for every batch call.

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

import hashlib
import json
import logging
import random
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import sleep
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

# ---------------------------------------------------------------------------
# Decisions store
# ---------------------------------------------------------------------------
# Maps a key "<session_id>:<round_num>" to the list of reply dicts.
# The agent's /decide endpoint processes the batch synchronously, POSTs the
# full decision list to /negotiate/agents-decisions, and only then returns
# an ACK.  By the time the runner receives that ACK the decisions are already
# present here — no threading.Event required.

_DECISIONS: dict[str, list] = {}
_DECISIONS_LOCK = threading.Lock()


def store_decisions(key: str, replies: list) -> None:
    """Called by the /negotiate/agents-decisions route to store replies."""
    with _DECISIONS_LOCK:
        _DECISIONS[key] = replies


# ---------------------------------------------------------------------------
# SSTP message helpers
# ---------------------------------------------------------------------------

def build_callback_message(
    payload: dict[str, Any],
    participant_id: str,
    session_id: str,
    sao_state: SAOState | None = None,
) -> SSTPNegotiateMessage:
    """Build a validated ``SSTPNegotiateMessage`` for one participant decision request.

    ``message_id`` is a deterministic UUID-5 derived from session, participant,
    and payload content — stable across retries of the same round state.
    ``sao_state`` carries the full SAO mechanism snapshot so the receiver has
    complete context about the current negotiation state.
    """
    payload_str = json.dumps(payload, sort_keys=True)
    payload_hash = hashlib.sha256(payload_str.encode()).hexdigest()
    message_id = str(uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"{session_id}:{participant_id}:{payload_hash}",
    ))
    return SSTPNegotiateMessage(
        kind="negotiate",
        message_id=message_id,
        dt_created=datetime.now(timezone.utc).isoformat(),
        origin=Origin(actor_id="negotiation-server", tenant_id=session_id),
        semantic_context=NegotiateSemanticContext(session_id=session_id, sao_state=sao_state),
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

    All participants **must** share the same ``callback_url`` — the first
    participant's URL is used.  Each round makes exactly ONE POST to that URL
    carrying a ``List[SSTPNegotiateMessage]`` (one message per participant).

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
    ) -> NegotiationResult:
        """Execute the negotiation and return the result.

        Args:
            issues: Ordered list of issue identifiers.
            options_per_issue: ``{issue_id: [option, ...]}`` for every issue.
            participants: All negotiating parties (must all have ``callback_url``).
            session_id: Correlation identifier threaded through every message.

        Returns:
            :class:`~app.agent.negotiation_model.NegotiationResult`.
        """
        callback_url: str = participants[0].callback_url  # shared endpoint

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
            session_id, standing_offer, participants[first_proposer_idx].id,
        )

        # payload-level history (sent to agents each round)
        payload_history: list[dict[str, Any]] = [{
            "round": 0,
            "proposer_id": "server",
            "offer": standing_offer,
        }]
        # NegMAS-compatible history: (step, proposer_name, offer_tuple)
        negmas_history: list[tuple] = [(
            -1, "server", tuple(standing_offer[issue] for issue in issues),
        )]

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
                        "issues": issues,
                        "options_per_issue": options_per_issue,
                        "current_offer": standing_offer,
                        "proposer_id": "server",
                        "history": payload_history,
                    }
                    messages.append(build_callback_message(payload, p.id, session_id, sao_state=round1_state))

                self._store_sao_checksums(messages, session_id, round_num)
                replies_raw = self._post_batch(callback_url, messages, session_id, round_num)
                if replies_raw is None:
                    return NegotiationResult(
                        agreement=None, timedout=False, broken=True,
                        steps=step, history=negmas_history,
                    )
                self._verify_sao_checksums(messages, replies_raw, session_id, round_num)
                replies = [unwrap_reply(r) for r in replies_raw]

                if all(r.get("action") == "accept" for r in replies):
                    logger.info(
                        "[%s] round 1 — agreement on server initial offer: %s",
                        session_id, standing_offer,
                    )
                    agreement = [
                        NegotiationOutcome(issue_id=issue, chosen_option=standing_offer[issue])
                        for issue in issues
                    ]
                    return NegotiationResult(
                        agreement=agreement, timedout=False, broken=False,
                        steps=round_num, history=negmas_history,
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
                    "issues": issues,
                    "options_per_issue": options_per_issue,
                    "history": payload_history,
                },
                proposer.id,
                session_id,
                sao_state=propose_state,
            )
            self._store_sao_checksums([propose_msg], session_id, round_num)
            propose_replies = self._post_batch(
                callback_url, [propose_msg], session_id, round_num
            )
            if not propose_replies:
                return NegotiationResult(
                    agreement=None, timedout=False, broken=True,
                    steps=round_num, history=negmas_history,
                )
            self._verify_sao_checksums([propose_msg], propose_replies, session_id, round_num)
            propose_reply = unwrap_reply(propose_replies[0])
            offer_raw = propose_reply.get("offer")
            if not (isinstance(offer_raw, dict) and all(issue in offer_raw for issue in issues)):
                logger.warning(
                    "[%s] round %d — proposer '%s' returned invalid/missing offer: %s",
                    session_id, round_num, proposer.id, propose_reply,
                )
                return NegotiationResult(
                    agreement=None, timedout=False, broken=True,
                    steps=round_num, history=negmas_history,
                )
            new_offer: dict[str, str] = {issue: str(offer_raw[issue]) for issue in issues}

            logger.info(
                "[%s] round %d — proposer '%s' offers %s",
                session_id, round_num, proposer.id, new_offer,
            )
            standing_offer = new_offer
            standing_offer_proposer_id = proposer.id
            payload_history.append({
                "round": round_num,
                "proposer_id": proposer.id,
                "offer": new_offer,
            })
            negmas_history.append((
                step,
                proposer.name,
                tuple(new_offer[issue] for issue in issues),
            ))

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
                respond_messages.append(build_callback_message(
                    {
                        "action": "respond",
                        "participant_id": p.id,
                        "round": round_num,
                        "n_steps": self.n_steps,
                        "can_counter_offer": False,
                        "allowed_actions": ["accept", "reject"],
                        "is_shadow_call": False,
                        "issues": issues,
                        "options_per_issue": options_per_issue,
                        "current_offer": new_offer,
                        "proposer_id": proposer.id,
                        "history": payload_history,
                    },
                    p.id,
                    session_id,
                    sao_state=respond_state,
                ))

            self._store_sao_checksums(respond_messages, session_id, round_num)
            respond_replies_raw = self._post_batch(
                callback_url, respond_messages, session_id, round_num
            )
            if respond_replies_raw is None:
                return NegotiationResult(
                    agreement=None, timedout=False, broken=True,
                    steps=round_num, history=negmas_history,
                )
            self._verify_sao_checksums(respond_messages, respond_replies_raw, session_id, round_num)
            respond_replies = [unwrap_reply(r) for r in respond_replies_raw]

            if all(r.get("action") == "accept" for r in respond_replies):
                logger.info(
                    "[%s] round %d — agreement on proposer '%s' offer: %s",
                    session_id, round_num, proposer.id, new_offer,
                )
                agreement = [
                    NegotiationOutcome(issue_id=issue, chosen_option=new_offer[issue])
                    for issue in issues
                ]
                return NegotiationResult(
                    agreement=agreement, timedout=False, broken=False,
                    steps=round_num, history=negmas_history,
                )

        # exhausted step budget
        return NegotiationResult(
            agreement=None, timedout=True, broken=False,
            steps=self.n_steps, history=negmas_history,
        )

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
        """Send ``messages`` to the agent's /decide endpoint, then retrieve
        decisions from the in-process ``_DECISIONS`` store.

        The agent's /decide runs **synchronously**: it processes all messages,
        POSTs the full list to POST /negotiate/agents-decisions (which calls
        ``store_decisions``), and only then returns the ACK.  By the time this
        method sees the ACK the decisions are already in ``_DECISIONS`` — no
        threading.Event or polling needed.

        Returns the list of reply dicts (one per message) or None on error.
        """
        key = f"{session_id}:{round_num}"
        try:
            resp = self._http.post(
                callback_url,
                json=[m.model_dump(mode="json") for m in messages],
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            ack = resp.json()
            if not isinstance(ack, dict) or ack.get("status") != "ack":
                logger.warning(
                    "[%s] round %d — /decide returned unexpected ACK body: %s",
                    session_id, round_num, ack,
                )
        except httpx.HTTPError as exc:
            logger.error("[%s] round %d — /decide HTTP error: %s", session_id, round_num, exc)
            return None
        except Exception as exc:
            logger.error("[%s] round %d — /decide unexpected error: %s", session_id, round_num, exc)
            return None

        # Decisions must already be present — the agent only returns ACK after
        # its synchronous POST to /negotiate/agents-decisions has completed.
        # A short retry loop guards against any marginal timing edge-case.
        result: list | None = None
        deadline = self._timeout
        while deadline > 0:
            with _DECISIONS_LOCK:
                result = _DECISIONS.pop(key, None)
            if result is not None:
                break
            sleep(0.01)
            deadline -= 0.01

        if result is None:
            logger.error(
                "[%s] round %d — decisions never arrived in store (timeout %.1fs)",
                session_id, round_num, self._timeout,
            )
            return None

        if len(result) != len(messages):
            logger.error(
                "[%s] round %d — agents-decisions returned %d replies (expected %d)",
                session_id, round_num, len(result), len(messages),
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
            echoed_sao_dict = sc_dict.get("sao_state") if isinstance(sc_dict, dict) else None
            if echoed_sao_dict is None:
                logger.warning(
                    "[%s] round %d — participant '%s' reply is missing sao_state "
                    "in semantic_context (expected checksum …%s)",
                    session_id, round_num, participant_id, stored_ck[-8:],
                )
                continue

            echoed_ck = hashlib.sha256(
                json.dumps(echoed_sao_dict, sort_keys=True).encode()
            ).hexdigest()
            if echoed_ck != stored_ck:
                logger.warning(
                    "[%s] round %d — TAMPERED sao_state from participant '%s' "
                    "(expected=…%s  received=…%s)",
                    session_id, round_num, participant_id,
                    stored_ck[-8:], echoed_ck[-8:],
                )
            else:
                logger.debug(
                    "[%s] round %d — sao_state integrity OK for participant '%s' (…%s)",
                    session_id, round_num, participant_id, stored_ck[-8:],
                )
