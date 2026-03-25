# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""SSTPCallbackNegotiator — a NegMAS SAO negotiator that delegates every
decision to an external agent via the SSTP protocol.

Instead of running a local utility curve (like BoulwareTBNegotiator), this
negotiator POSTs an ``SSTPNegotiateMessage`` to the participant's
``callback_url`` on every ``propose`` and ``respond`` call, then waits for
the agent's JSON reply.

Wire format
-----------
Both directions use **``List[SSTPNegotiateMessage]``** — the server always
sends a list and always expects a list back.  Currently each list contains a
single item per call (one decision request per agent per NegMAS step), but
the list envelope makes it straightforward to extend to true multi-agent
batching in future without changing the endpoint contract.

**Server → Agent** (both propose and respond requests):

    POST <callback_url>
    Content-Type: application/json

    [
      {
        "kind": "negotiate",
        "protocol": "SSTP",
        "version": "0",
        "message_id": "<uuid>",
        "dt_created": "<iso8601>",
        "origin": { "actor_id": "negotiation-server", "tenant_id": "<session_id>" },
        "semantic_context": {
          "schema_id": "urn:ioc:schema:negotiate:negmas-sao:v1",
          "schema_version": "1.0",
          "session_id": "<session_id>"
        },
        "payload": {

          // ---- for propose requests ----
          "action": "propose",
          "round": 4,
          "n_steps": 60,
          "can_counter_offer": true,             // true only for the designated proposer this step
          "allowed_actions": ["counter_offer"],  // only valid reply: submit an offer
          "issues": ["budget", "timeline"],
          "options_per_issue": {
            "budget":   ["low", "medium", "high"],
            "timeline": ["short", "standard", "long"]
          },

          // ---- for respond requests ----
          "action": "respond",
          "round": 3,
          "n_steps": 60,
          "can_counter_offer": false,             // always false — only accept / reject
          "allowed_actions": ["accept", "reject"],  // valid reply actions
          "issues": ["budget", "timeline"],
          "options_per_issue": {
            "budget":   ["low", "medium", "high"],
            "timeline": ["short", "standard", "long"]
          },
          "current_offer": { "budget": "medium", "timeline": "standard", ... },
          "proposer_id": "<id of the agent who made this offer>"
        }
      }
    ]

**Agent → Server** (reply body — ``List[SSTPNegotiateMessage]``):

    // reply to a propose request  (allowed_actions = ["counter_offer"])
    [
      {
        "kind": "negotiate", "protocol": "SSTP", ...,
        "payload": { "action": "counter_offer", "offer": { "budget": "low", ... } }
      }
    ]

    // reply to a respond request  (allowed_actions = ["accept", "reject"])
    [
      {
        "kind": "negotiate", "protocol": "SSTP", ...,
        "payload": { "action": "accept" }   // or "reject"
      }
    ]

The negotiator automatically unwraps the SSTP envelope: it reads
``reply["payload"]`` when a ``"payload"`` key is present, and falls back to
the raw dict for backward compatibility with plain-JSON replies.

Key fields for agent decision-making
-------------------------------------
* ``can_counter_offer`` — boolean; ``true`` only when it is the agent's own
  proposing turn (``action == "propose"``).  On a ``respond`` turn the agent
  may only accept or reject — the SAO protocol does not allow counter-offers
  in the respond phase.
* ``allowed_actions`` — explicit list of valid reply keys so agents do not
  have to infer permitted actions from ``can_counter_offer`` alone.
* ``issues`` / ``options_per_issue`` — the complete negotiation space,
  repeated on every message so agents require no prior out-of-band knowledge.

If the agent returns HTTP 4xx/5xx, or the ``"offer"``/``"action"`` keys are
missing, the negotiator falls back to a safe default:
- ``propose`` → returns ``None`` (NegMAS will skip this proposer's turn)
- ``respond`` → returns ``ResponseType.REJECT_OFFER``

Timeouts and retries are controlled by ``timeout`` (default 30 s).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from negmas.gb.common import ResponseType
from negmas.sao import SAONegotiator
from negmas.sao.common import SAOState

# Ensure workspace root is on sys.path so `protocol.sstp` is importable
# regardless of which directory the service is launched from.
_workspace_root = str(Path(__file__).resolve().parents[3])
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

from ....protocol.sstp import SSTPNegotiateMessage  # noqa: E402
from ....protocol.sstp._base import Origin, PolicyLabels, Provenance  # noqa: E402
from ....protocol.sstp.negotiate import NegotiateSemanticContext  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias — NegMAS Outcome is dict | tuple | None
# ---------------------------------------------------------------------------
Outcome = dict[str, Any] | tuple | None


class SSTPCallbackNegotiator(SAONegotiator):
    """A NegMAS SAO negotiator that delegates decisions to an external agent.

    On every ``propose`` and ``respond`` invocation it builds a full
    ``SSTPNegotiateMessage`` and POSTs it synchronously to ``callback_url``.
    The external agent processes the message and replies with a plain JSON
    object containing either an ``"offer"`` (for propose) or an ``"action"``
    (for respond).

    Args:
        name: Participant display name (passed to NegMAS).
        callback_url: HTTP endpoint of the external agent that will decide.
        participant_id: Stable identifier for this participant (used in the
            ``origin.actor_id`` field of outgoing messages).
        session_id: Negotiation session identifier threaded from the server.
        timeout: HTTP request timeout in seconds (default 30).
    """

    def __init__(
        self,
        name: str,
        callback_url: str,
        participant_id: str,
        session_id: str,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(name=name)
        self._callback_url = callback_url
        self._participant_id = participant_id
        self._session_id = session_id
        self._timeout = timeout
        self._http = httpx.Client(timeout=timeout)

    # ------------------------------------------------------------------
    # NegMAS hooks
    # ------------------------------------------------------------------

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome:
        """Ask the external agent to propose an offer for this round.

        Sends ``issues``, ``options_per_issue``, and:

        - ``can_counter_offer`` — ``true`` only when this agent is the designated
          SAO proposer for this step (NegMAS calls ``propose()`` on all agents at
          step 0 to seed opening offers, so only one agent truly has the slot).
        - ``allowed_actions: ["counter_offer"]`` — the only valid reply is to
          return ``{ "offer": { issue: value, ... } }``.

        Returns ``None`` if the call fails (NegMAS treats None as "skip turn").
        """
        issues = self._issue_names()
        options_per_issue = self._options_per_issue(issues)
        is_my_turn = self._is_my_proposing_turn(state)
        payload: dict[str, Any] = {
            "action": "propose",
            "round": state.step + 1,  # 1-based for humans
            "n_steps": self._n_steps(),
            "can_counter_offer": is_my_turn,  # True only for the designated proposer this step
            "allowed_actions": [
                "counter_offer"
            ],  # only valid reply: submit an offer dict
            "is_shadow_call": not is_my_turn,  # True = NegMAS seeding; agent should skip tracing
            "issues": issues,
            "options_per_issue": options_per_issue,
        }
        reply = self._call_agent(payload)
        if reply is None:
            logger.warning(
                "[%s] %s — propose callback failed, returning None",
                self._session_id,
                self.name,
            )
            return None

        offer_dict = reply.get("offer")
        if not isinstance(offer_dict, dict):
            logger.warning(
                "[%s] %s — propose reply missing 'offer' key: %s",
                self._session_id,
                self.name,
                reply,
            )
            return None

        # Convert the agent's dict offer to the tuple form NegMAS expects
        try:
            outcome = self._dict_to_outcome(offer_dict, issues)
            return outcome
        except Exception as exc:
            logger.warning(
                "[%s] %s — propose outcome conversion failed: %s",
                self._session_id,
                self.name,
                exc,
            )
            return None

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Ask the external agent to respond to the current offer.

        Sends ``issues``, ``options_per_issue``, the current offer, and:

        - ``can_counter_offer: false`` — the SAO respond turn does not allow
          counter-offers; the agent may only accept or reject.
        - ``allowed_actions: ["accept", "reject"]`` — the valid reply actions.

        Expects ``{ "action": "accept" | "reject" }`` in the reply.
        Returns ``ResponseType.REJECT_OFFER`` if the call fails.
        """
        issues = self._issue_names()
        options_per_issue = self._options_per_issue(issues)
        current_offer = (
            self._tuple_to_dict(state.current_offer, issues)
            if state.current_offer is not None
            else None
        )
        proposer_id = self._resolve_proposer(state, source)
        payload: dict[str, Any] = {
            "action": "respond",
            "round": state.step + 1,
            "n_steps": self._n_steps(),
            "can_counter_offer": False,  # SAO respond turn: no counter-offer allowed
            "allowed_actions": ["accept", "reject"],  # the only valid reply actions
            "is_shadow_call": self._is_my_proposing_turn(
                state
            ),  # True = proposer responding to own offer
            "issues": issues,
            "options_per_issue": options_per_issue,
            "current_offer": current_offer,
            "proposer_id": proposer_id,
        }
        reply = self._call_agent(payload)
        if reply is None:
            logger.warning(
                "[%s] %s — respond callback failed, defaulting to reject",
                self._session_id,
                self.name,
            )
            return ResponseType.REJECT_OFFER

        action = reply.get("action", "reject")
        return _parse_response_type(action)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_agent(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Build a validated SSTPNegotiateMessage, wrap it in a list, and POST to the agent's callback.

        The wire format is always ``List[SSTPNegotiateMessage]`` in both directions:
        - Request:  ``[ <SSTPNegotiateMessage> ]``
        - Response: ``[ <SSTPNegotiateMessage> ]``

        The single-item list convention keeps the interface uniform — future
        batching of multiple per-agent decisions in one HTTP round-trip requires
        no endpoint contract change.
        """
        message = self._build_sstp_message(payload)
        try:
            resp = self._http.post(
                self._callback_url,
                json=[message.model_dump(mode="json")],  # always a list
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            replies = resp.json()
            if not isinstance(replies, list) or len(replies) == 0:
                logger.warning(
                    "[%s] %s — batch reply is not a non-empty list: %s",
                    self._session_id,
                    self.name,
                    replies,
                )
                return None
            return self._unwrap_reply(replies[0])
        except httpx.HTTPError as exc:
            logger.error(
                "[%s] %s — HTTP error calling callback %s: %s",
                self._session_id,
                self.name,
                self._callback_url,
                exc,
            )
            return None
        except Exception as exc:
            logger.error(
                "[%s] %s — unexpected error calling callback: %s",
                self._session_id,
                self.name,
                exc,
            )
            return None

    @staticmethod
    def _unwrap_reply(data: dict[str, Any]) -> dict[str, Any]:
        """Extract the inner payload from an SSTP-wrapped reply.

        If the agent replied with a full ``SSTPNegotiateMessage`` envelope
        (identified by a ``"payload"`` key whose value is a dict), return
        that inner payload.  Otherwise return *data* unchanged so plain-JSON
        replies continue to work.
        """
        inner = data.get("payload")
        if isinstance(inner, dict):
            return inner
        return data

    def _build_sstp_message(self, payload: dict[str, Any]) -> SSTPNegotiateMessage:
        """Construct and Pydantic-validate a full SSTPNegotiateMessage.

        The returned object is a real :class:`~protocol.sstp.SSTPNegotiateMessage`
        instance — not a raw dict — so the wire format is guaranteed to comply
        with the SSTP schema before anything hits the network.
        """
        payload_str = json.dumps(payload, sort_keys=True)
        payload_hash = hashlib.sha256(payload_str.encode()).hexdigest()
        message_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{self._session_id}:{self._participant_id}:{payload_hash}",
            )
        )
        return SSTPNegotiateMessage(
            kind="negotiate",
            message_id=message_id,
            dt_created=datetime.now(timezone.utc).isoformat(),
            origin=Origin(
                actor_id="negotiation-server",
                tenant_id=self._session_id,
            ),
            semantic_context=NegotiateSemanticContext(
                session_id=self._session_id,
                sao_state=None,
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

    def _issue_names(self) -> list[str]:
        """Return the ordered issue names from the NegMAS outcome space."""
        try:
            return [issue.name for issue in self.nmi.outcome_space.issues]
        except Exception:
            return []

    def _options_per_issue(self, issues: list[str]) -> dict[str, list[str]]:
        """Return ``{issue_name: [option, ...]}`` for all issues.

        Values are coerced to strings to match the wire format.  Falls back
        to an empty list for any issue whose values cannot be read (e.g. a
        continuous issue that has no discrete enumeration).
        """
        result: dict[str, list[str]] = {}
        try:
            for issue in self.nmi.outcome_space.issues:
                try:
                    result[issue.name] = [str(v) for v in issue.values]
                except Exception:
                    result[issue.name] = []
        except Exception:
            result = {name: [] for name in issues}
        return result

    def _n_steps(self) -> int | None:
        try:
            return self.nmi.n_steps
        except Exception:
            return None

    def _is_my_proposing_turn(self, state: SAOState) -> bool:
        """Return True only when this agent is the designated SAO proposer for *state*.

        NegMAS calls ``propose()`` on *all* negotiators at step 0 to seed their
        opening offers before the alternation loop begins.  We use each agent's
        position in ``nmi.negotiator_ids`` to decide whose turn it really is::

            expected_proposer = nmi.negotiator_ids[state.step % n_negotiators]

        This means only the "true" proposer receives ``can_counter_offer: true``;
        the other agent's step-0 ``propose()`` call yields ``can_counter_offer: false``.
        """
        try:
            ids = list(self.nmi.negotiator_ids)
            return ids[state.step % len(ids)] == self.id
        except Exception:
            return True  # safe fallback: treat every propose() as a real turn

    def _resolve_proposer(self, state: SAOState, source: str | None) -> str:
        """Best-effort: return the proposer's participant id from the source hint."""
        return source or ""

    def _tuple_to_dict(
        self, outcome: tuple | dict | None, issues: list[str]
    ) -> dict[str, str] | None:
        """Convert a NegMAS tuple outcome to ``{issue: value}``."""
        if outcome is None:
            return None
        if isinstance(outcome, dict):
            return {k: str(v) for k, v in outcome.items()}
        return {issues[i]: str(v) for i, v in enumerate(outcome)}

    def _dict_to_outcome(self, offer_dict: dict[str, str], issues: list[str]) -> tuple:
        """Convert an agent's ``{issue: value}`` reply to a NegMAS tuple."""
        return tuple(offer_dict[issue] for issue in issues)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_response_type(action: str) -> ResponseType:
    """Map an agent's string action to a NegMAS ``ResponseType``."""
    action = (action or "").strip().lower()
    if action == "accept":
        return ResponseType.ACCEPT_OFFER
    if action == "end":
        return ResponseType.END_NEGOTIATION
    return ResponseType.REJECT_OFFER
