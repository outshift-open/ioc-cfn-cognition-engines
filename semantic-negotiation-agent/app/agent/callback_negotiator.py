"""SSTPCallbackNegotiator — a NegMAS SAO negotiator that delegates every
decision to an external agent via the SSTP protocol.

Instead of running a local utility curve (like BoulwareTBNegotiator), this
negotiator POSTs an ``SSTPNegotiateMessage`` to the participant's
``callback_url`` on every ``propose`` and ``respond`` call, then waits for
the agent's JSON reply.

Wire format
-----------
**Server → Agent** (both propose and respond requests):

    POST <callback_url>
    Content-Type: application/json

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
        "history": [ {"round":1,"proposer_id":"..","offer":{..}}, ... ]

        // ---- for respond requests ----
        "action": "respond",
        "round": 3,
        "current_offer": { "budget": "medium", "timeline": "standard", ... },
        "proposer_id": "<id of the agent who made this offer>",
        "history": [ ... ]
      }
    }

**Agent → Server** (reply body, plain JSON — no envelope needed):

    // reply to propose
    { "offer": { "budget": "low", "timeline": "short", ... } }

    // reply to respond
    { "action": "accept" }    // or "reject" or "end"

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

from protocol.sstp import SSTPNegotiateMessage  # noqa: E402
from protocol.sstp._base import Origin, PolicyLabels, Provenance  # noqa: E402
from protocol.sstp.negotiate import NegotiateSemanticContext  # noqa: E402

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

        Sends the full negotiation history so the agent has all context.
        Expects ``{ "offer": { issue: value, ... } }`` in the reply.

        Returns ``None`` if the call fails (NegMAS treats None as "skip turn").
        """
        issues = self._issue_names()
        payload: dict[str, Any] = {
            "action": "propose",
            "round": state.step + 1,  # 1-based for humans
            "history": self._serialise_history(state, issues),
            "n_steps": self._n_steps(),
        }
        reply = self._call_agent(payload)
        if reply is None:
            logger.warning("[%s] %s — propose callback failed, returning None", self._session_id, self.name)
            return None

        offer_dict = reply.get("offer")
        if not isinstance(offer_dict, dict):
            logger.warning(
                "[%s] %s — propose reply missing 'offer' key: %s",
                self._session_id, self.name, reply,
            )
            return None

        # Convert the agent's dict offer to the tuple form NegMAS expects
        try:
            return self._dict_to_outcome(offer_dict, issues)
        except Exception as exc:
            logger.warning("[%s] %s — propose outcome conversion failed: %s", self._session_id, self.name, exc)
            return None

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Ask the external agent to respond to the current offer.

        Sends the current offer and full history so the agent has all context.
        Expects ``{ "action": "accept" | "reject" | "end" }`` in the reply.

        Returns ``ResponseType.REJECT_OFFER`` if the call fails.
        """
        issues = self._issue_names()
        current_offer = (
            self._tuple_to_dict(state.current_offer, issues)
            if state.current_offer is not None
            else None
        )
        proposer_id = self._resolve_proposer(state, source)
        payload: dict[str, Any] = {
            "action": "respond",
            "round": state.step + 1,
            "current_offer": current_offer,
            "proposer_id": proposer_id,
            "history": self._serialise_history(state, issues),
            "n_steps": self._n_steps(),
        }
        reply = self._call_agent(payload)
        if reply is None:
            logger.warning("[%s] %s — respond callback failed, defaulting to reject", self._session_id, self.name)
            return ResponseType.REJECT_OFFER

        action = reply.get("action", "reject")
        return _parse_response_type(action)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_agent(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Build a validated SSTPNegotiateMessage and POST it to the agent's callback."""
        message = self._build_sstp_message(payload)
        try:
            resp = self._http.post(
                self._callback_url,
                json=message.model_dump(mode="json"),
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            logger.error(
                "[%s] %s — HTTP error calling callback %s: %s",
                self._session_id, self.name, self._callback_url, exc,
            )
            return None
        except Exception as exc:
            logger.error(
                "[%s] %s — unexpected error calling callback: %s",
                self._session_id, self.name, exc,
            )
            return None

    def _build_sstp_message(self, payload: dict[str, Any]) -> SSTPNegotiateMessage:
        """Construct and Pydantic-validate a full SSTPNegotiateMessage.

        The returned object is a real :class:`~protocol.sstp.SSTPNegotiateMessage`
        instance — not a raw dict — so the wire format is guaranteed to comply
        with the SSTP schema before anything hits the network.
        """
        payload_str = json.dumps(payload, sort_keys=True)
        payload_hash = hashlib.sha256(payload_str.encode()).hexdigest()
        message_id = str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{self._session_id}:{self._participant_id}:{payload_hash}",
        ))
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

    def _n_steps(self) -> int | None:
        try:
            return self.nmi.n_steps
        except Exception:
            return None

    def _resolve_proposer(self, state: SAOState, source: str | None) -> str:
        """Best-effort: return the proposer's participant id from the source hint."""
        return source or ""

    def _tuple_to_dict(self, outcome: tuple | dict | None, issues: list[str]) -> dict[str, str] | None:
        """Convert a NegMAS tuple outcome to ``{issue: value}``."""
        if outcome is None:
            return None
        if isinstance(outcome, dict):
            return {k: str(v) for k, v in outcome.items()}
        return {issues[i]: str(v) for i, v in enumerate(outcome)}

    def _dict_to_outcome(self, offer_dict: dict[str, str], issues: list[str]) -> tuple:
        """Convert an agent's ``{issue: value}`` reply to a NegMAS tuple."""
        return tuple(offer_dict[issue] for issue in issues)

    def _serialise_history(self, state: SAOState, issues: list[str]) -> list[dict]:
        """Serialise the current state history for the outgoing message."""
        rounds: list[dict] = []
        try:
            for idx, entry in enumerate(state.history or []):
                # state.history entries are (negotiator_name, offer_tuple)
                if hasattr(entry, "offer"):
                    name = getattr(entry, "current_proposer", "unknown")
                    offer = entry.offer
                else:
                    # Older NegMAS — entry may be a tuple (name, offer)
                    name, offer = (entry[0], entry[1]) if len(entry) >= 2 else ("unknown", None)
                rounds.append({
                    "round": idx + 1,
                    "proposer_id": str(name),
                    "offer": self._tuple_to_dict(offer, issues) or {},
                })
        except Exception as exc:
            logger.debug("[%s] history serialisation warning: %s", self._session_id, exc)
        return rounds


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
