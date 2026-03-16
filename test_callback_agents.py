"""test_callback_agents.py — Three local agent servers that respond to SSTPCallbackNegotiator.

Each agent runs as a tiny FastAPI server in a background thread.
The negotiation server (port 8089) calls back all agents on every NegMAS round.

Usage:
    # Terminal 1 — start the negotiation server:
    cd semantic-negotiation-agent
    poetry run uvicorn app.main:app --host 0.0.0.0 --port 8089

    # Terminal 2 — run this script:
    poetry run python test_callback_agents.py

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │  This script                                             │
    │                                                          │
    │  Agent A server :8091  ◄──── SSTPNegotiateMessage ────┐  │
    │  Agent B server :8092  ◄──── SSTPNegotiateMessage ───┐│  │
    │                                                       ││  │
    │  ─── POST /negotiate/initiate ───────────────────►   ││  │
    │       { agents: [{callback_url: :8091}, {:8092}] }   ││  │
    │                                                       ││  │
    │  Negotiation server :8089 (SAO mechanism)  ──────────┘│  │
    │                              ─────────────────────────┘  │
    └─────────────────────────────────────────────────────────┘

Agent decision logic (fully local — server never sees preferences):
    - Agent A  prefers CHEAP options  (index 0 in each issue list)
               simple linear concession: moves toward middle over rounds
    - Agent B  prefers PREMIUM options (last index in each issue list)
               NegMAS-style Boulware time-based concession:
               aspiration(t) = 1 - t^exponent  (exponent > 1 → Boulware)
               • propose: target the outcome whose utility ≈ aspiration(t)
               • respond: accept if utility(offer) ≥ aspiration(t)
    - Agent C  balanced/compromise — concedes from cheap side but with
               lower reservation (accepts more readily than A or B)
    - Propose: walk from preferred option toward middle as rounds progress
    - Respond: accept if utility(offer) >= threshold, else reject

Note: agents have NO prior knowledge of issues or options.  Every message
from the negotiation server includes ``issues`` and ``options_per_issue``
(added by SSTPCallbackNegotiator) and the agents use those fields directly.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import yaml
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Add workspace root so protocol.sstp is importable
_workspace_root = str(Path(__file__).resolve().parent)
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

from protocol.sstp import SSTPNegotiateMessage  # noqa: E402
from protocol.sstp._base import Origin, PolicyLabels, Provenance  # noqa: E402
from protocol.sstp.negotiate import NegotiateSemanticContext  # noqa: E402
from protocol.sstp.negmas_sao import ResponseType, SAOResponse, SAOState  # noqa: E402

NEG_SERVER = "http://localhost:8089"
AGENT_PORT = 8091  # single shared server — all agents are reachable here

# When webhook mode is active, the negotiation server POSTs the final result
# to this URL instead of blocking the initiate request.
_WEBHOOK_CALLBACK_PATH = "/negotiate/result"


# ── filesystem helpers ─────────────────────────────────────────────────────


def _slug(name: str) -> str:
    """Convert a display name to a lowercase, underscore-separated filesystem slug."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _save_json(path: Path, data: dict[str, Any]) -> None:
    """Write *data* as indented JSON to *path*, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _build_sstp_reply(
    session_id: str,
    agent_name: str,
    reply_payload: dict[str, Any],
    sao_response: SAOResponse | None = None,
    sao_state: SAOState | None = None,
) -> dict[str, Any]:
    """Wrap an agent's reply payload in a full SSTPNegotiateMessage envelope.

    The ``reply_payload`` becomes ``payload`` inside the message.  For a
    propose turn it should be ``{"action": "counter_offer", "offer": {...}}``;
    for a respond turn ``{"action": "accept" | "reject"}``.
    ``sao_response`` encodes the agent's SAO-level decision so the server can
    read the accept/reject/counter from the structured field without parsing
    the raw payload.
    """
    payload_str = json.dumps(reply_payload, sort_keys=True)
    payload_hash = hashlib.sha256(payload_str.encode()).hexdigest()
    message_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{session_id}:{_slug(agent_name)}:{payload_hash}",
        )
    )
    msg = SSTPNegotiateMessage(
        kind="negotiate",
        message_id=message_id,
        dt_created=datetime.now(timezone.utc).isoformat(),
        origin=Origin(
            actor_id=_slug(agent_name),
            tenant_id=session_id,
        ),
        semantic_context=NegotiateSemanticContext(
            session_id=session_id,
            sao_state=sao_state,
            sao_response=sao_response,
        ),
        payload_hash=payload_hash,
        policy_labels=PolicyLabels(
            sensitivity="internal",
            propagation="restricted",
            retention_policy="default",
        ),
        provenance=Provenance(sources=[], transforms=[]),
        payload=reply_payload,
    )
    return msg.model_dump(mode="json")


# ── agent decision engine ──────────────────────────────────────────────────


class LocalAgent:
    """Stateless decision engine for a callback agent.

    Preferences are private — never sent to the negotiation server.

    Args:
        name: Display name (logged in output).
        prefer_low: If True, prefer index-0 options (cheap side).
                    If False, prefer last-index options (premium side).
        accept_threshold: Minimum utility to accept an offer (0–1).
    """

    def __init__(
        self, name: str, prefer_low: bool, accept_threshold: float = 0.35
    ) -> None:
        self.name = name
        self.prefer_low = prefer_low
        self.accept_threshold = accept_threshold
        # Preferences are built lazily from the first payload that carries
        # options_per_issue — the agent has no prior knowledge of the space.
        self._prefs: dict[str, dict[str, float]] = {}

    def _build_prefs(
        self, options_per_issue: dict[str, list[str]]
    ) -> dict[str, dict[str, float]]:
        """Derive a utility map from the wire-format options list."""
        prefs: dict[str, dict[str, float]] = {}
        for issue, opts in options_per_issue.items():
            n = len(opts)
            denom = max(n - 1, 1)
            if self.prefer_low:
                prefs[issue] = {
                    o: round(1.0 - i / denom, 3) for i, o in enumerate(opts)
                }
            else:
                prefs[issue] = {o: round(i / denom, 3) for i, o in enumerate(opts)}
        return prefs

    def _ensure_prefs(
        self, options_per_issue: dict[str, list[str]]
    ) -> dict[str, dict[str, float]]:
        """Return the cached preference map, rebuilding if issues changed."""
        if set(options_per_issue.keys()) != set(self._prefs.keys()):
            self._prefs = self._build_prefs(options_per_issue)
        return self._prefs

    def utility(
        self, offer: dict[str, str], options_per_issue: dict[str, list[str]]
    ) -> float:
        """Compute mean utility for an offer, deriving the scale from options_per_issue."""
        prefs = self._ensure_prefs(options_per_issue)
        known = [issue for issue in offer if issue in prefs]
        if not known:
            return 0.0
        return sum(prefs[issue].get(offer[issue], 0.0) for issue in known) / len(known)

    def decide_propose(
        self,
        round_num: int,
        n_steps: int,
        options_per_issue: dict[str, list[str]],
    ) -> dict[str, str]:
        """Generate a counter-offer using the negotiation space from the payload.

        Starts at the preferred extreme, concedes toward the middle
        as rounds progress.  Returns ``(offer, None)`` for API compatibility
        with :class:`NegMASConcessionAgent` which returns ``(offer, aspiration)``.
        """
        progress = round_num / max(n_steps, 1)  # 0.0 → 1.0
        offer: dict[str, str] = {}
        for issue, opts in options_per_issue.items():
            n = len(opts)
            if self.prefer_low:
                idx = int(progress * (n // 2))
            else:
                idx = (n - 1) - int(progress * (n // 2))
            offer[issue] = opts[max(0, min(n - 1, idx))]
        return offer, None

    def decide_respond(
        self,
        offer: dict[str, str],
        round_num: int,
        n_steps: int,
        options_per_issue: dict[str, list[str]],
    ) -> str:
        """Return 'accept' or 'reject' for the incoming offer."""
        u = self.utility(offer, options_per_issue)
        # Threshold shrinks as rounds progress (willing to accept less over time)
        threshold = max(
            self.accept_threshold, 0.8 * (1.0 - round_num / max(n_steps, 1))
        )
        decision = "accept" if u >= threshold else "reject"
        print(
            f"  [{self.name}] respond  round={round_num}  utility={u:.3f}"
            f"  threshold={threshold:.3f}  → {decision}",
            flush=True,
        )
        return decision


class NegMASConcessionAgent(LocalAgent):
    """Agent that uses the NegMAS ``AspirationNegotiator`` concession formula.

    At relative time ``t = round / n_steps`` (0 → 1), the aspiration level is::

        aspiration(t) = 1 - t ** exponent

    Common presets (same as NegMAS time-based negotiators):

    - **Boulware**  ``exponent > 1``  — holds ideal offer long, concedes hard at end.
    - **Linear**    ``exponent = 1``  — steady linear concession.
    - **Conceder**  ``exponent < 1``  — concedes quickly early, then plateaus.

    **Propose**: enumerate every possible outcome, pick the one with utility
    closest to (but ≥) ``aspiration(t)``.  Falls back to the best outcome if
    none qualifies.

    **Respond**: accept if ``utility(offer) ≥ aspiration(t)``.

    Args:
        name: Display name.
        prefer_low: Determines which end of each option list is preferred.
        exponent: Concession exponent.  Default ``2.0`` (Boulware).
        min_reservation: Hard floor — never accept below this utility (0–1).
    """

    def __init__(
        self,
        name: str,
        prefer_low: bool,
        exponent: float = 2.0,
        min_reservation: float = 0.0,
    ) -> None:
        super().__init__(name, prefer_low, accept_threshold=min_reservation)
        self.exponent = exponent
        self.min_reservation = min_reservation

    def _aspiration(self, t: float) -> float:
        """NegMAS aspiration curve.  Returns 1.0 at t=0, approaches 0 at t=1."""
        return max(self.min_reservation, 1.0 - (t**self.exponent))

    def _all_outcomes_sorted(
        self, options_per_issue: dict[str, list[str]]
    ) -> list[tuple[dict, float]]:
        """Return all (offer, utility) pairs sorted by utility descending."""
        prefs = self._ensure_prefs(options_per_issue)
        issues = list(options_per_issue.keys())
        results: list[tuple[dict, float]] = []
        for combo in itertools.product(*[options_per_issue[i] for i in issues]):
            offer = dict(zip(issues, combo))
            u = sum(prefs[issue].get(offer[issue], 0.0) for issue in issues) / len(
                issues
            )
            results.append((offer, round(u, 4)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def decide_propose(
        self,
        round_num: int,
        n_steps: int,
        options_per_issue: dict[str, list[str]],
    ) -> dict[str, str]:
        """Propose the outcome with utility closest to (and ≥) aspiration(t)."""
        t = round_num / max(n_steps, 1)
        asp = self._aspiration(t)
        outcomes = self._all_outcomes_sorted(options_per_issue)
        # Walk from best → worst; return the last one that is still >= asp.
        # That is: the minimum-utility outcome that still satisfies the aspiration.
        best_qualifying: dict[str, str] = outcomes[0][0]  # fallback = ideal offer
        for offer, u in outcomes:
            if u >= asp:
                best_qualifying = (
                    offer  # keep updating → we want the minimum qualifying
                )
            else:
                break  # sorted desc — no further outcome can qualify
        return best_qualifying, asp

    def decide_respond(
        self,
        offer: dict[str, str],
        round_num: int,
        n_steps: int,
        options_per_issue: dict[str, list[str]],
    ) -> str:
        """Accept if utility(offer) ≥ aspiration(t), otherwise reject."""
        u = self.utility(offer, options_per_issue)
        t = round_num / max(n_steps, 1)
        asp = self._aspiration(t)
        decision = "accept" if u >= asp else "reject"
        print(
            f"  [{self.name}] respond  round={round_num}  utility={u:.3f}"
            f"  aspiration={asp:.3f}  → {decision}",
            flush=True,
        )
        return decision


# ── FastAPI mini-app factory ───────────────────────────────────────────────


def make_agent_app(
    agents: dict[str, LocalAgent],
    trace_state: dict[str, Path],
    webhook_results: dict[str, Any] | None = None,
    webhook_events: dict[str, threading.Event] | None = None,
) -> FastAPI:
    """Return a FastAPI app whose single POST /decide endpoint handles ALL agents.

    When *webhook_results* and *webhook_events* are provided, a
    ``POST /negotiate/result`` endpoint is also registered.  The negotiation
    server will POST the final ``InitiateResponse`` SSTP envelope there when
    ``payload.result_callback_url`` is set in the initiate request.

    The endpoint receives and returns **``List[SSTPNegotiateMessage]``**.  The
    :class:`~app.agent.batch_callback_runner.BatchCallbackRunner` sends the whole
    round's batch in one call — one message per participant.  Each message carries
    ``payload.participant_id`` which is used to dispatch to the right
    :class:`LocalAgent` instance.

    ``agents`` is keyed by participant id (as declared in the initiate payload).

    ``trace_state`` is a mutable dict with key ``"trace_dir"`` (a :class:`Path`).
    The caller updates ``trace_state["trace_dir"]`` before each mission so the
    same running server writes traces into the correct per-mission folder.

    Every non-shadow incoming message and the agent's reply are saved under
    ``<trace_dir>/round_{N:04d}/{action}__{agent_slug}__request.json`` and
    ``…__reply.json``.
    """
    app = FastAPI(title="Shared Agent Server")

    if webhook_results is not None and webhook_events is not None:

        @app.post(_WEBHOOK_CALLBACK_PATH)
        async def receive_result(request: Request) -> JSONResponse:
            """Receive the final negotiation result posted by the negotiation server."""
            body: dict[str, Any] = await request.json()
            payload = body.get("payload", {})
            session_id: str = payload.get("session_id") or (
                body.get("semantic_context", {}) or {}
            ).get("session_id", "unknown")
            webhook_results[session_id] = body
            ev = webhook_events.get(session_id)
            if ev is not None:
                ev.set()
            else:
                # store under a wildcard so callers can pick it up
                webhook_results["__latest__"] = body
                for ev in webhook_events.values():
                    ev.set()
            print(
                f"  [webhook] result received  session={session_id}"
                f"  status={payload.get('status')}",
                flush=True,
            )
            return JSONResponse({"status": "ok"})

    @app.post("/decide")
    async def decide(request: Request) -> JSONResponse:
        messages: list[dict[str, Any]] = await request.json()

        def _process_one(body: dict[str, Any]) -> dict[str, Any]:
            payload: dict[str, Any] = body.get("payload", {})
            participant_id: str = payload.get("participant_id", "")
            agent = agents.get(participant_id)
            if agent is None:
                # Fallback: try matching by name slug against any agent
                for pid, a in agents.items():
                    if _slug(a.name) in _slug(participant_id) or participant_id in pid:
                        agent = a
                        break
            if agent is None:
                # Last resort: use first agent
                agent = next(iter(agents.values()))

            action = payload.get("action")  # "propose" or "respond"
            round_num: int = payload.get("round", 1)
            n_steps: int = payload.get("n_steps") or 200
            _semantic_ctx: dict[str, Any] = body.get("semantic_context") or {}
            issues: list[str] = _semantic_ctx.get("issues") or []
            options_per_issue: dict[str, list[str]] = (
                _semantic_ctx.get("options_per_issue") or {}
            )
            session_id: str = _semantic_ctx.get("session_id") or "unknown-session"
            _sao_state_dict = _semantic_ctx.get("sao_state")
            incoming_sao_state: SAOState | None = (
                SAOState(**_sao_state_dict) if _sao_state_dict else None
            )
            # is_shadow_call=True (e.g. round 1 responders with no standing offer):
            # return a valid response but skip trace writing.
            is_shadow: bool = payload.get("is_shadow_call", False)

            slug = _slug(agent.name)
            round_dir = trace_state["trace_dir"] / f"round_{round_num:04d}"

            if not is_shadow:
                _save_json(round_dir / f"{action}__{slug}__request.json", body)

            if action == "propose":
                offer, aspiration = agent.decide_propose(
                    round_num, n_steps, options_per_issue
                )
                if not is_shadow:
                    asp_str = (
                        f"  aspiration={aspiration:.3f}"
                        if aspiration is not None
                        else ""
                    )
                    print(
                        f"  [{agent.name}] propose  round={round_num}{asp_str}  offer={offer}",
                        flush=True,
                    )
                reply_payload: dict[str, Any] = {
                    "action": "counter_offer",
                    "round": round_num,
                    "issues": issues,
                    "options_per_issue": options_per_issue,
                    "offer": offer,
                }
                # SAO-level response: proposing a counter-offer implicitly rejects the standing offer.
                sao_resp = SAOResponse(
                    response=ResponseType.REJECT_OFFER, outcome=offer
                )
                reply = _build_sstp_reply(
                    session_id,
                    agent.name,
                    reply_payload,
                    sao_response=sao_resp,
                    sao_state=incoming_sao_state,
                )
                if not is_shadow:
                    _save_json(round_dir / f"{action}__{slug}__reply.json", reply)
                return reply

            elif action == "respond":
                current_offer: dict[str, str] = payload.get("current_offer") or {}
                decision = agent.decide_respond(
                    current_offer, round_num, n_steps, options_per_issue
                )
                reply_payload = {
                    "action": decision,
                    "round": round_num,
                    "issues": issues,
                    "options_per_issue": options_per_issue,
                }
                sao_resp = SAOResponse(
                    response=(
                        ResponseType.ACCEPT_OFFER
                        if decision == "accept"
                        else ResponseType.REJECT_OFFER
                    ),
                    outcome=current_offer if decision == "accept" else None,
                )
                reply = _build_sstp_reply(
                    session_id,
                    agent.name,
                    reply_payload,
                    sao_response=sao_resp,
                    sao_state=incoming_sao_state,
                )
                if not is_shadow:
                    _save_json(round_dir / f"{action}__{slug}__reply.json", reply)
                return reply

            else:
                reply_payload = {"action": "reject", "round": round_num}
                sao_resp = SAOResponse(response=ResponseType.REJECT_OFFER)
                reply = _build_sstp_reply(
                    session_id,
                    agent.name,
                    reply_payload,
                    sao_response=sao_resp,
                    sao_state=incoming_sao_state,
                )
                if not is_shadow:
                    _save_json(round_dir / f"unknown__{slug}__reply.json", reply)
                return reply

        # Process all messages synchronously, POST the full decision list to
        # the negotiation server, and ONLY THEN return the ACK.
        # BatchCallbackRunner is waiting for the ACK; the guarantee is that by
        # the time the ACK arrives the decisions are already stored in
        # _DECISIONS — no threading.Event or polling needed.
        replies = [_process_one(msg) for msg in messages]
        try:
            httpx.post(
                f"{NEG_SERVER}/api/v1/negotiate/agents-decisions",
                json=replies,
                headers={"Content-Type": "application/json"},
                timeout=30.0,
            ).raise_for_status()
        except Exception as exc:
            print(
                f"[agent] failed to POST /negotiate/agents-decisions: {exc}", flush=True
            )

        return JSONResponse({"status": "ack"})

    return app


# ── server thread ──────────────────────────────────────────────────────────


def start_agent_server(
    agents: dict[str, LocalAgent],
    port: int,
    trace_state: dict,
    webhook_results: dict[str, Any] | None = None,
    webhook_events: dict[str, threading.Event] | None = None,
) -> threading.Thread:
    """Start the shared agent FastAPI server (all agents) in a daemon thread."""
    app = make_agent_app(agents, trace_state, webhook_results, webhook_events)
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)

    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    return t


def wait_for_server(port: int, retries: int = 20, delay: float = 0.3) -> None:
    """Block until the agent server is accepting connections."""
    for _ in range(retries):
        try:
            httpx.get(f"http://localhost:{port}/openapi.json", timeout=1.0)
            return
        except Exception:
            time.sleep(delay)
    raise RuntimeError(f"Agent server on port {port} did not start in time")


# ── mission definitions ───────────────────────────────────────────────────
#
# Missions are loaded from missions.yaml (next to this script).  Add or edit
# missions there — no changes to this file are required.
#
# Folder layout produced under neg_trace/:
#
#   neg_trace/
#     <YYYYMMDD_HHMMSS>/           ← one directory per run
#       project_contract/          ← one sub-folder per mission (slug of name)
#         00_initiate_request.json
#         round_0001/
#           propose__agent_a__request.json
#           propose__agent_a__reply.json
#           …
#         final_result.json
#       cloud_platform/
#         00_initiate_request.json
#         round_0001/
#         …
#         final_result.json

_MISSIONS_FILE = Path(__file__).resolve().parent / "missions.yaml"


def _load_missions(path: Path = _MISSIONS_FILE) -> list[dict[str, Any]]:
    """Load and return the missions list from *path* (a YAML file).

    The file must contain a top-level ``missions:`` key whose value is a
    sequence of mission dicts (see ``missions.yaml`` for the schema).
    """
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    missions = data.get("missions") or []
    if not missions:
        raise ValueError(f"No missions found in {path}")
    return missions


MISSIONS: list[dict[str, Any]] = _load_missions()


def _build_initiate_payload(mission: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Construct an SSTPNegotiateMessage initiate payload for *mission*.

    ``protocol``, ``version``, ``schema_id``, and ``schema_version`` are
    sourced from the canonical defaults in the protocol package.
    """
    mission_slug = _slug(mission["name"])
    return SSTPNegotiateMessage(
        kind="negotiate",
        message_id=f"init-{run_id}-{mission_slug}",
        dt_created=datetime.now(timezone.utc).isoformat(),
        origin=Origin(actor_id="test-runner", tenant_id="demo"),
        semantic_context=NegotiateSemanticContext(
            session_id=f"sess-{run_id}-{mission_slug}",
        ),
        payload_hash="0" * 64,
        policy_labels=PolicyLabels(
            sensitivity="internal",
            propagation="restricted",
            retention_policy="default",
        ),
        provenance=Provenance(sources=[], transforms=[]),
        payload={
            "content_text": mission["content_text"],
            "agents": [
                {
                    "id": "agent-a",
                    "name": "Agent A",
                    "callback_url": f"http://localhost:{AGENT_PORT}/decide",
                },
                {
                    "id": "agent-b",
                    "name": "Agent B",
                    "callback_url": f"http://localhost:{AGENT_PORT}/decide",
                },
                {
                    "id": "agent-c",
                    "name": "Agent C",
                    "callback_url": f"http://localhost:{AGENT_PORT}/decide",
                },
            ],
            "n_steps": mission["n_steps"],
        },
    ).model_dump(mode="json")


def _add_webhook_url(payload: dict[str, Any], callback_url: str) -> dict[str, Any]:
    """Return a copy of *payload* with ``result_callback_url`` injected into
    the inner ``payload`` dict so the negotiation server switches to async mode."""
    payload = dict(payload)
    payload["payload"] = dict(payload["payload"])
    payload["payload"]["result_callback_url"] = callback_url
    return payload


def run(
    neg_server: str, missions_file: Path | None = None, webhook: bool = False
) -> None:
    # ── load missions ─────────────────────────────────────────────────────
    missions = _load_missions(missions_file) if missions_file else MISSIONS

    # ── unique id for this run ────────────────────────────────────────────
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = run_timestamp  # used in session_id / message_id
    run_trace_dir = Path("neg_trace") / run_timestamp
    run_trace_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run trace root: {run_trace_dir.resolve()}")
    print(f"Missions file : {(missions_file or _MISSIONS_FILE).resolve()}")
    print(f"Missions to negotiate: {len(missions)}")
    for m in missions:
        print(f"  • {m['name']}")
    print()

    # ── shared agent instances (preferences reset per mission) ─────────────
    agent_a = NegMASConcessionAgent(
        "Agent A", prefer_low=True, exponent=1.5, min_reservation=0.2
    )
    agent_b = NegMASConcessionAgent(
        "Agent B", prefer_low=False, exponent=3.0, min_reservation=0.2
    )
    agent_c = NegMASConcessionAgent(
        "Agent C", prefer_low=True, exponent=2.0, min_reservation=0.1
    )
    agents: dict[str, LocalAgent] = {
        "agent-a": agent_a,
        "agent-b": agent_b,
        "agent-c": agent_c,
    }

    # ── mutable trace pointer — updated before each mission ───────────────
    trace_state: dict[str, Path] = {"trace_dir": run_trace_dir}

    # ── optional webhook state ─────────────────────────────────────────────
    webhook_results: dict[str, Any] = {}
    webhook_events: dict[str, threading.Event] = {}

    print(f"Starting shared agent server on :{AGENT_PORT}…")
    start_agent_server(
        agents,
        AGENT_PORT,
        trace_state,
        webhook_results if webhook else None,
        webhook_events if webhook else None,
    )
    wait_for_server(AGENT_PORT)
    print("Agent server is up.", "(webhook listener active)" if webhook else "", "\n")

    # ── verify negotiation server is reachable ─────────────────────────────
    try:
        httpx.get(f"{neg_server}/openapi.json", timeout=3.0).raise_for_status()
    except Exception as exc:
        print(f"ERROR: negotiation server at {neg_server} is not reachable: {exc}")
        print(
            "Start it with:  cd semantic-negotiation-agent && poetry run uvicorn app.main:app --host 0.0.0.0 --port 8089"
        )
        sys.exit(1)

    # ── per-run summary log (written to run root at the end) ──────────────
    run_log: list[dict[str, Any]] = []

    # ── iterate over missions ──────────────────────────────────────────────
    for idx, mission in enumerate(missions, start=1):
        _mission_start = time.monotonic()
        mission_slug = _slug(mission["name"])
        mission_trace_dir = run_trace_dir / mission_slug
        mission_trace_dir.mkdir(parents=True, exist_ok=True)

        # Point the running agent server at this mission's trace folder.
        trace_state["trace_dir"] = mission_trace_dir

        # Reset cached preferences — issue space may differ between missions.
        for a in agents.values():
            a._prefs = {}

        print(f"{'=' * 62}")
        print(f"  Mission {idx}/{len(missions)}: {mission['name']}")
        print(f"  Trace  : {mission_trace_dir.resolve()}")
        print(f"{'=' * 62}\n")

        initiate_payload = _build_initiate_payload(mission, run_id)

        session_id: str = (initiate_payload.get("semantic_context", {}) or {}).get(
            "session_id", "unknown"
        )

        if webhook:
            callback_url = f"http://localhost:{AGENT_PORT}{_WEBHOOK_CALLBACK_PATH}"
            initiate_payload = _add_webhook_url(initiate_payload, callback_url)
            ev = threading.Event()
            webhook_events[session_id] = ev

        _save_json(mission_trace_dir / "00_initiate_request.json", initiate_payload)

        print(f"POST {neg_server}/api/v1/negotiate/initiate …")
        resp = httpx.post(
            f"{neg_server}/api/v1/negotiate/initiate",
            json=initiate_payload,
            timeout=30.0 if webhook else 120.0,
        )

        print(f"HTTP {resp.status_code}")

        if webhook and resp.status_code == 202:
            print(f"  → 202 Accepted  (session={session_id})  waiting for callback…")
            delivered = ev.wait(timeout=300.0)  # up to 5 minutes
            if not delivered:
                print(
                    f"  WARNING: callback not received within timeout for session {session_id}"
                )
                result = {"error": "callback_timeout", "session_id": session_id}
            else:
                result = webhook_results.get(session_id) or webhook_results.get(
                    "__latest__", {}
                )
            webhook_events.pop(session_id, None)
        else:
            try:
                result = resp.json()
            except Exception:
                print(resp.text)
                result = {}

        _save_json(mission_trace_dir / "final_result.json", result)
        print(json.dumps(result, indent=2))

        # ── append summary entry to run log ───────────────────────────────
        _elapsed = round(time.monotonic() - _mission_start, 1)
        _payload = result.get("payload") or {}
        _trace = _payload.get("trace") or {}
        run_log.append(
            {
                # ── mission metadata ──────────────────────────────────────────
                "mission": mission["name"],
                "duration_s": _elapsed,
                "mode": "webhook" if webhook else "sync",
                "trace_dir": str(mission_trace_dir.resolve()),
                # ── SSTP envelope (top-level SSTPCommitMessage fields) ────────
                "kind": result.get("kind"),
                "protocol": result.get("protocol"),
                "version": result.get("version"),
                "message_id": result.get("message_id"),
                "dt_created": result.get("dt_created"),
                "origin": result.get("origin"),
                "semantic_context": result.get("semantic_context"),
                "payload_hash": result.get("payload_hash"),
                "policy_labels": result.get("policy_labels"),
                "provenance": result.get("provenance"),
                # ── SSTPCommitMessage-specific fields ─────────────────────────
                "state_object_id": result.get("state_object_id"),
                "parent_ids": result.get("parent_ids"),
                "logical_clock": result.get("logical_clock"),
                "confidence_score": result.get("confidence_score"),
                "risk_score": result.get("risk_score"),
                "ttl_seconds": result.get("ttl_seconds"),
                "merge_strategy": result.get("merge_strategy"),
                "payload_refs": result.get("payload_refs"),
                # ── negotiation outcome (from payload) ────────────────────────
                "session_id": session_id,
                "status": _payload.get("status"),
                "total_rounds": _payload.get("total_rounds"),
                "timedout": _trace.get("timedout"),
                "broken": _trace.get("broken"),
                "final_agreement": _trace.get("final_agreement"),
            }
        )

        print(f"\nMission {idx} trace saved to: {mission_trace_dir.resolve()}\n")

    # ── write run-level summary ────────────────────────────────────────────
    run_log_path = run_trace_dir / "run_log.json"
    _save_json(run_log_path, {"run_id": run_timestamp, "missions": run_log})
    print(f"Run log written : {run_log_path.resolve()}")
    print(
        f"All {len(missions)} missions complete.  Run trace root: {run_trace_dir.resolve()}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-agent callback negotiation test")
    parser.add_argument(
        "--neg-server",
        default=NEG_SERVER,
        help=f"Base URL of the negotiation server (default: {NEG_SERVER})",
    )
    parser.add_argument(
        "--missions-file",
        default=None,
        metavar="PATH",
        help=f"Path to a YAML missions file (default: missions.yaml next to this script)",
    )
    parser.add_argument(
        "--webhook",
        action="store_true",
        default=False,
        help=(
            "Use webhook (async) mode: send result_callback_url in the initiate payload, "
            "get immediate 202, then wait for the negotiation result to be POSTed back "
            "to the agent server's /negotiate/result endpoint."
        ),
    )
    args = parser.parse_args()
    run(
        args.neg_server,
        Path(args.missions_file) if args.missions_file else None,
        webhook=args.webhook,
    )
