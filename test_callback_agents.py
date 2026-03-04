"""test_callback_agents.py — Two local agent servers that respond to SSTPCallbackNegotiator.

Each agent runs as a tiny FastAPI server in a background thread.
The negotiation server (port 8089) calls back both agents on every NegMAS round.

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
    - Agent B  prefers PREMIUM options (last index in each issue list)
    - Each concedes linearly over rounds: at round N out of max,
      utility threshold = max(0.1, 0.8 × (1 - N/max))
    - Propose: walk from preferred option toward middle as rounds progress
    - Respond: accept if utility(offer) >= threshold, else reject
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ── stub issue/option map (mirrors options_generation.py stub) ─────────────

KNOWN_OPTIONS: dict[str, list[str]] = {
    "budget":   ["minimal", "low", "medium", "high", "uncapped"],
    "timeline": ["express", "short", "standard", "extended", "long"],
    "scope":    ["core", "standard", "extended", "full"],
    "quality":  ["basic", "standard", "premium"],
}

NEG_SERVER = "http://localhost:8089"
AGENT_A_PORT = 8091
AGENT_B_PORT = 8092


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

    def __init__(self, name: str, prefer_low: bool, accept_threshold: float = 0.35) -> None:
        self.name = name
        self.prefer_low = prefer_low
        self.accept_threshold = accept_threshold

        # Build private utility function: {issue: {option: value}}
        self._prefs: dict[str, dict[str, float]] = {}
        for issue, opts in KNOWN_OPTIONS.items():
            n = len(opts)
            denom = max(n - 1, 1)
            if prefer_low:
                self._prefs[issue] = {o: round(1.0 - i / denom, 3) for i, o in enumerate(opts)}
            else:
                self._prefs[issue] = {o: round(i / denom, 3) for i, o in enumerate(opts)}

    def utility(self, offer: dict[str, str]) -> float:
        """Compute mean utility for an offer across all known issues."""
        known = [issue for issue in offer if issue in self._prefs]
        if not known:
            return 0.0
        return sum(self._prefs[issue].get(offer[issue], 0.0) for issue in known) / len(known)

    def decide_propose(self, round_num: int, n_steps: int) -> dict[str, str]:
        """Generate a counter-offer.

        Starts at the preferred extreme, concedes toward the middle
        as rounds progress.
        """
        progress = round_num / max(n_steps, 1)   # 0.0 → 1.0
        offer: dict[str, str] = {}
        for issue, opts in KNOWN_OPTIONS.items():
            n = len(opts)
            if self.prefer_low:
                # Start at index 0; concede toward middle (n//2) with progress
                idx = int(progress * (n // 2))
            else:
                # Start at last index; concede toward middle
                idx = (n - 1) - int(progress * (n // 2))
            offer[issue] = opts[max(0, min(n - 1, idx))]
        return offer

    def decide_respond(self, offer: dict[str, str], round_num: int, n_steps: int) -> str:
        """Return 'accept' or 'reject' for the incoming offer."""
        u = self.utility(offer)
        # Threshold shrinks as rounds progress (willing to accept less over time)
        threshold = max(self.accept_threshold, 0.8 * (1.0 - round_num / max(n_steps, 1)))
        decision = "accept" if u >= threshold else "reject"
        print(
            f"  [{self.name}] respond  round={round_num}  utility={u:.3f}"
            f"  threshold={threshold:.3f}  → {decision}",
            flush=True,
        )
        return decision


# ── FastAPI mini-app factory ───────────────────────────────────────────────

def make_agent_app(agent: LocalAgent) -> FastAPI:
    """Return a minimal FastAPI app with a single POST /decide endpoint."""
    app = FastAPI(title=f"Agent: {agent.name}")

    @app.post("/decide")
    async def decide(request: Request) -> JSONResponse:
        body: dict[str, Any] = await request.json()

        # body is a full SSTPNegotiateMessage dict
        payload: dict[str, Any] = body.get("payload", {})
        action = payload.get("action")          # "propose" or "respond"
        round_num: int = payload.get("round", 1)
        n_steps: int = payload.get("n_steps") or 200

        if action == "propose":
            offer = agent.decide_propose(round_num, n_steps)
            print(
                f"  [{agent.name}] propose  round={round_num}  offer={offer}",
                flush=True,
            )
            return JSONResponse({"offer": offer})

        elif action == "respond":
            current_offer: dict[str, str] = payload.get("current_offer") or {}
            decision = agent.decide_respond(current_offer, round_num, n_steps)
            return JSONResponse({"action": decision})

        else:
            # Unknown action — reject safely
            return JSONResponse({"action": "reject"})

    return app


# ── server thread ──────────────────────────────────────────────────────────

def start_agent_server(agent: LocalAgent, port: int) -> threading.Thread:
    """Start the agent's FastAPI server in a daemon thread."""
    app = make_agent_app(agent)
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


# ── negotiation initiate ───────────────────────────────────────────────────

INITIATE_PAYLOAD = {
    "kind": "negotiate",
    "protocol": "SSTP",
    "version": "0",
    "message_id": "init-callback-demo-001",
    "dt_created": "2026-03-04T10:00:00Z",
    "origin": {"actor_id": "test-runner", "tenant_id": "demo"},
    "semantic_context": {
        "schema_id": "urn:ioc:schema:negotiate:negmas-sao:v1",
        "schema_version": "1.0",
        "session_id": "sess-callback-demo-001",
        "sao_state": None,
    },
    "payload_hash": "0" * 64,
    "policy_labels": {
        "sensitivity": "internal",
        "propagation": "restricted",
        "retention_policy": "default",
    },
    "provenance": {"sources": [], "transforms": []},
    "payload": {
        "content_text": "We need to agree on budget, timeline, scope, and quality for the project.",
        "agents": [
            {
                "id": "agent-a",
                "name": "Agent A",
                "callback_url": f"http://localhost:{AGENT_A_PORT}/decide",
            },
            {
                "id": "agent-b",
                "name": "Agent B",
                "callback_url": f"http://localhost:{AGENT_B_PORT}/decide",
            },
        ],
        "n_steps": 60,
    },
}


def run(neg_server: str) -> None:
    # ── start local agent servers ──────────────────────────────────────────
    agent_a = LocalAgent("Agent A", prefer_low=True,  accept_threshold=0.3)
    agent_b = LocalAgent("Agent B", prefer_low=False, accept_threshold=0.3)

    print(f"Starting Agent A server on :{AGENT_A_PORT}…")
    start_agent_server(agent_a, AGENT_A_PORT)
    print(f"Starting Agent B server on :{AGENT_B_PORT}…")
    start_agent_server(agent_b, AGENT_B_PORT)

    wait_for_server(AGENT_A_PORT)
    wait_for_server(AGENT_B_PORT)
    print("Both agent servers are up.\n")

    # ── verify negotiation server is reachable ─────────────────────────────
    try:
        httpx.get(f"{neg_server}/openapi.json", timeout=3.0).raise_for_status()
    except Exception as exc:
        print(f"ERROR: negotiation server at {neg_server} is not reachable: {exc}")
        print("Start it with:  cd semantic-negotiation-agent && poetry run uvicorn app.main:app --host 0.0.0.0 --port 8089")
        sys.exit(1)

    # ── fire initiate ──────────────────────────────────────────────────────
    print(f"POST {neg_server}/api/v1/negotiate/initiate …\n")
    resp = httpx.post(
        f"{neg_server}/api/v1/negotiate/initiate",
        json=INITIATE_PAYLOAD,
        timeout=120.0,   # NegMAS runs synchronously — give it time
    )

    print(f"\nHTTP {resp.status_code}")
    try:
        result = resp.json()
        print(json.dumps(result, indent=2))
    except Exception:
        print(resp.text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-agent callback negotiation test")
    parser.add_argument(
        "--neg-server",
        default=NEG_SERVER,
        help=f"Base URL of the negotiation server (default: {NEG_SERVER})",
    )
    args = parser.parse_args()
    run(args.neg_server)
