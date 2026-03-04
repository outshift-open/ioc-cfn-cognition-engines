"""
API routes for the Semantic Negotiation Agent.
"""
from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure workspace root is on sys.path so `protocol.sstp` is importable
# regardless of which directory uvicorn is launched from.
_workspace_root = str(Path(__file__).resolve().parents[3])
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

from protocol.sstp import SSTPNegotiateMessage  # noqa: E402

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from ..dependencies import get_pipeline
from ..agent.negotiation_model import NegotiationParticipant
from ..agent.semantic_negotiation import SemanticNegotiationPipeline
from .schemas import (
    NegotiationError,
    NegotiationHeader,
    NegotiationOutcomeResponse,
    NegotiationTrace,
    InitiateResponse,
    OfferResponseResponse,
    RoundOffer,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["negotiation"])


# ============== Helpers ==============


def _resolve_participant_id(negotiator_name: str, participant_id_by_name: Dict[str, str]) -> str:
    """Map a NegMAS negotiator name back to the caller's participant id.

    NegMAS appends a UUID suffix (e.g. "Agent A-8e5fc06b-..."). Try exact match
    first, then prefix match.
    """
    if negotiator_name in participant_id_by_name:
        return participant_id_by_name[negotiator_name]
    for name, pid in participant_id_by_name.items():
        if negotiator_name.startswith(name):
            return pid
    return negotiator_name


def _build_trace(
    result: Any,
    issues: List[str],
    participant_id_by_name: Dict[str, str],
) -> NegotiationTrace:
    """Serialise a NegotiationResult into a wire-safe NegotiationTrace."""
    rounds: List[RoundOffer] = []
    for idx, (_, name, offer_tuple) in enumerate(result.history):
        offer: Dict[str, str] = {}
        if offer_tuple is not None:
            for i, issue_id in enumerate(issues):
                offer[issue_id] = str(offer_tuple[i])
        rounds.append(
            RoundOffer(
                round=idx + 1,
                proposer_id=_resolve_participant_id(name, participant_id_by_name),
                offer=offer,
            )
        )

    final_agreement = None
    if result.agreement is not None:
        final_agreement = [
            NegotiationOutcomeResponse(issue_id=o.issue_id, chosen_option=o.chosen_option)
            for o in result.agreement
        ]

    return NegotiationTrace(
        rounds=rounds,
        final_agreement=final_agreement,
        timedout=result.timedout,
        broken=result.broken,
    )


# ============== Initiate ==============


@router.post(
    "/negotiate/initiate",
    response_model=InitiateResponse,
    response_model_exclude_none=True,
    summary="Initiate a semantic negotiation from a mission description",
    description=(
        "Accepts a mission description and a list of agents, then runs:\n\n"
        "1. **Component 1** — `IntentDiscovery` extracts negotiable issues from content_text.\n"
        "2. **Component 2** — `OptionsGeneration` produces candidate options per issue.\n"
        "3. **Component 3** — NegMAS SAO runs the negotiation and returns the trace.\n\n"
        "Request body is a full SSTP **`SSTPNegotiateMessage`** envelope.\n"
        "- `semantic_context.session_id` — caller-supplied session ID (required).\n"
        "- `payload.content_text` — natural-language mission or negotiation goal.\n"
        "- `payload.agents` — list of `{id, name}` for each participating agent (min 2).\n"
        "- `payload.n_steps` — optional SAO round budget."
    ),
)
async def negotiate_initiate(
    body: SSTPNegotiateMessage,
    pipeline: SemanticNegotiationPipeline = Depends(get_pipeline),
) -> InitiateResponse:
    """Run components 1 → 2 → 3 from a mission text and agent list."""
    session_id = body.semantic_context.session_id
    request_id = body.message_id
    header = NegotiationHeader(
        workspace_id=body.origin.tenant_id,
        mas_id=body.origin.actor_id,
    )

    payload = body.payload
    content_text: str = payload.get("content_text", "")
    agents_raw: List[Dict[str, Any]] = payload["agents"]
    n_steps: Optional[int] = payload.get("n_steps")

    active_pipeline = SemanticNegotiationPipeline(n_steps=n_steps) if n_steps is not None else pipeline

    # Component 1: discover issues from mission text
    logger.info("[%s] Component 1 — IntentDiscovery (content_text=%r)", session_id, content_text[:80])
    issues: List[str] = active_pipeline._intent_discovery.discover(content_text=content_text)
    logger.info("[%s] Discovered issues: %s", session_id, issues)

    # Component 2: generate options per issue
    logger.info("[%s] Component 2 — OptionsGeneration", session_id)
    options_per_issue: Dict[str, List[str]] = active_pipeline._options_generation.generate(issues)
    logger.info("[%s] Generated options: %s", session_id, options_per_issue)

    # Build participants.
    # Agents with a callback_url make their own decisions — no server-side
    # preferences needed.  Agents without a callback_url fall back to the
    # server-side negotiator strategy with auto-generated preferences.
    def _auto_prefs(agent_idx: int) -> Dict[str, Dict[str, float]]:
        prefs: Dict[str, Dict[str, float]] = {}
        for issue, opts in options_per_issue.items():
            n = len(opts)
            denom = max(n - 1, 1)
            if agent_idx % 2 == 0:
                prefs[issue] = {o: round(1.0 - i / denom, 3) for i, o in enumerate(opts)}
            else:
                prefs[issue] = {o: round(i / denom, 3) for i, o in enumerate(opts)}
        return prefs

    participants = []
    for idx, a in enumerate(agents_raw):
        cb = a.get("callback_url")
        if cb:
            logger.info(
                "[%s] participant '%s' (%s) — callback mode → %s",
                session_id, a["name"], a["id"], cb,
            )
            participants.append(NegotiationParticipant(
                id=a["id"],
                name=a["name"],
                callback_url=cb,
            ))
        else:
            logger.info(
                "[%s] participant '%s' (%s) — strategy mode (server-side %s)",
                session_id, a["name"], a["id"],
                active_pipeline._negotiation_model._negotiator_cls.__name__,
            )
            participants.append(NegotiationParticipant(
                id=a["id"],
                name=a["name"],
                preferences=_auto_prefs(idx),
            ))
    participant_id_by_name = {a["name"]: a["id"] for a in agents_raw}

    # Component 3: run SAO negotiation
    logger.info("[%s] Component 3 — NegotiationModel (%d agents, n_steps=%s)",
                session_id, len(participants), active_pipeline.n_steps)
    try:
        result = active_pipeline._negotiation_model.run(
            issues=issues,
            options_per_issue=options_per_issue,
            participants=participants,
            session_id=session_id,
        )
    except ValueError as exc:
        trace = NegotiationTrace(rounds=[], timedout=False, broken=True)
        error_resp = InitiateResponse(
            header=header,
            session_id=session_id,
            response_id=request_id,
            status="broken",
            current_round=RoundOffer(round=0, proposer_id="", offer={}),
            total_rounds=0,
            trace=trace,
            error=NegotiationError(message="BAD_REQUEST", detail={"reason": str(exc)}),
        )
        return JSONResponse(status_code=400, content=error_resp.model_dump(exclude_none=True))
    except Exception:
        logger.error("Unexpected error in /negotiate/initiate [%s]", request_id)
        trace = NegotiationTrace(rounds=[], timedout=False, broken=True)
        error_resp = InitiateResponse(
            header=header,
            session_id=session_id,
            response_id=request_id,
            status="broken",
            current_round=RoundOffer(round=0, proposer_id="", offer={}),
            total_rounds=0,
            trace=trace,
            error=NegotiationError(message="INTERNAL_ERROR", detail={"traceback": traceback.format_exc()}),
        )
        return JSONResponse(status_code=500, content=error_resp.model_dump(exclude_none=True))

    trace = _build_trace(result, issues, participant_id_by_name)
    total_rounds = len(trace.rounds)

    if total_rounds == 0:
        status = "timeout" if result.timedout else ("broken" if result.broken else "agreed")
        return InitiateResponse(
            header=header,
            session_id=session_id,
            response_id=request_id,
            status=status,
            current_round=RoundOffer(round=0, proposer_id="", offer={}),
            total_rounds=0,
            trace=trace,
        )

    first_round = trace.rounds[0]
    status = "ongoing"
    if total_rounds == 1:
        if result.agreement is not None:
            status = "agreed"
        elif result.broken:
            status = "broken"
        elif result.timedout:
            status = "timeout"

    return InitiateResponse(
        header=header,
        session_id=session_id,
        response_id=request_id,
        status=status,
        current_round=first_round,
        total_rounds=total_rounds,
        trace=trace,
    )


# ============== Offer response ==============


@router.post(
    "/negotiate/offer-response",
    response_model=OfferResponseResponse,
    response_model_exclude_none=True,
    summary="Respond to an offer in a negotiation session (stateless)",
    description=(
        "Steps through the pre-computed SAO trace supplied in `payload.trace`.\n\n"
        "**The server holds no state** — pass back the `trace` from `/negotiate/initiate` verbatim.\n\n"
        "Request body is a full SSTP **`SSTPNegotiateMessage`** envelope.\n"
        "- `semantic_context.session_id` — correlation identifier from initiate.\n"
        "- `payload.round` — 1-based round number being responded to.\n"
        "- `payload.action` — `accept` | `continue` | `reject`.\n"
        "- `payload.trace` — the `NegotiationTrace` received from `/negotiate/initiate`."
    ),
)
async def negotiate_offer_response(
    body: SSTPNegotiateMessage,
) -> OfferResponseResponse:
    """Step through the caller-supplied trace; no server-side state required."""
    session_id = body.semantic_context.session_id
    request_id = body.message_id
    header = NegotiationHeader(
        workspace_id=body.origin.tenant_id,
        mas_id=body.origin.actor_id,
    )

    payload = body.payload
    round_num: int = payload["round"]
    action: str = payload["action"]
    trace = NegotiationTrace.model_validate(payload["trace"])
    total_rounds = len(trace.rounds)

    if round_num < 1 or round_num > total_rounds:
        error_resp = OfferResponseResponse(
            header=header,
            session_id=session_id,
            response_id=request_id,
            status="broken",
            total_rounds=total_rounds,
            steps_remaining=0,
            error=NegotiationError(
                message="INVALID_ROUND",
                detail={"round": round_num, "total_rounds": total_rounds},
            ),
        )
        return JSONResponse(status_code=400, content=error_resp.model_dump(exclude_none=True))

    # ── reject ───────────────────────────────────────────────────────────────
    if action == "reject":
        return OfferResponseResponse(
            header=header,
            session_id=session_id,
            response_id=request_id,
            status="broken",
            total_rounds=total_rounds,
            steps_remaining=0,
        )

    # ── accept ───────────────────────────────────────────────────────────────
    if action == "accept":
        current = trace.rounds[round_num - 1]
        agreement = [
            NegotiationOutcomeResponse(issue_id=k, chosen_option=v)
            for k, v in current.offer.items()
        ]
        return OfferResponseResponse(
            header=header,
            session_id=session_id,
            response_id=request_id,
            status="agreed",
            agreement=agreement,
            total_rounds=total_rounds,
            steps_remaining=0,
        )

    # ── continue ─────────────────────────────────────────────────────────────
    next_round_idx = round_num  # 0-based index for the NEXT round
    if next_round_idx >= total_rounds:
        # Exhausted all pre-computed rounds — report NegMAS's own final outcome
        if trace.final_agreement is not None:
            status = "agreed"
            agreement: Optional[List[NegotiationOutcomeResponse]] = trace.final_agreement
        elif trace.broken:
            status = "broken"
            agreement = None
        else:
            status = "timeout"
            agreement = None
        return OfferResponseResponse(
            header=header,
            session_id=session_id,
            response_id=request_id,
            status=status,
            agreement=agreement,
            total_rounds=total_rounds,
            steps_remaining=0,
        )

    next_round = trace.rounds[next_round_idx]
    steps_remaining = total_rounds - (next_round_idx + 1)

    # If this is the final trace round and NegMAS reached agreement, surface it
    is_last = next_round_idx + 1 == total_rounds
    if is_last and trace.final_agreement is not None:
        return OfferResponseResponse(
            header=header,
            session_id=session_id,
            response_id=request_id,
            status="agreed",
            next_round=next_round,
            agreement=trace.final_agreement,
            total_rounds=total_rounds,
            steps_remaining=0,
        )

    return OfferResponseResponse(
        header=header,
        session_id=session_id,
        response_id=request_id,
        status="ongoing",
        next_round=next_round,
        total_rounds=total_rounds,
        steps_remaining=steps_remaining,
    )
