"""
API routes for the Semantic Negotiation Agent.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure workspace root is on sys.path so `protocol.sstp` is importable
# regardless of which directory uvicorn is launched from.
_workspace_root = str(Path(__file__).resolve().parents[3])
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

from protocol.sstp import SSTPNegotiateMessage, SSTPCommitMessage  # noqa: E402
from protocol.sstp._base import (
    Origin,
    PolicyLabels,
    Provenance,
    LogicalClock,
)  # noqa: E402
from protocol.sstp.commit import NegotiateCommitSemanticContext  # noqa: E402
from protocol.sstp.negotiate import NegotiateSemanticContext  # noqa: E402

import httpx

from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi.responses import JSONResponse

from ..dependencies import get_pipeline
from ..agent.batch_callback_runner import store_decisions
from ..agent.negotiation_model import NegotiationParticipant
from ..agent.semantic_negotiation import SemanticNegotiationPipeline
from .schemas import (
    AcceptedResponse,
    AgentDecision,
    NegotiationError,
    NegotiationHeader,
    NegotiationOutcomeResponse,
    NegotiationTrace,
    InitiateResponse,
    RoundOffer,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["negotiation"])


# ============== Helpers ==============


def _resolve_participant_id(
    negotiator_name: str, participant_id_by_name: Dict[str, str]
) -> str:
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
    decisions_map: Dict[int, List[Any]] = getattr(result, "round_decisions", {})
    rounds: List[RoundOffer] = []
    for idx, (_, name, offer_tuple) in enumerate(result.history):
        offer: Dict[str, str] = {}
        if offer_tuple is not None:
            for i, issue_id in enumerate(issues):
                offer[issue_id] = str(offer_tuple[i])
        round_num = idx + 1
        raw_decisions = decisions_map.get(round_num, [])
        decisions = [
            AgentDecision(
                participant_id=d["participant_id"],
                action=d["action"],
                offer=d.get("offer"),
            )
            for d in raw_decisions
        ]
        rounds.append(
            RoundOffer(
                round=round_num,
                proposer_id=_resolve_participant_id(name, participant_id_by_name),
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

    return NegotiationTrace(
        rounds=rounds,
        final_agreement=final_agreement,
        timedout=result.timedout,
        broken=result.broken,
    )


def _wrap_sstp_response(
    session_id: str,
    request_id: str,
    domain_resp: Any,
) -> SSTPNegotiateMessage:
    """Wrap a domain response in an SSTPNegotiateMessage envelope.

    ``message_id`` is the caller-supplied ``request_id`` — the server never
    generates its own IDs; unique-ID responsibility belongs to the caller.
    ``payload_hash`` is derived from the serialised payload for integrity.
    """
    if hasattr(domain_resp, "model_dump"):
        payload: Dict[str, Any] = domain_resp.model_dump(exclude_none=True, mode="json")
    else:
        payload = {k: v for k, v in (domain_resp or {}).items() if v is not None}

    payload_str = json.dumps(payload, sort_keys=True)
    payload_hash = hashlib.sha256(payload_str.encode()).hexdigest()
    return SSTPNegotiateMessage(
        kind="negotiate",
        message_id=request_id,
        dt_created=datetime.now(timezone.utc).isoformat(),
        origin=Origin(actor_id="negotiation_server", tenant_id=session_id),
        semantic_context=NegotiateSemanticContext(
            session_id=session_id, sao_state=None
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


# ============== Initiate ==============


@router.post(
    "/negotiate/initiate",
    response_model=SSTPNegotiateMessage,
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
    background_tasks: BackgroundTasks,
    pipeline: SemanticNegotiationPipeline = Depends(get_pipeline),
) -> SSTPNegotiateMessage:
    """Run components 1 → 2 → 3 from a mission text and agent list.

    **Webhook mode** (async): set ``payload.result_callback_url`` to your
    endpoint.  The server returns HTTP 202 immediately and POSTs the full
    ``SSTPNegotiateMessage`` result to that URL when the negotiation
    completes.

    **Synchronous mode**: omit ``result_callback_url``.  The server blocks
    until the negotiation finishes and returns the result directly.
    """
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
    result_callback_url: Optional[str] = payload.get("result_callback_url")

    active_pipeline = (
        SemanticNegotiationPipeline(n_steps=n_steps)
        if n_steps is not None
        else pipeline
    )

    # Components 1 → 2 → 3 all run in a single worker thread so that none of
    # the blocking LLM / NegMAS calls ever stall the event loop.  This keeps
    # the loop free to handle incoming /negotiate/agents-decisions callbacks
    # while the negotiation is in progress.
    def _run_pipeline() -> Any:
        # Component 1: discover issues from mission text
        logger.info(
            "[%s] Component 1 — IntentDiscovery (content_text=%r)",
            session_id,
            content_text[:80],
        )
        _issues: List[str] = active_pipeline._intent_discovery.discover(
            sentence=content_text
        )
        logger.info("[%s] Discovered issues: %s", session_id, _issues)

        # Component 2: generate options per issue
        logger.info("[%s] Component 2 — OptionsGeneration", session_id)
        _options: Dict[str, List[str]] = (
            active_pipeline._options_generation.generate_options(_issues, content_text)
        )
        logger.info("[%s] Generated options: %s", session_id, _options)

        # Build participants
        def _auto_prefs(agent_idx: int) -> Dict[str, Dict[str, float]]:
            prefs: Dict[str, Dict[str, float]] = {}
            for issue, opts in _options.items():
                n = len(opts)
                denom = max(n - 1, 1)
                if agent_idx % 2 == 0:
                    prefs[issue] = {
                        o: round(1.0 - i / denom, 3) for i, o in enumerate(opts)
                    }
                else:
                    prefs[issue] = {o: round(i / denom, 3) for i, o in enumerate(opts)}
            return prefs

        _participants = []
        for idx, a in enumerate(agents_raw):
            cb = a.get("callback_url")
            if cb:
                logger.info(
                    "[%s] participant '%s' (%s) — callback mode → %s",
                    session_id,
                    a["name"],
                    a["id"],
                    cb,
                )
                _participants.append(
                    NegotiationParticipant(
                        id=a["id"],
                        name=a["name"],
                        callback_url=cb,
                    )
                )
            else:
                logger.info(
                    "[%s] participant '%s' (%s) — strategy mode (server-side %s)",
                    session_id,
                    a["name"],
                    a["id"],
                    active_pipeline._negotiation_model._negotiator_cls.__name__,
                )
                _participants.append(
                    NegotiationParticipant(
                        id=a["id"],
                        name=a["name"],
                        preferences=_auto_prefs(idx),
                    )
                )

        # Component 3: run SAO negotiation
        logger.info(
            "[%s] Component 3 — NegotiationModel (%d agents, n_steps=%s)",
            session_id,
            len(_participants),
            active_pipeline.n_steps,
        )
        _result = active_pipeline._negotiation_model.run(
            issues=_issues,
            options_per_issue=_options,
            participants=_participants,
            session_id=session_id,
        )
        return _issues, _options, _participants, _result

    def _build_initiate_response(
        issues, options_per_issue, participants, result
    ) -> SSTPNegotiateMessage:
        """Shared helper: turn a completed NegotiationResult into the final SSTP envelope."""
        participant_id_by_name = {a["name"]: a["id"] for a in agents_raw}
        trace = _build_trace(result, issues, participant_id_by_name)
        total_rounds = len(trace.rounds)

        if total_rounds == 0:
            _status = (
                "timeout"
                if result.timedout
                else ("broken" if result.broken else "agreed")
            )
            return _wrap_sstp_response(
                session_id,
                request_id,
                InitiateResponse(
                    header=header,
                    session_id=session_id,
                    response_id=request_id,
                    status=_status,
                    current_round=RoundOffer(round=0, proposer_id="", offer={}),
                    total_rounds=0,
                    trace=trace,
                ),
            )

        first_round = trace.rounds[0]
        _status = "ongoing"
        if total_rounds == 1:
            if result.agreement is not None:
                _status = "agreed"
            elif result.broken:
                _status = "broken"
            elif result.timedout:
                _status = "timeout"

        return _wrap_sstp_response(
            session_id,
            request_id,
            InitiateResponse(
                header=header,
                session_id=session_id,
                response_id=request_id,
                status=_status,
                current_round=first_round,
                total_rounds=total_rounds,
                trace=trace,
            ),
        )

    # ── Webhook mode: return 202 immediately, deliver result via callback ──
    if result_callback_url:

        async def _run_and_deliver() -> None:
            async with httpx.AsyncClient(timeout=120) as _client:
                try:
                    _issues, _opts, _parts, _result = await asyncio.to_thread(
                        _run_pipeline
                    )
                    _neg_envelope = _build_initiate_response(
                        _issues, _opts, _parts, _result
                    )
                    # Re-wrap as SSTPCommitMessage to signal a finalized negotiation state
                    _agreed = (
                        _result.agreement is not None
                        and not _result.broken
                        and not _result.timedout
                    )
                    _n_rounds = _neg_envelope.payload.get("total_rounds", 0)
                    _trace_payload = _neg_envelope.payload.get("trace") or {}
                    _final_agreement = _trace_payload.get("final_agreement")
                    final_envelope = SSTPCommitMessage(
                        kind="commit",
                        message_id=request_id,
                        dt_created=datetime.now(timezone.utc).isoformat(),
                        origin=Origin(
                            actor_id="negotiation_server", tenant_id=session_id
                        ),
                        semantic_context=NegotiateCommitSemanticContext(
                            session_id=session_id,
                            final_agreement=_final_agreement,
                        ),
                        payload_hash=_neg_envelope.payload_hash,
                        policy_labels=PolicyLabels(
                            sensitivity="internal",
                            propagation="restricted",
                            retention_policy="default",
                        ),
                        provenance=Provenance(sources=[], transforms=[]),
                        payload=_neg_envelope.payload,
                        state_object_id=session_id,
                        parent_ids=[request_id],
                        logical_clock=LogicalClock(type="lamport", value=_n_rounds),
                        merge_strategy="add",
                        confidence_score=1.0 if _agreed else 0.0,
                        risk_score=0.0 if _agreed else 1.0,
                        ttl_seconds=86400,
                    )
                except ValueError as exc:
                    _trace = NegotiationTrace(rounds=[], timedout=False, broken=True)
                    _err_resp = InitiateResponse(
                        header=header,
                        session_id=session_id,
                        response_id=request_id,
                        status="broken",
                        current_round=RoundOffer(round=0, proposer_id="", offer={}),
                        total_rounds=0,
                        trace=_trace,
                        error=NegotiationError(
                            message="BAD_REQUEST", detail={"reason": str(exc)}
                        ),
                    )
                    final_envelope = _wrap_sstp_response(
                        session_id, request_id, _err_resp
                    )
                except Exception:
                    logger.error(
                        "Background pipeline error [%s]", session_id, exc_info=True
                    )
                    _trace = NegotiationTrace(rounds=[], timedout=False, broken=True)
                    _err_resp = InitiateResponse(
                        header=header,
                        session_id=session_id,
                        response_id=request_id,
                        status="broken",
                        current_round=RoundOffer(round=0, proposer_id="", offer={}),
                        total_rounds=0,
                        trace=_trace,
                        error=NegotiationError(
                            message="INTERNAL_ERROR",
                            detail={"traceback": traceback.format_exc()},
                        ),
                    )
                    final_envelope = _wrap_sstp_response(
                        session_id, request_id, _err_resp
                    )

                try:
                    await _client.post(
                        result_callback_url,
                        json=final_envelope.model_dump(mode="json"),
                        headers={"Content-Type": "application/json"},
                    )
                    logger.info(
                        "[%s] result delivered to %s", session_id, result_callback_url
                    )
                except Exception as cb_exc:
                    logger.error(
                        "[%s] failed to deliver result to %s: %s",
                        session_id,
                        result_callback_url,
                        cb_exc,
                    )

        background_tasks.add_task(_run_and_deliver)
        accepted = AcceptedResponse(
            header=header,
            session_id=session_id,
            response_id=request_id,
            result_callback_url=result_callback_url,
        )
        return JSONResponse(
            status_code=202,
            content=_wrap_sstp_response(session_id, request_id, accepted).model_dump(
                mode="json"
            ),
        )

    # ── Synchronous mode: block until negotiation completes ───────────────
    try:
        issues, options_per_issue, participants, result = await asyncio.to_thread(
            _run_pipeline
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
        return JSONResponse(
            status_code=400,
            content=_wrap_sstp_response(session_id, request_id, error_resp).model_dump(
                mode="json"
            ),
        )
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
            error=NegotiationError(
                message="INTERNAL_ERROR", detail={"traceback": traceback.format_exc()}
            ),
        )
        return JSONResponse(
            status_code=500,
            content=_wrap_sstp_response(session_id, request_id, error_resp).model_dump(
                mode="json"
            ),
        )

    return _build_initiate_response(issues, options_per_issue, participants, result)


# ============== Agent-decision callback ==============


@router.post(
    "/negotiate/agents-decisions",
    summary="Receive batch agent decisions from the /decide callback",
    description=(
        "Called by agent servers after they receive a ``POST /decide`` request.\n\n"
        "The agent processes the batch **synchronously**, POSTs the full\n"
        "``List[SSTPNegotiateMessage]`` decision list here, and only then returns\n"
        "the ACK to the ``BatchCallbackRunner``.  This guarantees that when the\n"
        "runner sees the ACK the decisions are already stored in the plain\n"
        "in-process dict — no threading.Event or polling is required."
    ),
)
async def agent_decision(
    body: List[SSTPNegotiateMessage],
) -> JSONResponse:
    """Store agent decisions in the in-process decisions dict."""
    if not body:
        return JSONResponse({"status": "empty"})

    first = body[0]
    session_id: str = (
        first.semantic_context.session_id if first.semantic_context else "unknown"
    )
    round_num: int = (first.payload or {}).get("round", 1)
    key = f"{session_id}:{round_num}"

    raw = [m.model_dump(mode="json") for m in body]
    store_decisions(key, raw)
    return JSONResponse({"status": "ok"})
