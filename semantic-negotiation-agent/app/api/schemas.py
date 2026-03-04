"""
Pydantic request/response models for the Semantic Negotiation Agent API.
"""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ============== Shared header ==============


class NegotiationHeader(BaseModel):
    """Header common to all negotiation requests."""

    workspace_id: str = Field(..., description="Workspace identifier")
    mas_id: str = Field(..., description="MAS identifier")
    agent_id: Optional[str] = Field(None, description="Calling agent identifier (optional)")


# ============== Negotiate request ==============


class ParticipantRequest(BaseModel):
    """One negotiating party supplied by the caller."""

    id: str = Field(..., description="Unique identifier for this participant")
    name: str = Field(..., description="Human-readable display name")
    preferences: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description=(
            "Per-issue option utilities. "
            "Shape: {issue_id: {option_label: utility_value}}. "
            "Values should be in [0.0, 1.0]. "
            "Ignored when callback_url is set — the external agent owns its own utility function."
        ),
    )
    issue_weights: Optional[Dict[str, float]] = Field(
        None,
        description=(
            "Optional per-issue importance weights. "
            "Defaults to uniform weights when omitted. "
            "Ignored when callback_url is set."
        ),
    )
    callback_url: Optional[str] = Field(
        None,
        description=(
            "HTTP endpoint of the external agent that will decide propose/respond actions "
            "for this participant. When set, the server-side strategy is bypassed: the server "
            "POSTs an SSTPNegotiateMessage to this URL on every NegMAS round and waits for the "
            "agent's JSON reply ({ \"offer\": {...} } or { \"action\": \"accept\" | \"reject\" | \"end\" }). "
            "preferences and issue_weights are ignored."
        ),
    )


class NegotiateRequest(BaseModel):
    """Top-level request body for POST /api/v1/negotiate."""

    header: NegotiationHeader
    request_id: str = Field(
        ...,
        description="Client-supplied request ID; echoed back as response_id",
    )
    issues: List[str] = Field(
        ...,
        description="Ordered list of issue identifiers to negotiate over",
    )
    options_per_issue: Dict[str, List[str]] = Field(
        ...,
        description="Candidate options for each issue. Shape: {issue_id: [option, ...]}",
    )
    participants: List[ParticipantRequest] = Field(
        ...,
        min_length=2,
        description="Negotiating parties (minimum 2)",
    )
    n_steps: Optional[int] = Field(
        None,
        description="Maximum SAO rounds. Falls back to the service default when omitted.",
    )


# ============== Negotiate response ==============


class NegotiationOutcomeResponse(BaseModel):
    """Agreed value for a single issue."""

    issue_id: str
    chosen_option: str


class NegotiationError(BaseModel):
    """Error block returned when negotiation fails."""

    message: str = Field(..., description="User-meaningful error message")
    detail: Optional[Dict[str, Any]] = Field(None, description="Debugging context")


class NegotiateResponse(BaseModel):
    """
    Response for POST /api/v1/negotiate.

    Either ``error`` is set **or** the outcome fields are set — never both.
    """

    header: NegotiationHeader
    response_id: str = Field(..., description="Echoed from request_id")
    agreement: Optional[List[NegotiationOutcomeResponse]] = Field(
        None,
        description="Agreed option per issue, or null when no agreement was reached",
    )
    timedout: Optional[bool] = None
    broken: Optional[bool] = None
    steps: Optional[int] = None
    history: Optional[List[Any]] = Field(
        None,
        description="SAO trace as (step, negotiator_id, offer) tuples",
    )
    error: Optional[NegotiationError] = None


# ============== Initiate ==============
# Note: the request body for /negotiate/initiate and /negotiate/offer-response
# is SSTPNegotiateMessage (protocol.sstp.SSTPNegotiateMessage).
# session_id is carried in semantic_context.session_id and is REQUIRED —
# the caller must always supply it; the server never generates one.


class RoundOffer(BaseModel):
    """The offer produced in a single SAO round."""

    round: int = Field(..., description="1-based round number within the SAO trace")
    proposer_id: str = Field(..., description="ID of the participant who made this proposal")
    offer: Dict[str, str] = Field(..., description="Proposed option per issue. Shape: {issue_id: option}")


class NegotiationTrace(BaseModel):
    """Immutable pre-computed SAO trace returned by /negotiate/initiate.

    Pass this back verbatim in every /negotiate/offer-response request.
    The server holds **no state** — all session data lives in this object.
    """

    rounds: List[RoundOffer] = Field(..., description="All SAO rounds in order (1-based)")
    final_agreement: Optional[List[NegotiationOutcomeResponse]] = Field(
        None,
        description="Agreement reached by NegMAS at the end of its own run, if any",
    )
    timedout: bool = Field(False, description="Whether NegMAS exhausted its step budget")
    broken: bool = Field(False, description="Whether a participant explicitly broke off")


class InitiateResponse(BaseModel):
    """Response for POST /api/v1/negotiate/initiate."""

    header: NegotiationHeader
    session_id: str = Field(
        ...,
        description="Opaque correlation identifier — echo back in every /negotiate/offer-response call",
    )
    response_id: str = Field(..., description="Echoed from request_id")
    status: Literal["ongoing", "agreed", "broken", "timeout"] = Field(
        ...,
        description=(
            "'ongoing' when more rounds are available; "
            "'agreed'/'broken'/'timeout' when the negotiation already resolved in round 1"
        ),
    )
    current_round: RoundOffer = Field(..., description="The first offer in the trace")
    total_rounds: int = Field(..., description="Total SAO rounds in the pre-computed trace")
    trace: NegotiationTrace = Field(
        ...,
        description=(
            "Complete pre-computed SAO trace. "
            "Store this client-side and pass it back with every /negotiate/offer-response request."
        ),
    )
    error: Optional[NegotiationError] = None


# ============== Offer response ==============
# Note: the request body is SSTPNegotiateMessage — see comment above.


class OfferResponseResponse(BaseModel):
    """Response for POST /api/v1/negotiate/offer-response."""

    header: NegotiationHeader
    session_id: str
    response_id: str
    status: Literal["ongoing", "agreed", "broken", "timeout"]
    next_round: Optional[RoundOffer] = Field(
        None, description="Next SAO round offer — present only when status='ongoing'"
    )
    agreement: Optional[List[NegotiationOutcomeResponse]] = Field(
        None, description="Final agreed outcome — present when status='agreed'"
    )
    total_rounds: int
    steps_remaining: int = Field(..., description="Rounds left before the session times out")
    error: Optional[NegotiationError] = None


# ============== Health ==============


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    version: str
