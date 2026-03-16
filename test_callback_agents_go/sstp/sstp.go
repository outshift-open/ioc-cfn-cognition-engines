// Package sstp is a Go replica of the Python protocol/sstp/ module.
// It mirrors:
//
//	_base.py      -> Origin, SemanticContext, PolicyLabels, Provenance,
//	                PayloadRef, LogicalClock, Message
//	negmas_sao.py -> ResponseType, ThreadState, MechanismState, GBState,
//	                SAOState, SAOResponse, SAONMI
//	negotiate.py  -> NegotiateSemanticContext, NegotiateMessage,
//	                + helpers: MakeUUID5, PayloadHash, Slug, OutcomeFromMap, BuildReply
package sstp

import (
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"time"
)

// ── _base.py — literals ───────────────────────────────────────────────────

const ProtocolSSTP = "SSTP"

const (
	SensitivityPublic       = "public"
	SensitivityInternal     = "internal"
	SensitivityRestricted   = "restricted"
	SensitivityConfidential = "confidential"
)

const (
	PropagationForward    = "forward"
	PropagationRestricted = "restricted"
	PropagationNoForward  = "no_forward"
)

const (
	EncodingJSON           = "json"
	EncodingStructuredText = "structured_text"
	EncodingHybrid         = "hybrid"
)

const (
	MergeStrategyAdd     = "add"
	MergeStrategyReplace = "replace"
	MergeStrategyMerge   = "merge"
	MergeStrategyCRDT    = "crdt"
)

const (
	PayloadRefInline   = "inline"
	PayloadRefExternal = "external"
)

// ── _base.py — sub-models ─────────────────────────────────────────────────

// Origin identifies the message producer.
// Mirrors: class Origin(BaseModel) in _base.py
type Origin struct {
	ActorID     string  `json:"actor_id"`
	TenantID    string  `json:"tenant_id"`
	Attestation *string `json:"attestation,omitempty"`
}

// SemanticContext holds schema and encoding metadata.
// Mirrors: class SemanticContext(BaseModel) in _base.py.
// For kind="negotiate" use NegotiateSemanticContext instead.
type SemanticContext struct {
	SchemaID      string `json:"schema_id"`
	SchemaVersion string `json:"schema_version"`
	Encoding      string `json:"encoding"`
}

// PolicyLabels holds data-handling policy annotations.
// Mirrors: class PolicyLabels(BaseModel) in _base.py
type PolicyLabels struct {
	Sensitivity     string `json:"sensitivity"`
	Propagation     string `json:"propagation"`
	RetentionPolicy string `json:"retention_policy"`
}

// Provenance holds lineage of this message.
// Mirrors: class Provenance(BaseModel) in _base.py
type Provenance struct {
	Sources    []string `json:"sources"`
	Transforms []string `json:"transforms"`
}

// PayloadRef is a reference to where payload content is stored.
// Mirrors: class PayloadRef(BaseModel) in _base.py
type PayloadRef struct {
	Type string `json:"type"`
	Ref  string `json:"ref"`
}

// LogicalClock is a Lamport scalar or vector clock snapshot.
// Mirrors: class LogicalClock(BaseModel) in _base.py
type LogicalClock struct {
	Type  string          `json:"type"`
	Value json.RawMessage `json:"value"`
}

// ── _base.py — envelope base ──────────────────────────────────────────────

// Message is the generic SSTP envelope base with SemanticContext as raw JSON.
// Mirrors: class _STBaseMessage(BaseModel) in _base.py.
// For kind="negotiate" use NegotiateMessage which provides a typed SemanticContext.
type Message struct {
	Kind            string          `json:"kind"`
	Protocol        string          `json:"protocol"`
	Version         string          `json:"version"`
	MessageID       string          `json:"message_id"`
	DtCreated       string          `json:"dt_created"`
	Origin          Origin          `json:"origin"`
	SemanticContext json.RawMessage `json:"semantic_context"`
	PayloadHash     string          `json:"payload_hash"`
	PolicyLabels    PolicyLabels    `json:"policy_labels"`
	Provenance      Provenance      `json:"provenance"`
	Payload         map[string]any  `json:"payload"`
	StateObjectID   *string         `json:"state_object_id,omitempty"`
	ParentIDs       []string        `json:"parent_ids"`
	LogicalClock    *LogicalClock   `json:"logical_clock,omitempty"`
	PayloadRefs     []PayloadRef    `json:"payload_refs"`
	ConfidenceScore *float64        `json:"confidence_score,omitempty"`
	TTLSeconds      *int            `json:"ttl_seconds,omitempty"`
	MergeStrategy   *string         `json:"merge_strategy,omitempty"`
	RiskScore       *float64        `json:"risk_score,omitempty"`
}

// ── negmas_sao.py ─────────────────────────────────────────────────────────

// ResponseType mirrors Python ResponseType(IntEnum) from negmas_sao.py.
type ResponseType int

const (
	ResponseAcceptOffer    ResponseType = 0
	ResponseRejectOffer    ResponseType = 1
	ResponseEndNegotiation ResponseType = 2
	ResponseNoResponse     ResponseType = 3
	ResponseWait           ResponseType = 4
	ResponseLeave          ResponseType = 5
)

// responseTypeNames maps ResponseType integer values to their Python enum name strings.
// Mirrors the _serialize_response field_serializer in SAOResponse (negmas_sao.py).
var responseTypeNames = map[ResponseType]string{
	ResponseAcceptOffer:    "ACCEPT_OFFER",
	ResponseRejectOffer:    "REJECT_OFFER",
	ResponseEndNegotiation: "END_NEGOTIATION",
	ResponseNoResponse:     "NO_RESPONSE",
	ResponseWait:           "WAIT",
	ResponseLeave:          "LEAVE",
}

// responseTypeValues maps Python enum name strings to ResponseType values.
// Mirrors the _coerce_response field_validator in SAOResponse (negmas_sao.py).
var responseTypeValues = map[string]ResponseType{
	"ACCEPT_OFFER":    ResponseAcceptOffer,
	"REJECT_OFFER":    ResponseRejectOffer,
	"END_NEGOTIATION": ResponseEndNegotiation,
	"NO_RESPONSE":     ResponseNoResponse,
	"WAIT":            ResponseWait,
	"LEAVE":           ResponseLeave,
}

// MarshalJSON serializes ResponseType as its human-readable name string
// (e.g. "ACCEPT_OFFER"), matching SAOResponse._serialize_response in the Python protocol.
func (r ResponseType) MarshalJSON() ([]byte, error) {
	if name, ok := responseTypeNames[r]; ok {
		return json.Marshal(name)
	}
	return json.Marshal(int(r))
}

// UnmarshalJSON accepts both integer values (0) and name strings ("ACCEPT_OFFER"),
// matching SAOResponse._coerce_response in the Python protocol.
func (r *ResponseType) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err == nil {
		if v, ok := responseTypeValues[s]; ok {
			*r = v
			return nil
		}
		return fmt.Errorf("unknown ResponseType name: %q", s)
	}
	var i int
	if err := json.Unmarshal(b, &i); err != nil {
		return fmt.Errorf("cannot unmarshal ResponseType from: %s", b)
	}
	*r = ResponseType(i)
	return nil
}

// Outcome holds a NegMAS negotiation outcome as raw JSON.
// Python: Outcome = dict[str, Any] | tuple | None
type Outcome = json.RawMessage

// ThreadState holds per-thread state in a GB negotiation round.
// Mirrors: class ThreadState(BaseModel) in negmas_sao.py
type ThreadState struct {
	NewOffer       Outcome                 `json:"new_offer"`
	NewData        map[string]any          `json:"new_data,omitempty"`
	NewResponses   map[string]ResponseType `json:"new_responses"`
	AcceptedOffers []Outcome               `json:"accepted_offers"`
}

// MechanismState is the base state for all NegMAS negotiation mechanisms.
// Mirrors: class MechanismState(BaseModel) in negmas_sao.py
type MechanismState struct {
	Running         bool            `json:"running"`
	Waiting         bool            `json:"waiting"`
	Started         bool            `json:"started"`
	Step            int             `json:"step"`
	Time            float64         `json:"time"`
	RelativeTime    float64         `json:"relative_time"`
	Broken          bool            `json:"broken"`
	Timedout        bool            `json:"timedout"`
	Agreement       Outcome         `json:"agreement"`
	Results         json.RawMessage `json:"results,omitempty"`
	NNegotiators    int             `json:"n_negotiators"`
	HasError        bool            `json:"has_error"`
	ErrorDetails    string          `json:"error_details"`
	ErredNegotiator string          `json:"erred_negotiator"`
	ErredAgent      string          `json:"erred_agent"`
}

// GBState is the Generalized Bargaining mechanism state.
// Mirrors: class GBState(MechanismState) in negmas_sao.py
type GBState struct {
	MechanismState
	Threads         map[string]ThreadState `json:"threads"`
	LastThread      string                 `json:"last_thread"`
	LeftNegotiators []string               `json:"left_negotiators"`
	NParticipating  int                    `json:"n_participating"`
}

// SAOState is the full mechanism state for the Stacked Alternating Offers protocol.
// Mirrors: class SAOState(GBState) in negmas_sao.py
type SAOState struct {
	GBState
	CurrentOffer         Outcome         `json:"current_offer"`
	CurrentProposer      *string         `json:"current_proposer,omitempty"`
	CurrentProposerAgent *string         `json:"current_proposer_agent,omitempty"`
	NAcceptances         int             `json:"n_acceptances"`
	NewOffers            json.RawMessage `json:"new_offers,omitempty"`
	NewOffererAgents     []any           `json:"new_offerer_agents,omitempty"`
	LastNegotiator       *string         `json:"last_negotiator,omitempty"`
	CurrentData          map[string]any  `json:"current_data,omitempty"`
	NewData              json.RawMessage `json:"new_data,omitempty"`
}

// SAOResponse is a single negotiator response in one SAO round.
// Mirrors: class SAOResponse(BaseModel) in negmas_sao.py
type SAOResponse struct {
	Response ResponseType   `json:"response"`
	Outcome  Outcome        `json:"outcome,omitempty"`
	Data     map[string]any `json:"data,omitempty"`
}

// SAONMI is the NegotiatorMechanismInterface configuration snapshot for SAO.
// Mirrors: class SAONMI(BaseModel, frozen=True) in negmas_sao.py
type SAONMI struct {
	ID                      string         `json:"id"`
	NOutcomes               float64        `json:"n_outcomes"`
	SharedTimeLimit         float64        `json:"shared_time_limit"`
	SharedNSteps            *int           `json:"shared_n_steps,omitempty"`
	PrivateTimeLimit        float64        `json:"private_time_limit"`
	PrivateNSteps           *int           `json:"private_n_steps,omitempty"`
	Pend                    float64        `json:"pend"`
	PendPerSecond           float64        `json:"pend_per_second"`
	StepTimeLimit           float64        `json:"step_time_limit"`
	NegotiatorTimeLimit     float64        `json:"negotiator_time_limit"`
	DynamicEntry            bool           `json:"dynamic_entry"`
	MaxNNegotiators         *int           `json:"max_n_negotiators,omitempty"`
	Annotation              map[string]any `json:"annotation"`
	EndOnNoResponse         bool           `json:"end_on_no_response"`
	OneOfferPerStep         bool           `json:"one_offer_per_step"`
	OfferingIsAccepting     bool           `json:"offering_is_accepting"`
	AllowNoneWithData       bool           `json:"allow_none_with_data"`
	AllowNegotiatorsToLeave bool           `json:"allow_negotiators_to_leave"`
}

// ── negotiate.py ──────────────────────────────────────────────────────────

// NegotiateSemanticContext is the SAO-specific semantic context for kind="negotiate".
// Mirrors: class NegotiateSemanticContext(BaseModel) in negotiate.py.
//
// SAOState is stored as json.RawMessage for lossless echo-back: the negotiation
// server verifies its SHA-256 checksum, so the bytes must be preserved verbatim.
type NegotiateSemanticContext struct {
	SchemaID        string            `json:"schema_id"`
	SchemaVersion   string            `json:"schema_version"`
	Encoding        string            `json:"encoding"`
	SessionID       string            `json:"session_id"`
	Issues          []string          `json:"issues"`
	OptionsPerIssue map[string][]string `json:"options_per_issue"`
	SAOState        json.RawMessage   `json:"sao_state"`
	SAOResponse     *SAOResponse      `json:"sao_response,omitempty"`
	NMI             *SAONMI           `json:"nmi,omitempty"`
}

// DefaultNegotiateSemanticContext returns a NegotiateSemanticContext with
// the canonical Python field defaults:
//
//	schema_id      = "urn:ioc:schema:negotiate:negmas-sao:v1"
//	schema_version = "1.0"
//	encoding       = "json"
//	sao_state      = null
func DefaultNegotiateSemanticContext(sessionID string) NegotiateSemanticContext {
	return NegotiateSemanticContext{
		SchemaID:      "urn:ioc:schema:negotiate:negmas-sao:v1",
		SchemaVersion: "1.0",
		Encoding:      EncodingJSON,
		SessionID:     sessionID,
		SAOState:      json.RawMessage("null"),
	}
}

// NegotiateMessage is a fully-typed SSTP envelope for kind="negotiate".
// Mirrors: class SSTPNegotiateMessage(_STBaseMessage) in negotiate.py,
// with SemanticContext narrowed to NegotiateSemanticContext.
type NegotiateMessage struct {
	Kind            string                   `json:"kind"`
	Protocol        string                   `json:"protocol"`
	Version         string                   `json:"version"`
	MessageID       string                   `json:"message_id"`
	DtCreated       string                   `json:"dt_created"`
	Origin          Origin                   `json:"origin"`
	SemanticContext NegotiateSemanticContext `json:"semantic_context"`
	PayloadHash     string                   `json:"payload_hash"`
	PolicyLabels    PolicyLabels             `json:"policy_labels"`
	Provenance      Provenance               `json:"provenance"`
	Payload         map[string]any           `json:"payload"`
	StateObjectID   *string                  `json:"state_object_id,omitempty"`
	ParentIDs       []string                 `json:"parent_ids"`
	PayloadRefs     []any                    `json:"payload_refs"`
}

// DefaultNegotiateMessage returns a NegotiateMessage with protocol/version/kind
// filled in, matching Python's _STBaseMessage defaults.
func DefaultNegotiateMessage(messageID, dtCreated string, origin Origin, sc NegotiateSemanticContext) NegotiateMessage {
	return NegotiateMessage{
		Kind:            "negotiate",
		Protocol:        ProtocolSSTP,
		Version:         "0",
		MessageID:       messageID,
		DtCreated:       dtCreated,
		Origin:          origin,
		SemanticContext: sc,
		PolicyLabels: PolicyLabels{
			Sensitivity:     SensitivityInternal,
			Propagation:     PropagationRestricted,
			RetentionPolicy: "default",
		},
		Provenance:  Provenance{Sources: []string{}, Transforms: []string{}},
		ParentIDs:   []string{},
		PayloadRefs: []any{},
	}
}

// ── Protocol helpers ──────────────────────────────────────────────────────

// UUIDNamespaceURL is the well-known UUID5 namespace for URLs (RFC 4122).
var UUIDNamespaceURL = [16]byte{
	0x6b, 0xa7, 0xb8, 0x11,
	0x9d, 0xad,
	0x11, 0xd1,
	0x80, 0xb4,
	0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8,
}

// MakeUUID5 returns a UUID5 string using SHA-1, matching Python's uuid.uuid5().
func MakeUUID5(namespace [16]byte, name string) string {
	h := sha1.New()
	h.Write(namespace[:])
	h.Write([]byte(name))
	sum := h.Sum(nil)
	sum[6] = (sum[6] & 0x0f) | 0x50
	sum[8] = (sum[8] & 0x3f) | 0x80
	s := hex.EncodeToString(sum[:16])
	return s[0:8] + "-" + s[8:12] + "-" + s[12:16] + "-" + s[16:20] + "-" + s[20:32]
}

// PayloadHash returns the SHA-256 hex digest of the canonical JSON of payload.
// Matching: hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
func PayloadHash(payload map[string]any) string {
	b, err := json.Marshal(payload)
	if err != nil {
		return strings.Repeat("0", 64)
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// Slug converts a display name to a lowercase filesystem-safe identifier.
// Matching Python's _slug() helper.
var reSlug = regexp.MustCompile("[^a-z0-9]+")

// Slug converts a display name to a slug.
func Slug(name string) string {
	return strings.Trim(reSlug.ReplaceAllString(strings.ToLower(name), "_"), "_")
}

// OutcomeFromMap encodes a map[string]any as an Outcome (json.RawMessage).
// Returns json.RawMessage("null") on a nil or empty map.
func OutcomeFromMap(m map[string]any) Outcome {
	if len(m) == 0 {
		return json.RawMessage("null")
	}
	b, err := json.Marshal(m)
	if err != nil {
		return json.RawMessage("null")
	}
	return b
}

// BuildReply constructs a NegotiateMessage reply envelope with a deterministic
// message_id via UUID5(session_id:slug(agentName):payload_hash).
// saoStateRaw is echoed verbatim from the incoming request so the server's
// SAOState checksum integrity check passes.
func BuildReply(sessionID, agentName string, replyPayload map[string]any, saoResponse *SAOResponse, saoStateRaw json.RawMessage) NegotiateMessage {
	if len(saoStateRaw) == 0 {
		saoStateRaw = json.RawMessage("null")
	}
	pHash := PayloadHash(replyPayload)
	msgID := MakeUUID5(UUIDNamespaceURL,
		fmt.Sprintf("%s:%s:%s", sessionID, Slug(agentName), pHash))
	sc := NegotiateSemanticContext{
		SchemaID:      "urn:ioc:schema:negotiate:negmas-sao:v1",
		SchemaVersion: "1.0",
		Encoding:      EncodingJSON,
		SessionID:     sessionID,
		SAOState:      saoStateRaw,
		SAOResponse:   saoResponse,
	}
	msg := DefaultNegotiateMessage(
		msgID,
		time.Now().UTC().Format(time.RFC3339Nano),
		Origin{ActorID: Slug(agentName), TenantID: sessionID},
		sc,
	)
	msg.Payload = replyPayload
	msg.PayloadHash = pHash
	return msg
}
