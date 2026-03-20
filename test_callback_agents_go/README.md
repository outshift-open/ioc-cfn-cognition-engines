# test_callback_agents_go

Go replica of [`test_callback_agents.py`](../test_callback_agents.py).

Three NegMAS-style Boulware concession agents share a single HTTP server
(`:8091`). The `semantic_negotiation` (`:8089`) drives the SAO mechanism
and calls back all agents on every round via `POST /decide`.

Callback contract (synchronous):
1. Negotiation server sends a batch of per-agent requests to `POST /decide`.
2. Agent handler computes all replies, then POSTs them to
   `POST /api/v1/negotiate/agents-decisions` on the negotiation server.
3. Only after a successful POST does the handler return `{"status":"ack"}` to the
   caller — guaranteeing the server has the decisions before it reads the ACK.

Protocol types (`NegotiateMessage`, `SAOState`, `SAOResponse`, …) are provided
by the local [`sstp/`](sstp/) sub-package, which is a Go replica of the Python
[`protocol/sstp/`](../protocol/sstp/) module.

---

## Prerequisites

| Tool | Version |
|------|---------|
| Go | ≥ 1.21 |
| Python / Poetry | for the negotiation server |

---

## Running

### 1 — Start the semantic_negotiation

```bash
cd semantic_negotiation
poetry install          # first time only
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8089
```

Wait for:
```
INFO:     Application startup complete.
```

### 2 — Run the Go callback agents (separate terminal)

```bash
cd test_callback_agents_go
go run .
```

The program:
1. Starts an agent HTTP server on `:8091`
2. POSTs an `SSTPNegotiateMessage` to `POST /api/v1/negotiate/initiate` on `:8089`
3. On each SAO round, receives a batch callback on `POST /decide`, computes
   replies, POSTs them to `POST /api/v1/negotiate/agents-decisions`, and returns
   `{"status":"ack"}`
4. Writes a full trace under `neg_trace_go/<timestamp>/`

Expected output (truncated):
```
Trace directory: neg_trace_go/20260306_130000

Starting shared agent server on :8091…
Agent server is up.

POST http://localhost:8089/api/v1/negotiate/initiate …

  [Agent A] respond  round=1  utility=0.208  aspiration=0.996  → reject
  [Agent B] respond  round=1  utility=0.792  aspiration=1.000  → reject
  ...
  [Agent B] respond  round=31  utility=0.583  aspiration=0.535  → accept

HTTP 200
...
Trace saved to: neg_trace_go/20260306_130000
```

---

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--neg-server` | `http://localhost:8089` | Base URL of the semantic_negotiation |
| `--trace-dir` | `neg_trace_go` | Root folder for trace output; each run creates a timestamped sub-directory |

Examples:

```bash
# custom negotiation server
go run . --neg-server http://staging.example.com:8089

# custom trace folder
go run . --trace-dir /tmp/runs/experiment_1

# both
go run . --neg-server http://localhost:8089 --trace-dir ./runs/exp_42
```

---

## Trace layout

```
neg_trace_go/
└── 20260306_130000/
    ├── 00_initiate_request.json   ← SSTP NegotiateMessage sent to /initiate
    ├── round_0001/
    │   ├── propose__agent_a__request.json   ← server → agent (propose role)
    │   ├── propose__agent_a__reply.json     ← agent → server
    │   ├── respond__agent_b__request.json   ← server → agent (respond role)
    │   ├── respond__agent_b__reply.json     ← agent → server
    │   └── ...
    ├── round_0002/ … round_00NN/
    └── final_result.json          ← full SAO result with agreement & trace
```

Every request/reply file is a complete `sstp.NegotiateMessage` JSON envelope:

```json
{
  "kind": "negotiate",
  "protocol": "SSTP",
  "semantic_context": {
    "schema_id": "urn:ioc:schema:negotiate:negmas-sao:v1",
    "session_id": "sess-callback-demo-go-001",
    "sao_state": { "step": 5, "running": true, ... },
    "sao_response": { "response": 1, "outcome": {...} }
  },
  "payload": { "action": "counter_offer", "offer": {...}, ... },
  ...
}
```

---

## Agents

| ID | Name | `prefer_low` | `exponent` | `min_reservation` | Strategy |
|----|------|-------------|------------|-------------------|----------|
| `agent-a` | Agent A | true | 1.5 | 0.2 | Near-linear conceder, prefers cheap options |
| `agent-b` | Agent B | false | 3.0 | 0.2 | Hard Boulware, holds premium until end |
| `agent-c` | Agent C | true | 2.0 | 0.1 | Balanced conceder, low reservation floor |

Concession formula (NegMAS `AspirationNegotiator`):

$$\text{aspiration}(t) = \max\bigl(\text{minReservation},\ 1 - t^{\text{exponent}}\bigr), \quad t = \frac{\text{round}}{n\_\text{steps}} \in [0, 1]$$

---

## Relationship to Python version

The Go program is a faithful replica of `test_callback_agents.py`:

| Aspect | Python | Go |
|--------|--------|----|
| Agent logic | `NegMASConcessionAgent` | `Agent` struct |
| Concession formula | `max(minRes, 1 − t^exp)` | same |
| SSTP envelope | `SSTPNegotiateMessage` (Pydantic) | `sstp.NegotiateMessage` (Go struct) |
| Agent server | FastAPI `:8091 POST /decide` | `net/http` `:8091 POST /decide` |
| `/decide` response | `{"status":"ack"}` after POSTing replies | same |
| Decision delivery | `POST /api/v1/negotiate/agents-decisions` | same |
| Initiate endpoint | `POST /api/v1/negotiate/initiate` | same |
| All agents share one server | ✅ | ✅ |
| Trace layout | `neg_trace/<timestamp>/` | `neg_trace_go/<timestamp>/` |
