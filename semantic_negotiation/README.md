# Semantic Negotiation Agent

A skeleton agent responsible for semantic negotiation between cognitive entities in this workspace.

## Features

- `SemanticNegotiationAgent` class with a stub API for proposal-based negotiation
- Methods: `negotiate`, `propose`, `evaluate`, `accept`, `reject`
- All methods raise `NotImplementedError` until implemented

## Project Layout

```
semantic-negotiation-agent/
├── requirements.txt
├── app/
│   ├── __init__.py
│   └── agent/
│       ├── __init__.py
│       └── semantic_negotiation.py
└── tests/
    ├── conftest.py
    ├── integration/
    └── unit/
        └── test_semantic_negotiation.py
```

## Usage

```python
from app.agent.semantic_negotiation import SemanticNegotiationAgent

agent = SemanticNegotiationAgent()
result = agent.negotiate()
```

## Configuration

No configuration required for the skeleton. Extend `SemanticNegotiationAgent.__init__` with settings as the implementation evolves.

## Running

### Start the negotiation server

```bash
# Terminal 1
cd semantic-negotiation-agent && poetry run uvicorn app.main:app --host 0.0.0.0 --port 8089
```

### Run the callback agent test harness

```bash
# Terminal 2 (Python)
python test_callback_agents.py

# Terminal 2 (Go)
cd test_callback_agents_go && go run .
```
