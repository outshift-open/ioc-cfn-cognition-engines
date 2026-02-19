from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_evidence_endpoint_smoke():
    payload = {
        "reasoner_cognition_request_id": "demo",
        "intent": "what does the orchestrator do?",
        "records": [],
        "meta": {},
    }
    r = client.post("/api/v1/reasoning/evidence", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "records" in body
