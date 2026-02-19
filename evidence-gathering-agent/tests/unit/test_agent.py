import pytest
from app.agent.service import EvidenceService


class FakeRepo:
    async def fetch_records(self, query):
        return [{"value": 1}, {"value": 2}]


@pytest.mark.asyncio
async def test_agent_processing():
    svc = EvidenceService(FakeRepo())
    resp = await svc.handle_request(type("Req", (), {"reasoner_cognition_request_id": "t", "intent": "demo"}))
    assert resp.status == "OK"
