"""Reviewer-briefing agent: advisory-only, optional-by-construction, grounded.

No test opens a network connection — the Anthropic client is monkeypatched.
What needs proving is the contract: unconfigured means 503-shaped errors (not
crashes), a valid structured response comes back stamped advisory-only and is
persisted NEXT TO the decision, and the decision itself is never touched.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.database import Base
from backend.app.models import LoanApplication
from backend.app.routers import agent as agent_router
from backend.app.services import underwriting_agent_service as svc


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()
    try:
        yield db
    finally:
        db.close()


BRIEFING = {
    "summary": "Applicant shows stable income but a thin credit file.",
    "riskFactors": ["Thin bureau file"],
    "mitigants": ["Long employment tenure"],
    "dataQualityFlags": ["monthly_income and annual_income disagree"],
    "suggestedChecks": ["Verify income documents"],
    "suggestedReasonCodes": ["REJ-THIN-FILE", "APR-EMP-STABLE"],
}


def _fake_response(text: str, stop_reason: str = "end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        model="claude-opus-5",
        content=[SimpleNamespace(type="text", text=text)],
    )


def _fake_client(monkeypatch, response):
    captured: dict[str, Any] = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            if isinstance(response, Exception):
                raise response
            return response

    monkeypatch.setattr(svc, "_client", lambda: SimpleNamespace(messages=FakeMessages()))
    return captured


def test_unconfigured_raises_unavailable(monkeypatch):
    monkeypatch.setattr(svc, "anthropic_configured", lambda: False)
    with pytest.raises(svc.AgentUnavailableError, match="ANTHROPIC_API_KEY"):
        svc.generate_briefing({"engine": {}})


def test_happy_path_returns_stamped_briefing(monkeypatch):
    monkeypatch.setattr(svc, "anthropic_configured", lambda: True)
    captured = _fake_client(monkeypatch, _fake_response(json.dumps(BRIEFING)))

    briefing = svc.generate_briefing({"application": {"loanAmount": 500000}, "engine": {"pMl": 0.9}})

    assert briefing["summary"] == BRIEFING["summary"]
    assert briefing["advisoryOnly"] is True
    assert briefing["model"] == "claude-opus-5"
    # Structured output requested, taxonomy included in the prompt.
    assert captured["output_config"]["format"]["type"] == "json_schema"
    assert "REJ-THIN-FILE" in captured["messages"][0]["content"]
    assert captured["model"] == svc.MODEL


def test_refusal_and_bad_json_are_briefing_errors(monkeypatch):
    monkeypatch.setattr(svc, "anthropic_configured", lambda: True)

    _fake_client(monkeypatch, _fake_response("", stop_reason="refusal"))
    with pytest.raises(svc.AgentBriefingError, match="declined"):
        svc.generate_briefing({"engine": {}})

    _fake_client(monkeypatch, _fake_response("not json at all"))
    with pytest.raises(svc.AgentBriefingError, match="non-JSON"):
        svc.generate_briefing({"engine": {}})


# ---------------------------------------------------------------------------
# router contract
# ---------------------------------------------------------------------------


def _stored_application(session) -> LoanApplication:
    item = LoanApplication(
        applicant_id="cust-agent",
        input_data={"firstName": "Brief", "lastName": "Case", "loanAmount": 500000},
        ml_prob=0.88,
        cbes_prob=0.61,
        final_decision="DEFER",
        confidence=0.4,
        documents=[],
    )
    session.add(item)
    session.commit()
    session.refresh(item)
    return item


def test_router_503_when_agent_unavailable(monkeypatch, session):
    item = _stored_application(session)

    def unavailable(report):
        raise svc.AgentUnavailableError("no key")

    monkeypatch.setattr(agent_router.underwriting_agent_service, "generate_briefing", unavailable)
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        agent_router.agent_briefing(item.id, db=session)
    assert excinfo.value.status_code == 503


def test_router_persists_briefing_next_to_decision(monkeypatch, session):
    item = _stored_application(session)
    seen: dict[str, Any] = {}

    def fake_generate(report):
        seen["report"] = report
        return dict(BRIEFING, advisoryOnly=True, model="claude-opus-5")

    monkeypatch.setattr(agent_router.underwriting_agent_service, "generate_briefing", fake_generate)
    payload = agent_router.agent_briefing(item.id, db=session)

    # The agent saw the real decision report for this application.
    assert seen["report"]["applicationId"] == item.id
    assert payload["briefing"]["advisoryOnly"] is True
    assert payload["briefing"]["generatedAt"]

    # Persisted under _decision_meta; the decision itself untouched.
    session.refresh(item)
    stored = item.input_data["_decision_meta"]["agent_briefing"]
    assert stored["summary"] == BRIEFING["summary"]
    assert item.final_decision == "DEFER"


def test_router_404_for_unknown_application(session):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        agent_router.agent_briefing("app-nope", db=session)
    assert excinfo.value.status_code == 404
