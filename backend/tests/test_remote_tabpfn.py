"""Remote TabPFN client + router: optional-by-construction, never decision-path.

No test here opens a network connection. httpx is monkeypatched at the module
under test, because what needs proving is the *contract*: unconfigured means a
calm "not configured" (not an exception), a dead tunnel means 503 (not 500),
and a malformed answer is rejected rather than trusted.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.database import Base
from backend.app.models import LoanApplication
from backend.app.routers import tabpfn as tabpfn_router
from backend.app.services import remote_tabpfn_service as svc


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()
    try:
        yield db
    finally:
        db.close()


def _configure(monkeypatch, url: str | None = "https://demo.trycloudflare.com", token: str | None = "tok"):
    monkeypatch.setattr(svc, "tabpfn_remote_url", lambda: url)
    monkeypatch.setattr(svc, "tabpfn_remote_token", lambda: token)


class _FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError("boom", request=None, response=SimpleNamespace(  # type: ignore[arg-type]
                status_code=self.status_code, text=self.text))


# ---------------------------------------------------------------------------
# service contract
# ---------------------------------------------------------------------------


def test_unconfigured_status_is_calm_not_an_error(monkeypatch):
    _configure(monkeypatch, url=None, token=None)
    status = svc.remote_status()
    assert status["configured"] is False
    assert status["reachable"] is False
    assert "SMARTLEND_TABPFN_URL" in status["detail"]


def test_unconfigured_score_raises_the_typed_error(monkeypatch):
    _configure(monkeypatch, url=None, token=None)
    with pytest.raises(svc.RemoteTabPFNError):
        svc.score_rows([{"age": 30}])


def test_score_happy_path_and_token_header(monkeypatch):
    _configure(monkeypatch)
    captured: dict[str, Any] = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(url=url, json=json, headers=headers)
        return _FakeResponse({"predictions": [{"p_default": 0.2}], "seconds": 1.0})

    monkeypatch.setattr(svc.httpx, "post", fake_post)
    out = svc.score_rows([{"age": 30}])
    assert out == [{"p_default": 0.2, "p_approve": 0.8}]
    assert captured["url"].endswith("/score")
    assert captured["headers"]["X-SmartLend-Token"] == "tok"


def test_score_rejects_count_mismatch_and_bad_probabilities(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setattr(
        svc.httpx, "post", lambda *a, **k: _FakeResponse({"predictions": []})
    )
    with pytest.raises(svc.RemoteTabPFNError, match="predictions"):
        svc.score_rows([{"age": 30}])

    monkeypatch.setattr(
        svc.httpx, "post", lambda *a, **k: _FakeResponse({"predictions": [{"p_default": 1.7}]})
    )
    with pytest.raises(svc.RemoteTabPFNError, match="out-of-range"):
        svc.score_rows([{"age": 30}])


def test_score_refuses_oversized_batches(monkeypatch):
    _configure(monkeypatch)
    with pytest.raises(svc.RemoteTabPFNError, match="max"):
        svc.score_rows([{"age": 30}] * (svc.MAX_ROWS_PER_CALL + 1))


def test_empty_rows_short_circuit_without_network(monkeypatch):
    _configure(monkeypatch)

    def explode(*a, **k):  # pragma: no cover - the point is it is never called
        raise AssertionError("network must not be touched for zero rows")

    monkeypatch.setattr(svc.httpx, "post", explode)
    assert svc.score_rows([]) == []


# ---------------------------------------------------------------------------
# router contract
# ---------------------------------------------------------------------------


def _stored_application(session) -> LoanApplication:
    item = LoanApplication(
        applicant_id="cust-tabpfn",
        input_data={
            "age": 34,
            "_decision_meta": {"engineered_features": {"age": 34.0, "annual_income": 900000.0}},
        },
        ml_prob=0.91,
        cbes_prob=0.55,
        final_decision="APPROVE",
        confidence=0.8,
        documents=[],
    )
    session.add(item)
    session.commit()
    session.refresh(item)
    return item


def test_second_opinion_503_when_remote_is_down(monkeypatch, session):
    item = _stored_application(session)

    def down(rows):
        raise svc.RemoteTabPFNError("tunnel is closed")

    monkeypatch.setattr(tabpfn_router.remote_tabpfn_service, "score_rows", down)
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        tabpfn_router.tabpfn_second_opinion(item.id, db=session)
    assert excinfo.value.status_code == 503


def test_second_opinion_scores_stored_features_and_persists(monkeypatch, session):
    item = _stored_application(session)
    seen: dict[str, Any] = {}

    def fake_score(rows):
        seen["rows"] = rows
        return [{"p_default": 0.25, "p_approve": 0.75}]

    monkeypatch.setattr(tabpfn_router.remote_tabpfn_service, "score_rows", fake_score)
    payload = tabpfn_router.tabpfn_second_opinion(item.id, db=session)

    # It scored the engineered (post-imputation) row the decision was made on.
    assert seen["rows"] == [{"age": 34.0, "annual_income": 900000.0}]
    opinion = payload["secondOpinion"]
    assert opinion["pApprove"] == 0.75
    assert opinion["advisoryOnly"] is True
    assert opinion["delta"] == pytest.approx(0.75 - 0.91, abs=1e-6)

    # Persisted next to the decision, decision untouched.
    session.refresh(item)
    stored = item.input_data["_decision_meta"]["tabpfn_second_opinion"]
    assert stored["pDefault"] == 0.25
    assert item.final_decision == "APPROVE"


def test_second_opinion_404_for_unknown_application(session):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        tabpfn_router.tabpfn_second_opinion("app-does-not-exist", db=session)
    assert excinfo.value.status_code == 404
