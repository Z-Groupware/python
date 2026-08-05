"""내부 API 인증 — 토큰 없이 통과하는 경로가 없어야 한다.

보안그룹으로 인바운드를 좁혀도 이 테스트를 둔다. 같은 VPC 안의 다른 인스턴스가
뚫렸을 때 방어선이 하나도 없는 상태가 되지 않게.
"""

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app

TOKEN = "test-internal-token"


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    monkeypatch.setenv("INTERNAL_TOKEN", TOKEN)
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("DRY_RUN", "1")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client():
    return TestClient(app)


def test_토큰_없으면_401(client):
    response = client.get("/internal/health")

    assert response.status_code == 401


def test_틀린_토큰이면_401(client):
    response = client.get("/internal/health", headers={"X-Internal-Token": "wrong"})

    assert response.status_code == 401


def test_올바른_토큰이면_통과(client):
    response = client.get("/internal/health", headers={"X-Internal-Token": TOKEN})

    assert response.status_code == 200
    assert response.json()["status"] == "UP"


def test_인프라_liveness는_무인증(client):
    # ALB·컨테이너 헬스체크는 토큰을 들고 있을 수 없다.
    response = client.get("/health")

    assert response.status_code == 200


def test_미구현_계층은_501로_거절한다(client):
    response = client.post("/internal/layers/l2/segment-topics", headers={"X-Internal-Token": TOKEN})

    # 200 + 빈 결과로 두면 Spring 이 "계층 정상 완료, 산출물 없음"으로 기록해
    # 미구현이 품질 문제로 위장된다.
    assert response.status_code == 501
    assert response.json()["api"] == "AI-03"


def test_L4는_발화가_없으면_모델을_부르지_않는다(client):
    body = {
        "tenantId": 7,
        "meetingId": 500,
        "topic": "제품 로드맵",
        "items": [],
        "utterances": [],
        "participants": [{"personId": 7, "name": "김서준"}],
        "view": "EXTRACT",
    }

    response = client.post(
        "/internal/layers/l4/extract-tuples",
        headers={"X-Internal-Token": TOKEN},
        json=body,
    )

    # GEMINI_API_KEY 가 비어 있어도 200 이어야 한다 = 호출 자체가 없었다는 뜻.
    assert response.status_code == 200
    assert response.json()["tuples"] == []
