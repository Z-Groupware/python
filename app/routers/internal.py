"""내부 API — Spring EC2 → AI EC2 (AI-01 ~ AI-10).

전부 `X-Internal-Token` 을 요구한다. 아직 구현되지 않은 계층은 501 로 명확히 거절한다.
200 에 빈 결과를 돌려주면 Spring 오케스트레이션이 "계층이 정상 완료했고 산출물이 없다"로
기록해 버려서, 미구현이 품질 문제로 위장된다.
"""

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from app.clients.gemini import GeminiClient
from app.config import Settings, get_settings
from app.layers import l4
from app.layers.runner import LayerRunner
from app.schemas.l4 import ExtractTuplesRequest, ExtractTuplesResponse
from app.security import require_internal_token

router = APIRouter(
    prefix="/internal",
    dependencies=[Depends(require_internal_token)],
)


def get_runner(settings: Settings = Depends(get_settings)) -> LayerRunner:
    return LayerRunner(GeminiClient(settings), settings)


def _not_implemented(api_id: str, name: str, planned: str) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        content={
            "code": "LAYER_NOT_IMPLEMENTED",
            "api": api_id,
            "name": name,
            "planned": planned,
            "message": f"{api_id}({name}) 는 아직 구현되지 않았습니다. 예정: {planned}",
        },
    )


# ── AI-06 · L4 tuple 추출 — 구현됨 (다른 계층의 원본 틀) ────────────────────────
@router.post("/layers/l4/extract-tuples", response_model=ExtractTuplesResponse)
async def extract_tuples(
    request: ExtractTuplesRequest,
    runner: LayerRunner = Depends(get_runner),
) -> ExtractTuplesResponse:
    return await l4.extract_tuples(request, runner)


# ── AI-10 · 헬스체크 — 워커 백오프 판단용 ──────────────────────────────────────
# 인프라 liveness(/health, 무인증)와 구분한다. 이쪽은 "계층을 받을 준비가 됐는지"이므로
# 토큰을 요구하고 모델 설정 여부까지 본다 — 키가 없는 채로 살아 있으면 워커가
# 계속 태우다 전부 실패한다.
@router.get("/health")
async def internal_health(settings: Settings = Depends(get_settings)) -> dict:
    return {
        "status": "UP",
        "model": settings.gemini_model,
        "geminiConfigured": bool(settings.gemini_api_key),
        "dryRun": settings.dry_run,
        "implemented": ["AI-06", "AI-10"],
    }


# ── 미구현 (일정 순서대로 붙는다) ──────────────────────────────────────────────
@router.post("/vad/cutpoint")
async def vad_cutpoint() -> JSONResponse:
    return _not_implemented("AI-01", "VAD 절단점 계산", "8/7")


@router.post("/layers/l1-5/resolve-reference")
async def resolve_reference() -> JSONResponse:
    return _not_implemented("AI-02", "L1.5 지시어 해소", "8/6")


@router.post("/layers/l2/segment-topics")
async def segment_topics() -> JSONResponse:
    return _not_implemented("AI-03", "L2 주제 분할", "8/6")


@router.post("/layers/l3/summarize-topic")
async def summarize_topic() -> JSONResponse:
    return _not_implemented("AI-04", "L3 주제별 정리", "8/6")


@router.post("/layers/l3-5/gate")
async def gate() -> JSONResponse:
    return _not_implemented("AI-05", "L3.5 확정/논의 게이트", "8/6")


@router.post("/layers/l5/verify")
async def verify() -> JSONResponse:
    return _not_implemented("AI-07", "L5 관점 다변화 검증", "8/6")


@router.post("/vector/upsert")
async def vector_upsert() -> JSONResponse:
    return _not_implemented("AI-08", "벡터 저장", "8/9")


@router.post("/similar")
async def similar() -> JSONResponse:
    return _not_implemented("AI-09", "유사 발화 조회", "8/9")
