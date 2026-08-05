"""내부 API — Spring EC2 → AI EC2 (AI-01 ~ AI-10).

전부 `X-Internal-Token` 을 요구한다. 아직 구현되지 않은 계층은 501 로 명확히 거절한다.
200 에 빈 결과를 돌려주면 Spring 오케스트레이션이 "계층이 정상 완료했고 산출물이 없다"로
기록해 버려서, 미구현이 품질 문제로 위장된다.
"""

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from app.clients.gemini import GeminiClient
from app.config import Settings, get_settings
from app.layers import l1_5, l2, l3, l4
from app.layers.runner import LayerRunner
from app.schemas.l1_5 import ResolveReferenceRequest, ResolveReferenceResponse
from app.schemas.l2 import SegmentTopicsRequest, SegmentTopicsResponse
from app.schemas.l3 import SummarizeTopicRequest, SummarizeTopicResponse
from app.schemas.l4 import ExtractTuplesRequest, ExtractTuplesResponse
from app.security import require_internal_token

router = APIRouter(
    prefix="/internal",
    dependencies=[Depends(require_internal_token)],
)

# AI-10 이 돌려주는 목록. 라우팅과 따로 관리하면 하나를 붙이고 다른 하나를 잊는다 —
# 그러면 워커가 "구현됐다"를 보고 호출했다가 501 을 받는다.
IMPLEMENTED = ["AI-02", "AI-03", "AI-04", "AI-06", "AI-10"]


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


# ── 계층 (파이프라인 순서대로) ────────────────────────────────────────────────
# 전부 l4.py 의 틀을 복제한 것이다 — SPEC · 응답 스키마 · 후처리 세 곳만 다르다.


# AI-02 · L1.5 지시어 해소 — L4 의 담당자 판정이 여기 의존한다.
@router.post("/layers/l1-5/resolve-reference", response_model=ResolveReferenceResponse)
async def resolve_reference(
    request: ResolveReferenceRequest,
    runner: LayerRunner = Depends(get_runner),
) -> ResolveReferenceResponse:
    return await l1_5.resolve_reference(request, runner)


# AI-03 · L2 주제 분할 — 오버랩 3발화는 응답 후처리가 붙인다(프롬프트 부탁이 아니다).
@router.post("/layers/l2/segment-topics", response_model=SegmentTopicsResponse)
async def segment_topics(
    request: SegmentTopicsRequest,
    runner: LayerRunner = Depends(get_runner),
) -> SegmentTopicsResponse:
    return await l2.segment_topics(request, runner)


# AI-04 · L3 주제별 정리 — 주제마다 한 번씩 호출된다(명세 「주제별 N회」).
@router.post("/layers/l3/summarize-topic", response_model=SummarizeTopicResponse)
async def summarize_topic(
    request: SummarizeTopicRequest,
    runner: LayerRunner = Depends(get_runner),
) -> SummarizeTopicResponse:
    return await l3.summarize_topic(request, runner)


# AI-06 · L4 tuple 추출 — 다른 계층의 원본 틀.
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
        "implemented": IMPLEMENTED,
    }


# ── 미구현 (일정 순서대로 붙는다) ──────────────────────────────────────────────
@router.post("/vad/cutpoint")
async def vad_cutpoint() -> JSONResponse:
    return _not_implemented("AI-01", "VAD 절단점 계산", "8/7")


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
