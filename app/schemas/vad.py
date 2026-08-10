"""AI-01 VAD 절단점 계산의 요청·응답.

<h2>계약이 명세에 없다 — 이건 제안이다</h2>
AI-02~09 은 명세에 요청·응답 예시가 있지만 AI-01 은 한 줄짜리 표 항목("S3 키로 ±20초만
전달. onnxruntime 버전 silero-vad")과 동작 규칙(±20초 창·700ms 무음·FALLBACK_OVERLAP)뿐이다.
소비처인 Spring 블록 조립 경로가 아직 없어 맞춰볼 상대도 없다.

그래서 이 스키마는 **호출자와 합의되기 전의 제안**이다. 바뀌면 이 파일과 라우터만 고치면
된다 — 판정 로직(app/vad/cutpoint.py)은 계약과 무관하게 그대로다.
"""

from pydantic import Field

from app.schemas.common import CamelModel
from app.vad.cutpoint import DEFAULT_MIN_SILENCE_MS, DEFAULT_WINDOW_MS


class CutpointRequest(CamelModel):
    """절단점을 찾을 오디오와 목표 지점.

    <h2>임계값을 요청으로 연다</h2>
    700ms·±20초는 명세가 정한 초기값이지 불변값이 아니다. 서버 상수로 박으면 값을 조정할
    때마다 배포해야 하고, 그러면 회의별로 다른 값을 시험해 보는 것 자체가 불가능해진다.
    기본값은 명세값이라 안 보내면 명세대로 동작한다.
    """

    meeting_id: int
    s3_key: str
    # 버킷을 요청으로 받는다. 서버 설정에 두면 회사·환경마다 서버를 따로 띄워야 한다.
    bucket: str

    # 이 오프셋에서 자르고 싶다(= 10분 경계). 창의 중심이자 FALLBACK 시의 절단점이다.
    target_offset_ms: int = Field(ge=0)

    window_ms: int = Field(default=DEFAULT_WINDOW_MS, ge=1_000, le=120_000)
    min_silence_ms: int = Field(default=DEFAULT_MIN_SILENCE_MS, ge=100, le=10_000)


class CutpointResponse(CamelModel):
    """절단점과 그 근거.

    <h2>cutReason 이 응답의 절반이다</h2>
    stt_block.cut_reason 에 그대로 저장돼, 나중에 인식 품질이 나쁜 블록을 조사할 때 "말 중간을
    잘라서인가"를 가르는 유일한 단서가 된다. 값을 지어내면 그 조사가 성립하지 않는다.

    silenceMs 는 VAD_SILENCE 일 때만 값이 있다. FALLBACK 에서 0 을 채우면 "0ms 무음을 찾았다"로
    읽혀 못 찾은 것과 구분되지 않는다.
    """

    cut_offset_ms: int
    cut_reason: str
    silence_ms: int | None = None
