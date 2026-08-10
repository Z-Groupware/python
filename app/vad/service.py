"""AI-01 조립 — S3 → 디코딩 → 모델 → 판정.

네 단계를 여기서만 잇는다. 각 단계가 자기 파일에 있고 이 파일에는 순서와 실패 처리만 있다 —
그래야 판정 규칙(cutpoint.py)을 오디오 없이, 디코딩(audio.py)을 모델 없이 테스트할 수 있다.
"""

from __future__ import annotations

import logging

from app.clients import s3
from app.config import Settings
from app.schemas.vad import CutpointRequest, CutpointResponse
from app.vad import silero
from app.vad.audio import FRAME_MS, decode_window
from app.vad.cutpoint import CUT_FALLBACK_OVERLAP, choose_cutpoint, window_bounds

log = logging.getLogger(__name__)


async def find_cutpoint(request: CutpointRequest, settings: Settings) -> CutpointResponse:
    """절단점을 찾는다. 못 찾는 것은 실패가 아니다."""
    window_start_ms, window_end_ms = window_bounds(request.target_offset_ms, request.window_ms)
    duration_ms = window_end_ms - window_start_ms

    audio = await s3.fetch(request.bucket, request.s3_key, settings.s3_region)
    pcm = await decode_window(audio, window_start_ms, duration_ms)

    probs = silero.speech_probs(pcm, settings.vad_model_path)
    if not probs:
        # 창에 프레임이 하나도 없다 — 파일이 그 지점보다 짧다. 목표에서 자르고 이유를 남긴다.
        log.warning(
            "VAD 창이 비어 있다 — meetingId=%s key=%s window=%s..%s",
            request.meeting_id,
            request.s3_key,
            window_start_ms,
            window_end_ms,
        )
        return CutpointResponse(cut_offset_ms=request.target_offset_ms, cut_reason=CUT_FALLBACK_OVERLAP)

    cut = choose_cutpoint(
        probs,
        frame_ms=FRAME_MS,
        window_start_ms=window_start_ms,
        target_offset_ms=request.target_offset_ms,
        min_silence_ms=request.min_silence_ms,
    )

    log.info(
        "VAD 절단점 — meetingId=%s target=%sms cut=%sms reason=%s silence=%s",
        request.meeting_id,
        request.target_offset_ms,
        cut.cut_offset_ms,
        cut.cut_reason,
        cut.silence_ms,
    )
    return CutpointResponse(
        cut_offset_ms=cut.cut_offset_ms, cut_reason=cut.cut_reason, silence_ms=cut.silence_ms
    )
