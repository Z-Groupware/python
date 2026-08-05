"""Gemini 호출 — 구조화 출력 전용.

프롬프트로 "JSON 으로 답해줘"라고 부탁하지 않는다. 응답 스키마를 붙여 구조로 강제한다.
부탁과 강제는 다르다 — 부탁은 대체로 지켜지지만 '대체로'가 파이프라인 중간에 있으면
뒤 계층 전체가 그 확률만큼 깨진다.

temperature=0 으로 고정한다. 품질 지표를 gold set 대비로 재려면 같은 입력에 같은
출력이 나와야 한다. 온도가 있으면 프롬프트를 안 바꿨는데도 점수가 흔들린다.
"""

from __future__ import annotations

import asyncio
import json
import random

from app.config import Settings
from app.errors import LayerError, LayerErrorKind, classify_provider_error
from app.schemas.common import Usage


class GeminiClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = None  # 최초 호출 때 만든다 (DRY_RUN 이면 아예 안 만든다)

    def _ensure_client(self):
        if self._client is None:
            if not self._settings.gemini_api_key:
                raise LayerError(
                    LayerErrorKind.PERMANENT,
                    "GEMINI_KEY_MISSING",
                    "GEMINI_API_KEY 가 설정되지 않았습니다.",
                )
            from google import genai  # 지연 임포트 — DRY_RUN 로컬 실행에 SDK 가 필요 없게

            self._client = genai.Client(api_key=self._settings.gemini_api_key)
        return self._client

    async def generate_json(self, *, prompt: str, response_schema: dict) -> tuple[dict, Usage]:
        """실패 정책을 여기 한 곳에 둔다. 계층 코드가 재시도를 각자 구현하면
        어떤 계층은 재시도하고 어떤 계층은 안 하는 상태가 조용히 생긴다."""
        from google.genai import types

        client = self._ensure_client()
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=0,
        )

        attempts = len(self._settings.retry_delays_sec) + 1
        last: LayerError | None = None

        for attempt in range(attempts):
            try:
                response = await client.aio.models.generate_content(
                    model=self._settings.gemini_model,
                    contents=prompt,
                    config=config,
                )
                return self._parse(response)
            except LayerError:
                raise
            # SDK 예외 계층에 기대지 않고 상태코드·메시지로 분류한다 — SDK 버전이 올라가며
            # 예외 클래스가 바뀌어도 재시도 판정이 조용히 무너지지 않게.
            except Exception as exc:
                last = classify_provider_error(exc)
                if not last.retryable or attempt == attempts - 1:
                    raise last from exc
                await asyncio.sleep(self._delay_for(attempt, last))

        raise last if last else LayerError(LayerErrorKind.TRANSIENT, "PROVIDER_UNAVAILABLE", "재시도 소진")

    def _delay_for(self, attempt: int, error: LayerError) -> float:
        # 429 는 서버가 알려준 대기시간을 존중한다. 우리 백오프를 우선하면
        # 레이트리밋이 안 풀린 상태로 다시 때려 쿼터만 더 태운다.
        if error.kind is LayerErrorKind.RATE_LIMIT and error.retry_after_sec:
            return error.retry_after_sec

        base = self._settings.retry_delays_sec[attempt]
        # 지터 — 여러 계층이 동시에 실패하면 재시도가 같은 순간에 겹쳐 다시 실패한다.
        ratio = self._settings.retry_jitter_ratio
        return base * random.uniform(1 - ratio, 1 + ratio)

    @staticmethod
    def _parse(response) -> tuple[dict, Usage]:
        meta = getattr(response, "usage_metadata", None)
        usage = Usage(
            tokens_in=getattr(meta, "prompt_token_count", 0) or 0,
            tokens_out=getattr(meta, "candidates_token_count", 0) or 0,
        )

        text = (getattr(response, "text", None) or "").strip()
        if not text:
            # 스키마를 붙였는데도 빈 응답이면 세이프티 차단 등 입력 자체의 문제다.
            # 같은 입력으로 다시 보내도 같은 결과라 재시도하지 않는다.
            raise LayerError(LayerErrorKind.PERMANENT, "EMPTY_RESPONSE", "구조화 응답이 비어 있습니다.")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LayerError(LayerErrorKind.PERMANENT, "SCHEMA_INVALID", f"JSON 파싱 실패: {exc}") from exc

        if not isinstance(parsed, dict):
            raise LayerError(LayerErrorKind.PERMANENT, "SCHEMA_INVALID", "최상위가 객체가 아닙니다.")
        return parsed, usage
