"""계층 공통 러너 — 프롬프트 파일 + 응답 스키마를 받아 한 번 호출한다.

계층이 9개(L1.5·L2·L3·L3.5·L4·L5 + 확장)라 계층마다 호출·재시도·토큰 집계를 따로
쓰면 같은 버그를 아홉 번 고치게 된다. 여기 한 곳에 두고, 계층은 다음 셋만 준다.

    1. 프롬프트 파일 (app/prompts/*.txt)
    2. 응답 스키마 (dict — 참석자·발화 목록을 enum 으로 박은 것)
    3. 후처리 (파싱된 dict → 계층 DTO)

프롬프트 치환은 `{{NAME}}` 토큰을 문자열 치환한다. `str.format` 은 프롬프트 안의
JSON 예시 중괄호를 인자로 오인해 터진다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.clients.gemini import GeminiClient
from app.config import Settings
from app.errors import LayerError, LayerErrorKind
from app.schemas.common import Usage

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
_PLACEHOLDER = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


@dataclass(frozen=True)
class LayerSpec:
    """계층 하나의 정체. prompt_version 은 파일명과 함께 올린다 —
    프롬프트를 고쳤는데 버전을 안 올리면 지표 변화의 원인을 되짚을 수 없다."""

    layer: str
    prompt_file: str
    prompt_version: str
    dry_run_payload: dict


@lru_cache
def load_prompt(prompt_file: str) -> str:
    path = PROMPT_DIR / prompt_file
    if not path.is_file():
        raise LayerError(
            LayerErrorKind.PERMANENT,
            "PROMPT_MISSING",
            f"프롬프트 파일이 없습니다: {prompt_file}",
        )
    return path.read_text(encoding="utf-8")


def render_prompt(template: str, variables: dict[str, str]) -> str:
    rendered = template
    for key, value in variables.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)

    # 남은 토큰이 있으면 프롬프트와 호출부가 어긋난 것이다. 그대로 보내면
    # 모델에게 "{{UTTERANCES}}" 라는 문자열을 근거로 답하라고 시키는 셈이 된다.
    leftover = _PLACEHOLDER.findall(rendered)
    if leftover:
        raise LayerError(
            LayerErrorKind.PERMANENT,
            "PROMPT_UNRESOLVED",
            f"치환되지 않은 자리표시자: {sorted(set(leftover))}",
        )
    return rendered


class LayerRunner:
    def __init__(self, client: GeminiClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    @property
    def model_name(self) -> str:
        """계층 응답의 `model` 필드에 그대로 실린다 — 어느 모델이 만든 라벨인지 없으면
        모델을 바꾼 뒤 정확도 변화를 모델 탓으로 돌릴 수도, 아닐 수도 없다."""
        return self._settings.gemini_model

    @property
    def dry_run(self) -> bool:
        return self._settings.dry_run

    async def run(
        self,
        spec: LayerSpec,
        *,
        variables: dict[str, str],
        response_schema: dict,
    ) -> tuple[dict, Usage]:
        if self._settings.dry_run:
            # 라우팅·스키마만 확인하는 모드. 토큰을 태우지 않는다.
            return spec.dry_run_payload, Usage()

        prompt = render_prompt(load_prompt(spec.prompt_file), variables)
        return await self._client.generate_json(prompt=prompt, response_schema=response_schema)
