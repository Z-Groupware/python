"""계층 공통 DTO.

API 표면은 camelCase 다(명세 기준). 파이썬 쪽은 snake_case 로 쓰고 alias 로 변환한다.
`populate_by_name=True` 라 테스트에서는 snake_case 로도 만들 수 있다.
"""

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
    )


class Usage(CamelModel):
    """계층별 토큰. 선택 항목이 아니다 — 없으면 QLTY-03(비용)이 성립하지 않고
    특화 모델 전환의 손익분기점을 계산할 수 없다."""

    tokens_in: int = 0
    tokens_out: int = 0


class Participant(CamelModel):
    """닫힌 목록의 원소. person_id 가 None 인 항목이 `unknown_person` 탈출구다
    (게스트·외부 참석자). 탈출구가 없으면 모델이 명단 안에서 억지로 하나를 고른다."""

    person_id: int | None = None
    name: str


class Utterance(CamelModel):
    """근거 발화 후보. 이 목록의 id 만 evidence 로 반환될 수 있다(근거 강제)."""

    utterance_id: int
    speaker_id: int | None = None
    start_ms: int | None = None
    text: str


class FewShotExample(CamelModel):
    """AI-09 가 돌려준 예시. Spring 이 따로 조회하지 않고 Python 이 내부에서 부르므로,
    무엇을 썼는지 응답에 실어 보내야 review_log.input_context 가 온전해진다."""

    input_text: str
    payload: dict
    score: float | None = None


class LayerMeta(CamelModel):
    """모든 계층 응답에 공통으로 붙는다. 프롬프트를 바꾼 뒤 정확도 변화를 추적하는 키다."""

    usage: Usage = Field(default_factory=Usage)
    model: str
    prompt_version: str
