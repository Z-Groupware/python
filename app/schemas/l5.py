"""AI-07 · L5 — 관점 다변화 검증 DTO.

**다수결을 반환하지 않는다.** 같은 모델에 같은 프롬프트를 여러 번 돌리면 오류가 서로
상관되어, 틀릴 때도 똑같이 틀린다. 그 100% 일치가 오답에 최고 신뢰도를 부여한다.

대신 **관점을 바꿔서 묻는다** — `EXTRACT_NARROW`(앞뒤 3발화만 보고 다시 뽑기) vs
`VERIFY`("이 tuple 이 맞나?"). 생성과 검증은 다른 작업이라 오류가 덜 겹친다.
불일치는 다수결로 덮지 않고 `agree=false` 로 넘긴다 — 앙상블의 가치는 정답 선택이
아니라 **불확실성 탐지**다.
"""

from datetime import date
from typing import Literal

from pydantic import Field

from app.schemas.common import CamelModel, Participant, Usage, Utterance
from app.schemas.l4 import AssignmentTuple, TopicItem

View = Literal["EXTRACT_NARROW", "VERIFY"]
Verdict = Literal["ACCEPT", "REJECT"]

REASON_MAX = 500

# 두 관점이 비교하는 필드. dueDate 를 넣는 이유는 틀린 마감이 그대로 보드에 꽂히기 때문이다.
COMPARED_FIELDS = ("title", "assigneeCandidatePersonId", "dueDate")


class VerifyRequest(CamelModel):
    """검증 대상 tuple 하나 + 그것을 다시 뽑는 데 필요한 문맥.

    문맥을 통째로 받는 이유는 Python 이 **내부에서 L4 를 한 번 더 돌리기** 때문이다.
    Spring 이 두 관점을 각각 호출해 결과를 모아 넘기면 인스턴스 간 왕복이 두 번이 되고,
    관점 조합 규칙(한쪽 실패 시 안전한 쪽)이 Spring 과 Python 두 곳에 생긴다.
    """

    tenant_id: int
    meeting_id: int
    topic: str

    # 검증 대상 — L4(view=EXTRACT)가 뽑은 기준 tuple.
    tuple: AssignmentTuple

    items: list[TopicItem] = Field(default_factory=list)
    utterances: list[Utterance]
    participants: list[Participant]
    meeting_date: date | None = None
    query_text: str | None = None


class ViewResult(CamelModel):
    """관점 하나의 결과. 관점마다 채워지는 필드가 다르다(명세 응답 예시와 같은 모양)."""

    view: View

    # EXTRACT_NARROW — 좁은 시야로 다시 뽑은 tuple. 재현되지 않았으면 None 이다.
    tuple: AssignmentTuple | None = None

    # VERIFY — 기준 tuple 이 근거 발화로 확인되는가.
    verdict: Verdict | None = None
    reason: str | None = None

    # 이 관점이 실패했다. 실패를 숨기지 않는 이유는 agree=false 의 원인이 '불일치'인지
    # '한쪽이 죽음'인지가 다르기 때문이다 — 후자가 계속 쌓이면 고칠 것은 코드다.
    error: str | None = None

    # 관점별 토큰. 최상단 usage 는 이 둘의 합이다. 나눠서도 두는 이유는 두 관점의
    # 비용이 크게 다르기 때문이다 — VERIFY 는 발화를 좁혀 보내므로 훨씬 싸고,
    # 합계만 있으면 어느 쪽을 줄여야 하는지 판단할 수 없다.
    usage: Usage = Field(default_factory=Usage)


class VerifyResponse(CamelModel):
    agree: bool

    # 두 관점이 갈린 필드. agree=true 면 비어 있다.
    disagreement_fields: list[str] = Field(default_factory=list)

    results: list[ViewResult] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    model: str
    prompt_version: str
