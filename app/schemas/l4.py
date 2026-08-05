"""AI-06 · L4 — assignment tuple 추출 DTO.

L4 를 첫 계층으로 짜는 이유는 여기서 잡은 틀(요청 DTO → 프롬프트 → 응답 스키마 →
후처리)을 L1.5 · L2 · L3 · L3.5 · L5 가 그대로 복제하기 때문이다. 계층마다 다른 건
프롬프트 파일과 응답 모델뿐이다.
"""

from typing import Literal

from pydantic import Field

from app.schemas.common import CamelModel, FewShotExample, Participant, Usage, Utterance

# 명단 밖 담당자를 가리키는 탈출구. 닫힌 목록에 이게 없으면 모델이 참석자 중
# 아무나 하나를 억지로 고르고, 그 값은 검증도 안 된다.
UNKNOWN_PERSON = "unknown_person"

# assigneeSource 를 nullable 로 두는 대신 enum 안에 넣는다 — 구조화 출력에서
# "nullable + enum" 조합은 제공자마다 처리가 갈려서, 값으로 받고 후처리에서 None 으로 바꾼다.
SOURCE_UNKNOWN = "UNKNOWN"


class TopicItem(CamelModel):
    """L3.5 게이트를 통과한(gateStatus=CONFIRMED) 정리 항목."""

    item_type: Literal["DECISION", "DISCUSSION", "BLOCKER"]
    content: str
    evidence_utterance_ids: list[int] = Field(default_factory=list)


class ExtractTuplesRequest(CamelModel):
    """명세의 요청에 `utterances` 를 더했다.

    명세 예시에는 `items` 만 있는데, 그것만으로는 `evidenceUtteranceId` 를 닫힌
    목록으로 강제할 수 없다. 후보 발화 목록을 함께 받아 그 id 들만 enum 으로 넣으면
    없는 발화를 근거로 붙이는 경로가 구조적으로 사라진다.
    """

    tenant_id: int
    meeting_id: int
    topic: str
    items: list[TopicItem]
    utterances: list[Utterance]
    participants: list[Participant]
    query_text: str | None = None

    # EXTRACT        주제 전체를 보고 뽑기 (기본)
    # EXTRACT_NARROW 근거 발화 앞뒤 3발화만 보고 다시 뽑기 — L5 관점 다변화의 한쪽
    view: Literal["EXTRACT", "EXTRACT_NARROW"] = "EXTRACT"


class AssignmentTuple(CamelModel):
    title: str
    assignee_candidate_person_id: int | None = None
    assignee_source: Literal["EXPLICIT_CALL", "FIRST_PERSON"] | None = None
    due_date: str | None = None
    evidence_utterance_id: int


class ExtractTuplesResponse(CamelModel):
    tuples: list[AssignmentTuple] = Field(default_factory=list)
    used_few_shot: list[FewShotExample] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    model: str
    prompt_version: str
