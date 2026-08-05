"""L4 · assignment tuple 추출 (AI-06).

이 파일이 나머지 계층의 원본이다. 새 계층을 추가할 때 복제해서 바꾸는 것은
① SPEC(프롬프트 파일·버전) ② build_response_schema ③ 후처리 세 곳뿐이다.

핵심은 정확도 4원칙을 **프롬프트가 아니라 응답 스키마로** 거는 것이다.
프롬프트의 "목록에서만 고르세요"는 지켜지지 않을 확률이 남지만,
enum 으로 박으면 목록 밖 값이 나올 방법 자체가 없다.
"""

from __future__ import annotations

import json
from datetime import date

from app.layers import few_shot
from app.layers.runner import LayerRunner, LayerSpec
from app.schemas.common import FewShotExample, Participant, Utterance
from app.schemas.l4 import (
    SOURCE_UNKNOWN,
    UNKNOWN_PERSON,
    AssignmentTuple,
    ExtractTuplesRequest,
    ExtractTuplesResponse,
    TopicItem,
)

SPEC = LayerSpec(
    layer="L4",
    prompt_file="l4_extract_tuples.v1.txt",
    prompt_version="v1",
    dry_run_payload={"tuples": []},
)

_VIEW_INSTRUCTION = {
    # 기본 관점 — 주제 전체를 보고 뽑는다.
    "EXTRACT": "주제 전체를 읽고 배정을 모두 뽑는다.",
    # L5 관점 다변화의 한쪽. 같은 프롬프트를 두 번 돌리는 게 아니라 시야를 좁힌다 —
    # 넓은 문맥에서 생긴 추론(앞에서 언급된 사람을 뒤 발화의 담당자로 잇는 것 등)이
    # 근거 발화만으로 재현되는지 보는 것이 목적이다.
    "EXTRACT_NARROW": (
        "근거 발화와 그 앞뒤 3발화만 근거로 삼는다. "
        "그 범위에서 담당자가 확인되지 않으면 unknown_person 으로 둔다."
    ),
}


def evidence_pool(items: list[TopicItem], utterances: list[Utterance]) -> list[Utterance]:
    """근거로 지정할 수 있는 발화만 골라낸다.

    항목(items)이 자기 근거 발화 id 를 들고 왔으면 **그 발화들로 좁힌다.** L4 의 일은
    L3.5 가 확정한 항목을 배정으로 바꾸는 것이므로, 근거도 그 항목이 선 근거여야
    추적이 이어진다. 좁히지 않으면 주제 안의 아무 발화나 근거로 붙을 수 있고,
    그러면 "확정된 항목"과 "뽑힌 배정"의 연결이 끊긴다.

    비워 보내면(아직 Spring 이 채우지 않는 단계) 주제의 발화 전체를 후보로 둔다 —
    좁힐 근거가 없을 때 전부 버리면 배정을 하나도 못 뽑는다.
    """
    approved = {uid for item in items for uid in item.evidence_utterance_ids}
    if not approved:
        return utterances

    narrowed = [u for u in utterances if u.utterance_id in approved]
    # 항목이 준 id 가 발화 목록에 하나도 없으면 둘이 어긋난 것이다. 전부 버리면
    # 조용히 빈 결과가 되므로, 문맥 전체를 후보로 두고 후처리 검증에 맡긴다.
    return narrowed or utterances


def build_response_schema(participants: list[Participant], evidence_utterances: list[Utterance]) -> dict:
    """참석자와 **근거 가능 발화**를 그대로 enum 으로 박은 응답 스키마.

    회의마다 참석자가 다르므로 스키마는 요청 시점에 만들어진다. 고정 스키마로 두면
    personId 를 검증할 방법이 없어져 4원칙의 '닫힌 목록'이 프롬프트 부탁으로 내려앉는다.
    """
    person_enum = [str(p.person_id) for p in participants if p.person_id is not None]
    person_enum.append(UNKNOWN_PERSON)

    utterance_enum = [str(u.utterance_id) for u in evidence_utterances]

    return {
        "type": "OBJECT",
        "properties": {
            "tuples": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "title": {"type": "STRING"},
                        "assigneeCandidatePersonId": {"type": "STRING", "enum": person_enum},
                        "assigneeSource": {
                            "type": "STRING",
                            "enum": ["EXPLICIT_CALL", "FIRST_PERSON", SOURCE_UNKNOWN],
                        },
                        "dueDate": {"type": "STRING", "nullable": True},
                        "evidenceUtteranceId": {"type": "STRING", "enum": utterance_enum},
                    },
                    "required": [
                        "title",
                        "assigneeCandidatePersonId",
                        "assigneeSource",
                        "evidenceUtteranceId",
                    ],
                    "propertyOrdering": [
                        "title",
                        "assigneeCandidatePersonId",
                        "assigneeSource",
                        "dueDate",
                        "evidenceUtteranceId",
                    ],
                },
            }
        },
        "required": ["tuples"],
    }


async def extract_tuples(request: ExtractTuplesRequest, runner: LayerRunner) -> ExtractTuplesResponse:
    examples = await few_shot.lookup(
        tenant_id=request.tenant_id,
        layer=SPEC.layer,
        query_text=request.query_text,
    )

    pool = evidence_pool(request.items, request.utterances)

    # 근거 후보가 없으면 반환할 수 있는 항목이 없다. 모델을 부르지 않고 끝낸다 —
    # enum 이 빈 배열인 스키마는 만들 수도 없고, 불러도 결과가 전부 버려지므로
    # 토큰만 태운다.
    if not pool:
        return ExtractTuplesResponse(
            tuples=[],
            used_few_shot=examples,
            model=runner.model_name,
            prompt_version=SPEC.prompt_version,
        )

    schema = build_response_schema(request.participants, pool)
    variables = build_prompt_variables(request, examples)

    raw, usage = await runner.run(SPEC, variables=variables, response_schema=schema)

    return ExtractTuplesResponse(
        tuples=parse_tuples(raw, request.participants, pool),
        used_few_shot=examples,
        usage=usage,
        model=runner.model_name,
        prompt_version=SPEC.prompt_version,
    )


def build_prompt_variables(request: ExtractTuplesRequest, examples: list[FewShotExample]) -> dict[str, str]:
    """프롬프트 자리표시자 → 값. 프롬프트 파일과 이 함수가 어긋나면
    `render_prompt` 가 PROMPT_UNRESOLVED 로 즉시 실패한다(조용히 넘어가지 않는다)."""
    eligible_ids = {u.utterance_id for u in evidence_pool(request.items, request.utterances)}
    return {
        "TOPIC": request.topic,
        "MEETING_DATE": _format_meeting_date(request.meeting_date),
        "PARTICIPANTS": _format_participants(request.participants),
        # 문맥은 전부 보여주고 근거 가능 여부만 표시한다. 근거 후보만 보여주면
        # 앞뒤 맥락이 사라져 "그거"·"아까 그건" 같은 지시어를 해석할 수 없다.
        "UTTERANCES": _format_utterances(request.utterances, eligible_ids),
        "ITEMS": _format_items(request.items),
        "FEW_SHOT": _format_few_shot(examples),
        "VIEW_INSTRUCTION": _VIEW_INSTRUCTION[request.view],
    }


def parse_tuples(
    raw: dict, participants: list[Participant], evidence_utterances: list[Utterance]
) -> list[AssignmentTuple]:
    """스키마를 걸었어도 한 번 더 검증한다.

    구조화 출력은 제공자 구현에 의존하고, 그 구현이 흔들리는 순간 조용히 오염된
    담당자·근거가 액션 테이블까지 흘러간다. 여기서 걸러내면 최악이 '항목 누락'이다.

    `evidence_utterances` 는 `evidence_pool()` 이 좁힌 목록이어야 한다 — 스키마에
    박은 enum 과 같은 집합으로 검증해야 둘이 어긋나지 않는다.
    """
    allowed_person_ids = {p.person_id for p in participants if p.person_id is not None}
    allowed_utterance_ids = {u.utterance_id for u in evidence_utterances}

    results: list[AssignmentTuple] = []
    seen: set[tuple[str, int | None, int]] = set()

    for item in raw.get("tuples") or []:
        if not isinstance(item, dict):
            continue

        title = str(item.get("title") or "").strip()
        if not title:
            continue

        evidence_id = _as_int(item.get("evidenceUtteranceId"))
        # 근거 강제 — 목록에 없는 id 는 그 항목을 버린다(§규칙 2).
        if evidence_id is None or evidence_id not in allowed_utterance_ids:
            continue

        assignee = _as_int(item.get("assigneeCandidatePersonId"))
        if assignee is not None and assignee not in allowed_person_ids:
            assignee = None  # 명단 밖 = unknown_person 과 같게 취급

        source = item.get("assigneeSource")
        if source not in ("EXPLICIT_CALL", "FIRST_PERSON"):
            source = None

        key = (title, assignee, evidence_id)
        if key in seen:  # 세그먼트별 호출의 중복 병합
            continue
        seen.add(key)

        results.append(
            AssignmentTuple(
                title=title,
                assignee_candidate_person_id=assignee,
                assignee_source=source,
                due_date=_as_iso_date(item.get("dueDate")),
                evidence_utterance_id=evidence_id,
            )
        )

    return results


def _as_int(value: object) -> int | None:
    """`unknown_person` 같은 탈출구 문자열은 None 이 된다.

    personId 를 문자열 enum 으로 받는 이유: 구조화 출력의 enum 은 문자열만 지원한다.
    정수 enum 을 쓸 수 없어 문자열로 받고 여기서 되돌린다.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _as_iso_date(value: object) -> str | None:
    """형식이 어긋나면 None. 기권 우선 — 틀린 기한은 잘못된 마감으로 보드에 꽂힌다."""
    if not value:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("null", "none", "unknown"):
        return None
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        return None


def _format_participants(participants: list[Participant]) -> str:
    lines = []
    for person in participants:
        pid = UNKNOWN_PERSON if person.person_id is None else str(person.person_id)
        lines.append(f"- personId={pid} · {person.name}")
    return "\n".join(lines) or "- (없음)"


def _format_meeting_date(meeting_date: date | None) -> str:
    if meeting_date is None:
        # 프롬프트 규칙 3과 짝이다 — 기준일이 없으면 상대 표현을 계산하지 말라고 명시한다.
        return "(제공되지 않음 — 상대적 기한 표현은 계산하지 말고 dueDate 를 null 로 둔다)"
    return meeting_date.isoformat()


def _format_utterances(utterances: list[Utterance], eligible_ids: set[int]) -> str:
    lines = []
    for utterance in utterances:
        speaker = "미정" if utterance.speaker_id is None else f"personId={utterance.speaker_id}"
        # 화자 미정을 숨기지 않는다. 1인칭 발화의 화자가 미정이면 담당자도 미정이어야 한다.
        line = f"- id={utterance.utterance_id} · 화자 {speaker} · {utterance.text}"
        if utterance.utterance_id in eligible_ids:
            line += "   ← 근거 지정 가능"
        lines.append(line)
    return "\n".join(lines) or "- (없음)"


def _format_items(items: list[TopicItem]) -> str:
    lines = [f"- [{item.item_type}] {item.content}" for item in items]
    return "\n".join(lines) or "- (없음)"


def _format_few_shot(examples) -> str:
    if not examples:
        return "- (없음 — 축적 전이다. 예시 없이 규칙만으로 판단한다)"
    lines = []
    for example in examples:
        payload = json.dumps(example.payload, ensure_ascii=False)
        lines.append(f'- 발화: "{example.input_text}"\n  확정 결과: {payload}')
    return "\n".join(lines)
