"""L5 · 관점 다변화 검증 (AI-07).

다른 계층과 달리 이 계층은 **모델을 두 번, 서로 다른 관점으로 부른다.**

    EXTRACT_NARROW   l4 를 view=EXTRACT_NARROW 로 재실행 — 근거 발화 앞뒤만 보고 다시 뽑기
    VERIFY           "이 tuple 이 맞나?" — 생성이 아니라 검사

두 관점을 **Python 안에서 조합한다.** Spring 이 각각 호출해 결과를 모으면 인스턴스 간
왕복이 두 번이 되고, 더 나쁘게는 조합 규칙("한쪽만 실패하면 안전한 쪽으로")이 Spring 과
Python 두 곳에 생겨 한쪽만 고쳐지는 상태가 만들어진다.

두 호출은 병렬로 돈다. 순차로 돌리면 사용자가 기다리는 시간이 그대로 두 배가 되는데,
두 관점은 서로의 결과를 보지 않으므로(그게 관점 다변화의 전제다) 순서가 의미 없다.
"""

from __future__ import annotations

import asyncio
import json

from app.errors import LayerError, LayerErrorKind
from app.layers import formatting as fmt
from app.layers import l4
from app.layers.runner import LayerRunner, LayerSpec
from app.schemas.common import Participant, Usage, Utterance
from app.schemas.l4 import AssignmentTuple, ExtractTuplesRequest
from app.schemas.l5 import (
    COMPARED_FIELDS,
    REASON_MAX,
    VerifyRequest,
    VerifyResponse,
    ViewResult,
)

SPEC = LayerSpec(
    layer="L5",
    prompt_file="l5_verify.v1.txt",
    prompt_version="v1",
    dry_run_payload={"verdict": "ACCEPT", "reason": "(DRY_RUN)"},
)

VERDICTS = ["ACCEPT", "REJECT"]

# 한 관점이 재현하지 못했을 때 붙는 불일치 표시. 필드 하나가 갈린 것과 "아예 안 나왔다"는
# 다른 신호라 따로 둔다 — 후자가 잦으면 좁은 시야에서 근거가 부족하다는 뜻이다.
NOT_REPRODUCED = "notReproduced"


def build_response_schema() -> dict:
    """VERIFY 관점의 응답. 참석자·발화 enum 이 없는 유일한 계층이다 —
    새 값을 뽑는 것이 아니라 주어진 것을 판정하기만 하므로 닫을 목록이 없다."""
    return {
        "type": "OBJECT",
        "properties": {
            "verdict": {"type": "STRING", "enum": VERDICTS},
            "reason": {"type": "STRING"},
        },
        "required": ["verdict", "reason"],
        "propertyOrdering": ["verdict", "reason"],
    }


async def verify(request: VerifyRequest, runner: LayerRunner) -> VerifyResponse:
    narrow, verify_result = await asyncio.gather(
        _run_narrow(request, runner),
        _run_verify(request, runner),
        return_exceptions=False,
    )

    # 둘 다 실패했으면 검증이 아예 수행되지 않은 것이다. 그걸 agree=false 로 돌려주면
    # "관점이 갈렸다"로 기록돼 검증이 돈 것처럼 보인다 — 계층 실패로 던져야
    # Spring 이 analysis_layer 를 FAILED 로 남기고 재시도한다.
    if narrow.error and verify_result.error:
        raise _both_views_failed(narrow, verify_result)

    disagreements = compare(request.tuple, narrow)
    # 한 관점만 실패해도 안전한 쪽으로 — agree=false 로 검토에 보낸다(명세).
    # 실패한 관점을 "동의"로 세면 검증이 반쪽만 돌았는데 자동 확정으로 나간다.
    agree = not disagreements and not narrow.error and verify_result.verdict == "ACCEPT"

    return VerifyResponse(
        agree=agree,
        disagreement_fields=disagreements,
        results=[narrow, verify_result],
        usage=_sum_usage(narrow, verify_result),
        model=runner.model_name,
        prompt_version=SPEC.prompt_version,
    )


def compare(baseline: AssignmentTuple, narrow: ViewResult) -> list[str]:
    """기준 tuple 과 좁은 시야의 재추출을 필드 단위로 맞춰본다.

    좁은 시야에서 **재현되지 않은 것 자체가 신호다.** 넓은 문맥에서만 성립하는 배정은
    앞에서 언급된 사람을 뒤 발화의 담당자로 이어 붙인 추론일 때가 많고, 그게 담당자
    오배정(`WRONG_ASSIGNEE`)의 주된 경로다.

    관점이 실패한 경우는 여기서 불일치로 세지 않는다 — 실패는 `error` 로 이미 드러나고,
    "갈렸다"와 "못 물어봤다"를 섞으면 disagreementFields 가 원인 조사에 쓸모없어진다.
    """
    if narrow.error:
        return []
    if narrow.tuple is None:
        return [NOT_REPRODUCED]

    differences = []
    if narrow.tuple.title != baseline.title:
        differences.append("title")
    if narrow.tuple.assignee_candidate_person_id != baseline.assignee_candidate_person_id:
        differences.append("assigneeCandidatePersonId")
    if narrow.tuple.due_date != baseline.due_date:
        differences.append("dueDate")

    # COMPARED_FIELDS 는 명세가 돌려주기로 한 필드명 목록이다. 위 비교와 어긋나면
    # 응답에 없는 필드명이 나가므로 순서를 그것에 맞춘다.
    return [field for field in COMPARED_FIELDS if field in differences]


async def _run_narrow(request: VerifyRequest, runner: LayerRunner) -> ViewResult:
    """L4 를 좁은 시야로 재실행한다. 같은 프롬프트를 두 번 돌리는 것이 아니라
    `VIEW_INSTRUCTION` 이 바뀐다 — 그래서 오류가 서로 상관되지 않는다."""
    try:
        narrow_request = ExtractTuplesRequest(
            tenant_id=request.tenant_id,
            meeting_id=request.meeting_id,
            topic=request.topic,
            items=request.items,
            utterances=_evidence_window(request),
            participants=request.participants,
            query_text=request.query_text,
            meeting_date=request.meeting_date,
            view="EXTRACT_NARROW",
        )
        response = await l4.extract_tuples(narrow_request, runner)
    except LayerError as exc:  # 한쪽 실패는 전체 실패가 아니다 — 안전한 쪽으로 접는다
        return ViewResult(view="EXTRACT_NARROW", error=_describe(exc))

    return ViewResult(
        view="EXTRACT_NARROW",
        tuple=_match(response.tuples, request.tuple),
        usage=response.usage,
    )


async def _run_verify(request: VerifyRequest, runner: LayerRunner) -> ViewResult:
    try:
        raw, usage = await runner.run(
            SPEC,
            variables=_verify_variables(request),
            response_schema=build_response_schema(),
        )
    except LayerError as exc:
        return ViewResult(view="VERIFY", error=_describe(exc))

    verdict = raw.get("verdict")
    # 판정값이 깨져 돌아오면 ACCEPT 로 넘기지 않는다. 애매한 것은 검토로 보낸다.
    if verdict not in VERDICTS:
        verdict = "REJECT"

    return ViewResult(
        view="VERIFY",
        verdict=verdict,
        reason=fmt.clip(fmt.as_text(raw.get("reason")), REASON_MAX) or "(모델이 판정 근거를 반환하지 않음)",
        usage=usage,
    )


def _evidence_window(request: VerifyRequest) -> list[Utterance]:
    """근거 발화와 그 앞뒤 3발화만 남긴다.

    프롬프트에 "앞뒤 3발화만 보라"고 적어도 나머지 발화가 함께 들어가 있으면 모델은
    그것을 읽는다. 관점을 좁히는 유일하게 확실한 방법은 **넘기지 않는 것**이다.
    """
    window = 3
    ids = [u.utterance_id for u in request.utterances]
    if request.tuple.evidence_utterance_id not in ids:
        # 근거 발화가 문맥에 없다 — 좁힐 기준이 없으므로 전부 넘긴다. 이 경우
        # 재현이 안 되면 compare 가 notReproduced 로 잡는다.
        return request.utterances

    center = ids.index(request.tuple.evidence_utterance_id)
    return request.utterances[max(0, center - window) : center + window + 1]


def _match(candidates: list[AssignmentTuple], baseline: AssignmentTuple) -> AssignmentTuple | None:
    """같은 근거 발화에서 나온 tuple 을 짝으로 본다.

    title 로 맞추지 않는 이유: 좁은 시야에서 표현이 조금 달라지는 것은 흔하고, 그걸
    '재현 실패'로 세면 거의 모든 항목이 검토로 넘어가 게이트가 무의미해진다.
    근거 발화는 두 관점이 같아야 하는 유일한 값이다.
    """
    for candidate in candidates:
        if candidate.evidence_utterance_id == baseline.evidence_utterance_id:
            return candidate
    return None


def _verify_variables(request: VerifyRequest) -> dict[str, str]:
    return {
        "TOPIC": request.topic,
        "MEETING_DATE": fmt.format_meeting_date(request.meeting_date),
        "PARTICIPANTS": fmt.format_participants(request.participants),
        "TARGET_TUPLE": _format_target(request.tuple, request.participants),
        "UTTERANCES": fmt.format_utterances(request.utterances),
    }


def _format_target(target: AssignmentTuple, participants: list[Participant]) -> str:
    names = {p.person_id: p.name for p in participants if p.person_id is not None}
    assignee = (
        f"personId={target.assignee_candidate_person_id}"
        f" ({names.get(target.assignee_candidate_person_id, '이름 미상')})"
        if target.assignee_candidate_person_id is not None
        else "미정"
    )
    return (
        f"- 할 일   : {target.title}\n"
        f"- 담당자  : {assignee}\n"
        f"- 판정근거: {target.assignee_source or 'UNKNOWN'}\n"
        f"- 기한    : {target.due_date or '없음'}\n"
        f"- 근거 발화 id: {target.evidence_utterance_id}"
    )


def _sum_usage(*results: ViewResult) -> Usage:
    """두 관점의 토큰을 합산한다. 한쪽만 세면 L5 가 실제보다 싸 보이고,
    특화 모델 전환의 손익분기점(QLTY-03)이 그만큼 틀어진다."""
    return Usage(
        tokens_in=sum(r.usage.tokens_in for r in results),
        tokens_out=sum(r.usage.tokens_out for r in results),
    )


def _describe(exc: LayerError) -> str:
    """실패 사유는 **오류 코드까지만** 싣는다.

    원문 메시지를 그대로 실으면 제공자 응답 본문이 우리 API 응답에 섞이고, 거기 무엇이
    들어 있을지는 보증할 수 없다(`ProviderAvailability` 가 사유에 상태코드만 담는 것과 같은 이유).
    본문이 필요한 진단은 서버 로그를 본다.
    """
    return exc.code


def _both_views_failed(*results: ViewResult) -> LayerError:
    detail = json.dumps([r.error for r in results], ensure_ascii=False)
    return LayerError(
        LayerErrorKind.TRANSIENT,
        "ALL_VIEWS_FAILED",
        f"두 관점이 모두 실패했습니다 — 검증이 수행되지 않았습니다: {detail}",
    )
