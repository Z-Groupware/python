"""L4 프롬프트 렌더링.

프롬프트에 `{{NEW_VAR}}` 를 추가하고 배선을 잊는 사고를 여기서 잡는다. 그대로 나가면
모델에게 "{{NEW_VAR}}" 라는 문자열을 근거로 답하라고 시키는 셈이 되고, 결과는
그럴듯하게 나와서 사람이 눈치채지 못한다.
"""

import pytest

from app.errors import LayerError
from app.layers.l4 import SPEC, build_prompt_variables
from app.layers.runner import load_prompt, render_prompt
from app.schemas.common import FewShotExample
from app.schemas.l4 import ExtractTuplesRequest

REQUEST = ExtractTuplesRequest.model_validate(
    {
        "tenantId": 7,
        "meetingId": 500,
        "topic": "제품 로드맵",
        "items": [{"itemType": "DECISION", "content": "온보딩 플로우를 3단계로 축소"}],
        "utterances": [
            {"utteranceId": 8812, "speakerId": 7, "startMs": 1284000, "text": "서준님이 정리해주세요"},
            {"utteranceId": 8813, "speakerId": None, "startMs": 1286000, "text": "그럼 그렇게 가시죠"},
        ],
        "participants": [{"personId": 7, "name": "김서준"}, {"personId": None, "name": "명단 외"}],
        "queryText": "서준님이 정리해주세요",
        "view": "EXTRACT",
    }
)


def _render(request=REQUEST, examples=None):
    return render_prompt(load_prompt(SPEC.prompt_file), build_prompt_variables(request, examples or []))


def test_자리표시자가_전부_치환된다():
    rendered = _render()

    assert "{{" not in rendered


def test_참석자와_발화가_프롬프트에_실린다():
    rendered = _render()

    assert "personId=7 · 김서준" in rendered
    assert "personId=unknown_person · 명단 외" in rendered
    assert "id=8812" in rendered
    # 화자 미정을 숨기지 않는다 — 1인칭 발화의 화자가 미정이면 담당자도 미정이어야 한다.
    assert "화자 미정" in rendered


def test_예시가_없으면_없다고_명시한다():
    rendered = _render(examples=[])

    # 빈칸으로 두면 모델이 앞 문맥을 예시로 오독한다.
    assert "(없음" in rendered


def test_예시가_있으면_발화와_확정결과가_함께_들어간다():
    example = FewShotExample(
        input_text="이 부분은 제가 금요일까지 하겠습니다",
        payload={"title": "리팩터링", "assigneeCandidatePersonId": 42},
    )

    rendered = _render(examples=[example])

    assert "이 부분은 제가 금요일까지 하겠습니다" in rendered
    assert "리팩터링" in rendered


def test_NARROW_관점은_시야를_좁히라고_지시한다():
    narrow = REQUEST.model_copy(update={"view": "EXTRACT_NARROW"})

    rendered = _render(narrow)

    assert "앞뒤 3발화" in rendered


def test_치환되지_않은_자리표시자는_즉시_실패한다():
    with pytest.raises(LayerError) as exc:
        render_prompt("주제는 {{TOPIC}} 이고 참석자는 {{PARTICIPANTS}}", {"TOPIC": "x"})

    assert exc.value.code == "PROMPT_UNRESOLVED"
    assert not exc.value.retryable
