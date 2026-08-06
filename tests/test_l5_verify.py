"""L5 관점 다변화 조합 검증.

앙상블의 가치는 정답 선택이 아니라 **불확실성 탐지**다. 그래서 여기서 지켜야 하는 것은
"틀린 쪽을 골라내는가"가 아니라 **"갈렸을 때 반드시 검토로 보내는가"** 다.
안전한 쪽으로 접히지 않는 경로가 하나라도 있으면 그게 자동 확정으로 새는 구멍이 된다.
"""

import pytest

from app.errors import LayerError, LayerErrorKind
from app.layers.l5 import (
    NOT_REPRODUCED,
    _evidence_window,
    _match,
    blocking,
    build_response_schema,
    compare,
    verify,
)
from app.schemas.common import Participant, Usage, Utterance
from app.schemas.l4 import AssignmentTuple, ExtractTuplesResponse
from app.schemas.l5 import VerifyRequest, ViewResult

PARTICIPANTS = [Participant(person_id=7, name="김서준"), Participant(person_id=42, name="이태연")]
UTTERANCES = [Utterance(utterance_id=300 + i, speaker_id=7, text=f"발화 {i}") for i in range(10)]

BASELINE = AssignmentTuple(
    title="A/B 테스트 도구 비교 정리",
    assignee_candidate_person_id=7,
    assignee_source="EXPLICIT_CALL",
    due_date="2026-08-08",
    evidence_utterance_id=305,
)


def request() -> VerifyRequest:
    return VerifyRequest(
        tenant_id=7,
        meeting_id=500,
        topic="제품 로드맵",
        tuple=BASELINE,
        utterances=UTTERANCES,
        participants=PARTICIPANTS,
    )


class FakeRunner:
    """VERIFY 관점만 대신한다. NARROW 는 l4 를 통해 도는 경로라 따로 가로챈다."""

    model_name = "fake-model"

    def __init__(self, payload=None, error=None):
        self._payload = payload or {"verdict": "ACCEPT", "reason": "근거 발화에서 확인됨"}
        self._error = error

    async def run(self, spec, *, variables, response_schema):
        if self._error:
            raise self._error
        return self._payload, Usage(tokens_in=10, tokens_out=5)


@pytest.fixture
def patch_narrow(monkeypatch):
    """l4.extract_tuples 를 갈아끼운다 — L5 의 일은 조합이지 추출이 아니다."""

    def _patch(tuples=None, error=None):
        async def fake(request_, runner):
            if error:
                raise error
            return ExtractTuplesResponse(
                tuples=tuples or [],
                usage=Usage(tokens_in=100, tokens_out=20),
                model="fake-model",
                prompt_version="v1",
            )

        monkeypatch.setattr("app.layers.l5.l4.extract_tuples", fake)

    return _patch


class TestResponseSchema:
    def test_닫을_목록이_없다(self):
        # 새 값을 뽑는 것이 아니라 주어진 것을 판정한다 — 참석자·발화 enum 이 없는 유일한 계층.
        schema = build_response_schema()

        assert set(schema["properties"]) == {"verdict", "reason"}
        assert schema["properties"]["verdict"]["enum"] == ["ACCEPT", "REJECT"]


class TestCompare:
    def test_같으면_불일치가_없다(self):
        narrow = ViewResult(view="EXTRACT_NARROW", tuple=BASELINE.model_copy())

        assert compare(BASELINE, narrow) == []

    def test_담당자가_갈리면_잡아낸다(self):
        narrow = ViewResult(
            view="EXTRACT_NARROW",
            tuple=BASELINE.model_copy(update={"assignee_candidate_person_id": 42}),
        )

        assert compare(BASELINE, narrow) == ["assigneeCandidatePersonId"]

    def test_좁은_시야에서_재현되지_않은_것도_신호다(self):
        # 넓은 문맥에서만 성립하는 배정은 앞에서 언급된 사람을 뒤 발화의 담당자로
        # 이어 붙인 추론일 때가 많다 — 담당자 오배정의 주된 경로다.
        narrow = ViewResult(view="EXTRACT_NARROW", tuple=None)

        assert compare(BASELINE, narrow) == [NOT_REPRODUCED]

    def test_관점이_실패한_것은_불일치로_세지_않는다(self):
        # "갈렸다"와 "못 물어봤다"를 섞으면 disagreementFields 가 원인 조사에 쓸모없어진다.
        narrow = ViewResult(view="EXTRACT_NARROW", error="RATE_LIMITED")

        assert compare(BASELINE, narrow) == []

    def test_명세가_정한_필드명_순서로_돌려준다(self):
        narrow = ViewResult(
            view="EXTRACT_NARROW",
            tuple=BASELINE.model_copy(update={"title": "다름", "due_date": None}),
        )

        assert compare(BASELINE, narrow) == ["title", "dueDate"]

    def test_제목이_갈린_것도_그대로_보고한다(self):
        # 검토로 보내지 않을 뿐 값을 버리지는 않는다 — 제목이 자주 갈리는지는
        # 프롬프트를 조일 근거이고, 신호까지 없애면 볼 방법이 없다.
        narrow = ViewResult(
            view="EXTRACT_NARROW",
            tuple=BASELINE.model_copy(update={"title": "A/B 테스트 도구 비교 정리 및 공유"}),
        )

        assert compare(BASELINE, narrow) == ["title"]


class TestBlocking:
    """갈린 것 중 **사람을 부를 것**만 남기는가."""

    def test_제목만_갈린_것은_검토_사유가_아니다(self):
        # 좁은 시야에서 같은 일을 조금 다르게 부르는 것은 흔하다. 그걸 검토로 보내면
        # 목록이 부풀어 진짜 오배정이 표현 차이들 사이에 묻힌다.
        assert blocking(["title"]) == []

    def test_담당자와_기한은_검토_사유다(self):
        # 사람이 고칠 것은 이 둘이다. 틀린 담당자는 남의 일을 받게 하고,
        # 틀린 마감은 그대로 보드에 꽂힌다.
        assert blocking(["assigneeCandidatePersonId"]) == ["assigneeCandidatePersonId"]
        assert blocking(["dueDate"]) == ["dueDate"]

    def test_재현_실패는_필드가_아니어도_막는다(self):
        # 좁은 시야에서 아예 안 나온 것은 필드 하나가 다른 것과 성질이 다르다 —
        # 넓은 문맥에서만 성립하는 배정이 담당자 오배정의 주된 경로다.
        assert blocking([NOT_REPRODUCED]) == [NOT_REPRODUCED]

    def test_제목과_담당자가_함께_갈리면_막는다(self):
        assert blocking(["title", "assigneeCandidatePersonId"]) == ["assigneeCandidatePersonId"]


class TestEvidenceWindow:
    def test_근거_발화_앞뒤_3발화만_넘긴다(self):
        # "앞뒤 3발화만 보라"고 적어도 나머지가 함께 들어가 있으면 모델은 그것을 읽는다.
        # 관점을 좁히는 유일하게 확실한 방법은 넘기지 않는 것이다.
        window = _evidence_window(request())

        assert [u.utterance_id for u in window] == [302, 303, 304, 305, 306, 307, 308]

    def test_근거_발화가_문맥에_없으면_전부_넘긴다(self):
        broken = request().model_copy(
            update={"tuple": BASELINE.model_copy(update={"evidence_utterance_id": 9999})}
        )

        assert len(_evidence_window(broken)) == len(UTTERANCES)


class TestMatch:
    def test_근거_발화로_짝을_맞춘다(self):
        # title 로 맞추면 좁은 시야의 사소한 표현 차이가 전부 '재현 실패'가 되어
        # 거의 모든 항목이 검토로 넘어간다 — 게이트가 무의미해진다.
        candidate = BASELINE.model_copy(update={"title": "표현이 조금 다른 제목"})

        assert _match([candidate], BASELINE) is candidate

    def test_근거가_다르면_짝이_아니다(self):
        candidate = BASELINE.model_copy(update={"evidence_utterance_id": 301})

        assert _match([candidate], BASELINE) is None


@pytest.mark.asyncio
class TestVerify:
    async def test_두_관점이_모두_동의하면_agree다(self, patch_narrow):
        patch_narrow(tuples=[BASELINE.model_copy()])

        response = await verify(request(), FakeRunner())

        assert response.agree is True
        assert response.disagreement_fields == []

    async def test_제목만_갈리면_검토로_보내지_않는다(self, patch_narrow):
        # 실호출에서 바로 나온 모양이다 — "제품 로드맵 초안 작성" vs
        # "…작성 및 공유". 담당자·기한·근거가 전부 같고 VERIFY 도 ACCEPT 였는데
        # 표현 차이 하나로 검토 대상이 됐다(python#11).
        patch_narrow(tuples=[BASELINE.model_copy(update={"title": "A/B 테스트 도구 비교 정리 및 공유"})])

        response = await verify(request(), FakeRunner())

        assert response.agree is True
        # 값은 버리지 않는다. agree=true 인 채로 갈린 필드가 실려 나간다.
        assert response.disagreement_fields == ["title"]

    async def test_제목이_갈려도_담당자가_갈리면_검토로_보낸다(self, patch_narrow):
        patch_narrow(
            tuples=[
                BASELINE.model_copy(update={"title": "다른 제목", "assignee_candidate_person_id": 42})
            ]
        )

        response = await verify(request(), FakeRunner())

        assert response.agree is False
        assert response.disagreement_fields == ["title", "assigneeCandidatePersonId"]

    async def test_불일치는_다수결로_덮지_않는다(self, patch_narrow):
        patch_narrow(tuples=[BASELINE.model_copy(update={"assignee_candidate_person_id": 42})])

        response = await verify(request(), FakeRunner())

        assert response.agree is False
        assert response.disagreement_fields == ["assigneeCandidatePersonId"]

    async def test_VERIFY가_REJECT면_일치해도_검토로_보낸다(self, patch_narrow):
        patch_narrow(tuples=[BASELINE.model_copy()])
        runner = FakeRunner({"verdict": "REJECT", "reason": "근거 발화에 담당자 지목이 없음"})

        response = await verify(request(), runner)

        assert response.agree is False

    async def test_판정값이_깨지면_ACCEPT로_넘기지_않는다(self, patch_narrow):
        patch_narrow(tuples=[BASELINE.model_copy()])

        response = await verify(request(), FakeRunner({"verdict": "MAYBE", "reason": "음"}))

        assert response.agree is False

    async def test_한_관점만_실패하면_안전한_쪽으로_접는다(self, patch_narrow):
        # 실패한 관점을 '동의'로 세면 검증이 반쪽만 돌았는데 자동 확정으로 나간다.
        patch_narrow(error=LayerError(LayerErrorKind.RATE_LIMIT, "RATE_LIMITED", "429"))

        response = await verify(request(), FakeRunner())

        assert response.agree is False
        narrow = next(r for r in response.results if r.view == "EXTRACT_NARROW")
        assert narrow.error == "RATE_LIMITED"

    async def test_두_관점이_모두_실패하면_계층_실패로_던진다(self, patch_narrow):
        # agree=false 로 돌려주면 "관점이 갈렸다"로 기록돼 검증이 돈 것처럼 보인다.
        patch_narrow(error=LayerError(LayerErrorKind.TRANSIENT, "PROVIDER_UNAVAILABLE", "down"))
        runner = FakeRunner(error=LayerError(LayerErrorKind.TRANSIENT, "PROVIDER_UNAVAILABLE", "down"))

        with pytest.raises(LayerError) as caught:
            await verify(request(), runner)

        assert caught.value.code == "ALL_VIEWS_FAILED"
        assert caught.value.retryable is True

    async def test_실패_사유에_제공자_응답_본문을_싣지_않는다(self, patch_narrow):
        patch_narrow(
            error=LayerError(LayerErrorKind.RATE_LIMIT, "RATE_LIMITED", "prepayment credits are depleted")
        )

        response = await verify(request(), FakeRunner())

        narrow = next(r for r in response.results if r.view == "EXTRACT_NARROW")
        assert narrow.error == "RATE_LIMITED"
        assert "prepayment" not in (narrow.error or "")

    async def test_두_관점의_토큰을_합산한다(self, patch_narrow):
        # 한쪽만 세면 L5 가 실제보다 싸 보이고 특화 모델 전환의 손익분기점이 틀어진다.
        patch_narrow(tuples=[BASELINE.model_copy()])

        response = await verify(request(), FakeRunner())

        assert response.usage.tokens_in == 110  # NARROW 100 + VERIFY 10
        assert response.usage.tokens_out == 25

    async def test_두_관점의_결과를_모두_돌려준다(self, patch_narrow):
        patch_narrow(tuples=[BASELINE.model_copy()])

        response = await verify(request(), FakeRunner())

        assert [r.view for r in response.results] == ["EXTRACT_NARROW", "VERIFY"]
