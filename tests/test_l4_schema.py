"""L4 응답 스키마 강제와 후처리 검증.

3주 일정에서 테스트를 두 곳에만 쓴다면 하나는 여기다. 여기가 뚫리면 명단 밖 담당자와
없는 근거 발화가 액션 테이블까지 흘러가고, 그건 화면에서 사람이 알아볼 수 없다.
"""

from app.layers.l4 import build_response_schema, parse_tuples
from app.schemas.common import Participant, Utterance
from app.schemas.l4 import UNKNOWN_PERSON

PARTICIPANTS = [
    Participant(person_id=7, name="김서준"),
    Participant(person_id=42, name="이태연"),
    Participant(person_id=None, name="명단 외"),
]
UTTERANCES = [
    Utterance(
        utterance_id=8812, speaker_id=7, start_ms=1284000, text="A/B 테스트 도구는 서준님이 정리해주세요"
    ),
    Utterance(utterance_id=8813, speaker_id=None, start_ms=1286000, text="그럼 그렇게 가시죠"),
]


class TestResponseSchema:
    def test_담당자는_참석자와_탈출구만_허용한다(self):
        schema = build_response_schema(PARTICIPANTS, UTTERANCES)
        enum = schema["properties"]["tuples"]["items"]["properties"]["assigneeCandidatePersonId"]["enum"]

        assert enum == ["7", "42", UNKNOWN_PERSON]

    def test_근거는_전달된_발화_id만_허용한다(self):
        schema = build_response_schema(PARTICIPANTS, UTTERANCES)
        enum = schema["properties"]["tuples"]["items"]["properties"]["evidenceUtteranceId"]["enum"]

        assert enum == ["8812", "8813"]

    def test_근거와_담당자와_출처는_필수다(self):
        schema = build_response_schema(PARTICIPANTS, UTTERANCES)
        required = schema["properties"]["tuples"]["items"]["required"]

        assert "evidenceUtteranceId" in required
        assert "assigneeCandidatePersonId" in required
        assert "assigneeSource" in required
        # 기한은 없을 수 있다 — 기권 우선.
        assert "dueDate" not in required


class TestParseTuples:
    def test_정상_항목을_통과시킨다(self):
        raw = {
            "tuples": [
                {
                    "title": "A/B 테스트 도구 비교 정리",
                    "assigneeCandidatePersonId": "7",
                    "assigneeSource": "EXPLICIT_CALL",
                    "dueDate": "2026-08-08",
                    "evidenceUtteranceId": "8812",
                }
            ]
        }

        result = parse_tuples(raw, PARTICIPANTS, UTTERANCES)

        assert len(result) == 1
        assert result[0].assignee_candidate_person_id == 7
        assert result[0].assignee_source == "EXPLICIT_CALL"
        assert result[0].due_date == "2026-08-08"
        assert result[0].evidence_utterance_id == 8812

    def test_없는_근거_발화는_항목을_버린다(self):
        raw = {
            "tuples": [
                {
                    "title": "존재하지 않는 근거",
                    "assigneeCandidatePersonId": "7",
                    "assigneeSource": "EXPLICIT_CALL",
                    "evidenceUtteranceId": "99999",
                }
            ]
        }

        assert parse_tuples(raw, PARTICIPANTS, UTTERANCES) == []

    def test_근거가_없으면_항목을_버린다(self):
        raw = {"tuples": [{"title": "근거 없음", "assigneeCandidatePersonId": "7"}]}

        assert parse_tuples(raw, PARTICIPANTS, UTTERANCES) == []

    def test_unknown_person은_None이_된다(self):
        raw = {
            "tuples": [
                {
                    "title": "외부 참석자 몫",
                    "assigneeCandidatePersonId": UNKNOWN_PERSON,
                    "assigneeSource": "EXPLICIT_CALL",
                    "evidenceUtteranceId": "8812",
                }
            ]
        }

        result = parse_tuples(raw, PARTICIPANTS, UTTERANCES)

        # 항목 자체는 살린다 — 담당자만 미정이고, 게이트가 needs_review 로 보낸다.
        assert len(result) == 1
        assert result[0].assignee_candidate_person_id is None

    def test_명단_밖_담당자는_None이_된다(self):
        raw = {
            "tuples": [
                {
                    "title": "명단에 없는 사람",
                    "assigneeCandidatePersonId": "999",
                    "assigneeSource": "EXPLICIT_CALL",
                    "evidenceUtteranceId": "8812",
                }
            ]
        }

        result = parse_tuples(raw, PARTICIPANTS, UTTERANCES)

        assert result[0].assignee_candidate_person_id is None

    def test_출처_UNKNOWN은_None이_된다(self):
        raw = {
            "tuples": [
                {
                    "title": "출처 미정",
                    "assigneeCandidatePersonId": "7",
                    "assigneeSource": "UNKNOWN",
                    "evidenceUtteranceId": "8812",
                }
            ]
        }

        assert parse_tuples(raw, PARTICIPANTS, UTTERANCES)[0].assignee_source is None

    def test_형식이_깨진_기한은_비운다(self):
        raw = {
            "tuples": [
                {
                    "title": "기한 깨짐",
                    "assigneeCandidatePersonId": "7",
                    "assigneeSource": "FIRST_PERSON",
                    "dueDate": "다음 주 금요일",
                    "evidenceUtteranceId": "8812",
                }
            ]
        }

        # 틀린 날짜를 채우면 잘못된 마감으로 보드에 꽂힌다. 비우는 게 낫다.
        assert parse_tuples(raw, PARTICIPANTS, UTTERANCES)[0].due_date is None

    def test_중복_항목을_병합한다(self):
        item = {
            "title": "같은 일",
            "assigneeCandidatePersonId": "7",
            "assigneeSource": "EXPLICIT_CALL",
            "evidenceUtteranceId": "8812",
        }

        result = parse_tuples({"tuples": [item, dict(item)]}, PARTICIPANTS, UTTERANCES)

        assert len(result) == 1
