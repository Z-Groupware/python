"""few-shot 조회 — AI-09 `/internal/similar` 의 내부 사용.

Spring 이 따로 조회하지 않고 여기서 부른다. 인스턴스가 분리돼 있어 벡터·예시를
네트워크로 왕복시킬 이유가 없기 때문이다. 대신 무엇을 썼는지 `usedFewShot` 으로
돌려줘야 review_log.input_context 가 온전해진다 — 그게 없으면 나중에 "이 라벨은
어떤 예시를 보고 나온 판단인지"를 재현할 수 없다.

Qdrant 연결은 일정상 뒤에 붙는다(AI-08·09). 지금은 빈 목록을 돌려주되 호출 경로와
응답 필드는 처음부터 뚫어 둔다 — 나중에 필드를 추가하면 Spring 쪽 DTO 도 같이 고쳐야 한다.
"""

from __future__ import annotations

from app.schemas.common import FewShotExample

# 세 필터는 전부 필수다. tenant 가 빠지면 다른 회사 발화가 프롬프트에 들어간다 —
# 정확도 문제가 아니라 유출이다.
PROVENANCE_HUMAN_VERIFIED = "HUMAN_VERIFIED"


async def lookup(
    *,
    tenant_id: int,
    layer: str,
    query_text: str | None,
    dept_id: int | None = None,
    top_k: int = 5,
    provenance: str = PROVENANCE_HUMAN_VERIFIED,
) -> list[FewShotExample]:
    if not query_text:
        return []

    # TODO(AI-08·09): Qdrant 조회로 교체. 임베딩 대상은 근거 발화 텍스트이고
    # 확정 tuple 은 payload 로 붙는다 — tuple 을 임베딩하면 쿼리(발화)와 다른
    # 공간이 되어 유사도가 망가진다.
    return []
