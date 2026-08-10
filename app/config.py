"""설정 — 환경변수 하나로만 주입받는다.

이 서버는 업무 DB에 접속하지 않는다. 그래서 DB 설정이 여기 없는 것이 정상이다.
상태를 갖지 않는 계산 서버로 두면 계층을 특화 모델로 갈아끼우거나 인스턴스를
늘려도 계약이 깨지지 않는다. 모든 입력은 요청 본문으로 받고 출력은 응답으로 준다.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Spring EC2 에서만 들어오는 호출임을 확인하는 공유 시크릿.
    # 보안그룹으로 인바운드를 좁혀도 이건 별도로 둔다 — 같은 VPC 안의 다른
    # 인스턴스가 뚫렸을 때 방어선이 하나도 없는 상태가 되지 않게.
    internal_token: str = ""

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash"

    # 임베딩 모델(AI-08·09). 생성 모델과 **따로 둔다** — 계층 모델을 특화 모델로 갈아끼울 때
    # 임베딩까지 함께 바뀌면 기존 컬렉션 전체를 못 쓰게 된다. 두 축은 독립적으로 움직인다.
    gemini_embed_model: str = "gemini-embedding-001"

    # 벡터 차원. **컬렉션을 만들 때 박히는 값이다** — 모델 기본값에 맡기면 모델을 바꾸는
    # 순간 기존 컬렉션에 넣지 못한다. 여기서 못박고 output_dimensionality 로 강제한다.
    embed_dim: int = 768

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    # 컬렉션 이름에 차원·모델을 넣지 않는다. 이름이 스키마를 설명하기 시작하면 바꿀 때마다
    # 코드 상수까지 함께 고쳐야 한다. 재색인이 필요하면 이 값을 새 이름으로 준다.
    qdrant_collection: str = "meeting_tuple_vector"

    # few-shot 조회 자체를 끈다. Qdrant 를 아직 안 띄운 환경에서 계층만 돌려볼 때 쓴다 —
    # 켜 둔 채로 두면 계층 호출마다 조회 실패 로그가 쌓인다(동작은 한다).
    few_shot_enabled: bool = True

    # 계층 호출 실패 정책 (명세 「레이어 호출 실패 정책」)
    # 일시적 실패는 지수 백오프 3회, 지터 ±20%.
    retry_delays_sec: tuple[float, ...] = (2.0, 8.0, 30.0)
    retry_jitter_ratio: float = 0.2

    # ── AI-01 VAD ────────────────────────────────────────────────────────────
    # silero-vad ONNX 모델 파일. **이미지에 함께 넣는다** — 런타임에 받아오면 모델이 바뀔 때
    # 절단점이 배포와 무관하게 조용히 달라지고, "어제와 오늘의 블록 경계가 다른" 이유를
    # 아무도 못 찾는다. 없으면 VAD_MODEL_MISSING 으로 명확히 실패한다.
    vad_model_path: str = "/app/models/silero_vad.onnx"

    # 동시에 도는 VAD 추론 수. 기본 실행기(min(32, cpu+4))에 맡기지 않는 이유는 t3.medium
    # (2 vCPU)에 Qdrant 까지 같이 떠 있어, 코어보다 많은 추론이 서로 코어를 뺏으면 헬스체크까지
    # 늦어져 컨테이너가 재시작되기 때문이다. 인스턴스를 키우면 이 값을 올린다.
    vad_max_workers: int = 1

    # S3 리전. 자격증명은 여기 두지 않는다 — AI EC2 의 IAM 롤을 boto3 가 집는다.
    # 키를 .env 에 두면 그 파일이 유출 표면이 되고 자동 회전도 안 된다.
    s3_region: str = ""

    # 1이면 Gemini 를 부르지 않고 고정 스텁을 돌려준다. 라우팅·스키마 확인용.
    dry_run: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
