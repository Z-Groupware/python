# 잇다(Z) · AI 계층 서버

회의 분석 파이프라인의 **AI EC2** 쪽. VAD 절단점, Gemini 계층(L1.5~L5), 벡터(Qdrant)를 담당한다.
Spring(BACKEND 리포)이 이 서버를 부르는 **단방향**이다.

담당 이태연 · 도메인 A(캡처 파이프라인) 내부 API 10개

---

## 이 서버가 하지 않는 것

| 하지 않는다 | 어디서 하나 | 왜 |
|---|---|---|
| 업무 DB 접속 | Spring | 상태 없는 계산 서버로 두면 계층을 특화 모델로 갈아끼워도 계약이 안 깨진다 |
| STT 호출 | Spring | AWS Transcribe IAM 롤이 Spring EC2 에 있다 |
| ffmpeg 조립 | Spring | 오디오 원본을 다루는 쪽이 한 곳이어야 한다 |
| 화자 귀속(L1) · 규칙 검사(L6) · 자동확정 게이트 | Spring | 코드 판정이다. LLM 을 쓰지 않는다 |
| 작업 큐 소비 | Spring (SQS) | 이 서버는 HTTP 동기 호출만 받는다 |

오디오는 항상 **S3 키**로 주고받는다 — 두 인스턴스 사이에 공유 볼륨이 없다.

---

## 엔드포인트

전부 `X-Internal-Token` 헤더 필수. 미구현 계층은 **501** 로 거절한다(200 + 빈 결과로 두면
Spring 이 "계층 정상 완료, 산출물 없음"으로 기록해 미구현이 품질 문제로 위장된다).

| ID | 경로 | 상태 |
|---|---|---|
| AI-01 | `POST /internal/vad/cutpoint` | 예정 8/7 |
| **AI-02** | `POST /internal/layers/l1-5/resolve-reference` | **구현됨** |
| **AI-03** | `POST /internal/layers/l2/segment-topics` | **구현됨** |
| **AI-04** | `POST /internal/layers/l3/summarize-topic` | **구현됨** |
| AI-05 | `POST /internal/layers/l3-5/gate` | 예정 8/6 |
| **AI-06** | `POST /internal/layers/l4/extract-tuples` | **구현됨** |
| AI-07 | `POST /internal/layers/l5/verify` | 예정 8/6 |
| AI-08 | `POST /internal/vector/upsert` | 예정 8/9 |
| AI-09 | `POST /internal/similar` | 예정 8/9 |
| **AI-10** | `GET /internal/health` | **구현됨** |

AI-10 이 돌려주는 `implemented` 목록과 실제 라우팅이 어긋나지 않는지 테스트가 검증한다
(`test_internal_auth.py`). 계층을 붙이고 목록을 잊으면 워커가 미구현 계층을 부른다.

`GET /health` 는 무인증 — ALB·컨테이너 liveness 용이다. 계층 수용 가능 여부는 AI-10 이 답한다.

---

## 계층 하나 추가하는 법

L4(`app/layers/l4.py`)가 나머지 계층의 원본 틀이다. 복제해서 **세 곳만** 바꾼다.

1. **`SPEC`** — 프롬프트 파일명과 `prompt_version`
2. **`build_response_schema()`** — 참석자·발화를 enum 으로 박은 응답 스키마
3. **후처리** — 파싱된 dict → 계층 DTO

호출·재시도·토큰 집계는 `app/layers/runner.py` 가 공통으로 갖는다. 계층마다 따로 쓰면
같은 버그를 아홉 번 고치게 된다.

### 정확도 4원칙은 프롬프트가 아니라 스키마로 걸린다

| 원칙 | 어떻게 강제되나 |
|---|---|
| 닫힌 목록 | `assigneeCandidatePersonId` 를 참석자 id 의 문자열 enum 으로. `unknown_person` 탈출구 포함 |
| 근거 강제 | `evidenceUtteranceId` 를 전달된 발화 id enum 으로 + `required`. 후처리에서 한 번 더 검증 |
| 기권 우선 | `dueDate` 는 `required` 아님 + 형식 깨지면 `None`. 틀린 값보다 빈 값 |
| 한 계층 한 목표 | 계층당 프롬프트 1개. 요약·배정·분류를 섞지 않는다 |

> 구조화 출력의 enum 은 **문자열만** 지원한다. 그래서 personId·utteranceId 를 문자열로 받고
> 후처리에서 정수로 되돌린다. 이 제약이 `unknown_person` 을 문자열 탈출구로 두는 이유이기도 하다.

> 모델이 스스로 말한 `confidence` 숫자는 받지 않는다. 자기보고 신뢰도는 85~95 에 몰리고
> 실제 정확도와 맞지 않는다. 자동 확정은 Spring 이 코드로 판정하는 4조건으로 한다.

---

## 로컬 실행

전제: [uv](https://docs.astral.sh/uv/) (파이썬 설치까지 uv 가 한다)

```bash
uv sync
cp .env.example .env      # INTERNAL_TOKEN · GEMINI_API_KEY 채우기
uv run uvicorn app.main:app --reload --port 8000
```

`DRY_RUN=1` 이면 Gemini 를 부르지 않고 고정 스텁을 돌려준다 — 라우팅·스키마만 볼 때 쓴다.

```bash
uv run pytest        # 테스트
uv run ruff check .  # 린트
```

### 모델 고정

`GEMINI_MODEL` 기본값은 BACKEND 리포 `GeminiModels.PINNED` 와 같은 구체 버전이다.
`-latest` 별칭은 Google 이 실제 모델을 교체하는 순간 코드가 그대로인데도 결과가 바뀌어
품질 지표를 신뢰할 수 없게 만든다.

---

## 실패 정책

| 분류 | 예 | 처리 | HTTP |
|---|---|---|---|
| `TRANSIENT` | 타임아웃 · 5xx · 순단 | 백오프 3회 (2s · 8s · 30s, 지터 ±20%) | 503 |
| `RATE_LIMIT` | 429 | `Retry-After` 존중 | 503 |
| `PERMANENT` | 스키마 위반 · 컨텍스트 초과 · 빈 응답 | 즉시 실패 | 422 |

`retryable` 을 응답 본문에 넣는다. Spring 이 메시지 문자열로 재시도를 추측하게 두면
영구 실패를 세 번 재시도해 토큰만 태운다 — 판정은 실패를 만든 쪽이 한다.
