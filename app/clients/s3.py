"""S3 — 오디오는 본문이 아니라 키로 온다.

명세 「내부 API 규칙」이 못박은 것이다 — *"파일 전달: S3 경유. 로컬 볼륨 공유가 없다 —
모든 오디오 참조는 S3 키로 넘긴다."* 두 EC2 사이에 공유 디스크가 없고, 20초짜리 오디오를
HTTP 본문에 실으면 Spring 이 먼저 받아 디코딩까지 해야 해서 같은 일을 두 곳에서 하게 된다.

<h2>자격증명을 설정으로 받지 않는다</h2>
AI EC2 의 IAM 롤을 boto3 가 알아서 집는다. 액세스 키를 .env 에 두면 그 파일이 유출 표면이
되고, 롤과 달리 자동으로 돌지도 않는다. 로컬 개발에서는 평소 쓰던 AWS 프로필이 그대로 잡힌다.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

from app.errors import LayerError, LayerErrorKind


@lru_cache
def _client(region: str):
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - 의존성 누락은 배포 문제다
        raise LayerError(
            LayerErrorKind.PERMANENT,
            "S3_RUNTIME_MISSING",
            "boto3 가 설치되어 있지 않습니다.",
        ) from exc

    return boto3.client("s3", region_name=region or None)


async def fetch(bucket: str, key: str, region: str = "") -> bytes:
    """객체를 통째로 받아온다.

    boto3 는 동기라 스레드로 뺀다 — 이벤트 루프에서 그대로 부르면 다운로드 동안 다른 요청이
    전부 멈춘다. 계층 서버는 한 번에 여러 회의를 받는다.

    <h2>범위 요청(Range)을 쓰지 않는다</h2>
    청크 하나가 15초 분량이라 파일 자체가 작고, webm 은 컨테이너라 바이트 범위와 시간 범위가
    대응하지 않는다. 시간으로 자르는 것은 디코딩 단계(ffmpeg -ss)의 몫이다.
    """
    client = _client(region)

    def _get() -> bytes:
        return client.get_object(Bucket=bucket, Key=key)["Body"].read()

    try:
        return await asyncio.to_thread(_get)
    except Exception as exc:
        raise _classify(exc, bucket, key) from exc


def _classify(exc: Exception, bucket: str, key: str) -> LayerError:
    """S3 실패를 재시도 가능/불가로 나눈다.

    없는 키를 세 번 더 물어봐야 여전히 없다. 반대로 순단·스로틀링은 기다리면 풀린다 —
    둘을 섞으면 한쪽은 헛돌고 다른 한쪽은 회복할 기회를 잃는다(errors.py 와 같은 판단).
    """
    code = ""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = str(response.get("Error", {}).get("Code", ""))

    if code in {"NoSuchKey", "NoSuchBucket", "404", "AccessDenied", "403"}:
        return LayerError(
            LayerErrorKind.PERMANENT,
            "AUDIO_NOT_FOUND",
            f"오디오를 읽을 수 없습니다(s3://{bucket}/{key}): {code or exc}",
        )

    if code in {"SlowDown", "RequestLimitExceeded", "Throttling", "ThrottlingException"}:
        return LayerError(LayerErrorKind.RATE_LIMIT, "S3_THROTTLED", str(exc))

    return LayerError(LayerErrorKind.TRANSIENT, "S3_UNAVAILABLE", str(exc))
