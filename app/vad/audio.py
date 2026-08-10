"""오디오 디코딩 — 컨테이너 무엇이든 16kHz mono PCM 으로.

<h2>왜 ffmpeg 인가</h2>
청크는 브라우저가 만든 `.webm`(Opus)이다. silero 는 **16kHz mono float PCM** 만 먹는다.
파이썬 순수 구현으로 Opus 를 풀 방법이 마땅치 않아 시스템 ffmpeg 에 맡긴다 — 런타임 이미지에
설치되어 있어야 한다(Dockerfile).

<h2>필요한 구간만 뽑는다</h2>
10분짜리 파일을 통째로 디코딩하면 20초를 보려고 그 30배를 푼다. `-ss`/`-t` 로 잘라서
읽으므로 메모리도 시간도 창 크기에 비례한다.
"""

from __future__ import annotations

import asyncio

from app.errors import LayerError, LayerErrorKind

# silero 가 지원하는 샘플레이트는 8k·16k 뿐이다. 16k 를 쓴다 — 8k 는 자음 구분이 나빠져
# 무음 판정이 흔들린다.
SAMPLE_RATE = 16_000

# silero 가 16kHz 에서 요구하는 고정 창 크기(샘플). 이 값이 곧 프레임 하나의 길이다.
FRAME_SAMPLES = 512
FRAME_MS = FRAME_SAMPLES * 1000 // SAMPLE_RATE  # 32ms

_FFMPEG = "ffmpeg"


async def decode_window(audio: bytes, start_ms: int, duration_ms: int) -> bytes:
    """오디오 바이트에서 [start, start+duration) 구간을 s16le PCM 으로 뽑는다.

    stdin 으로 넣고 stdout 으로 받는다. 임시 파일을 쓰지 않는 이유는 실패 경로에서 남는
    파일을 지울 책임이 생기고, 그 정리가 빠지면 디스크가 조용히 찬다.

    ⚠ `-ss` 를 **입력 앞에** 둔다. 출력 쪽에 두면 ffmpeg 이 파일 앞부터 전부 디코딩한 뒤
    버리므로, 구간만 뽑으려던 이유가 사라진다.
    """
    args = [
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_ms / 1000:.3f}",
        "-t",
        f"{duration_ms / 1000:.3f}",
        "-i",
        "pipe:0",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "pipe:1",
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            _FFMPEG,
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        # 이미지에 ffmpeg 이 없다. 재시도해도 같으므로 PERMANENT 다 — 배포 문제이지
        # 이 요청의 문제가 아니라는 것이 코드에 드러나야 한다.
        raise LayerError(
            LayerErrorKind.PERMANENT,
            "FFMPEG_MISSING",
            "ffmpeg 을 찾을 수 없습니다. 런타임 이미지에 설치되어야 합니다.",
        ) from exc

    stdout, stderr = await process.communicate(audio)

    if process.returncode != 0:
        # 깨진 파일이나 지원하지 않는 코덱이다. 같은 입력으로 다시 보내도 같은 결과라
        # 재시도하지 않는다(errors.py 의 PERMANENT 정의와 같은 판단).
        raise LayerError(
            LayerErrorKind.PERMANENT,
            "AUDIO_DECODE_FAILED",
            f"오디오를 디코딩하지 못했습니다: {stderr.decode('utf-8', 'replace')[:300]}",
        )

    return stdout


def frame_count(pcm: bytes) -> int:
    """PCM 이 몇 프레임인가. 남는 꼬리는 버린다 — silero 가 고정 길이만 받는다."""
    return len(pcm) // (FRAME_SAMPLES * 2)
