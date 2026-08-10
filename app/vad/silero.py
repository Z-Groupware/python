"""silero-vad(ONNX) 실행기 — PCM 을 발화 확률로 바꾼다.

<h2>onnxruntime 을 직접 쓴다</h2>
래퍼 패키지를 끼우지 않는다. 이 계층이 모델에서 쓰는 것은 입력 셋(`input`·`state`·`sr`)과
출력 둘(확률·다음 state)뿐이라, 래퍼가 주는 편의보다 그 API 가 버전마다 바뀌는 위험이 크다.
torch 를 끌어오지 않는 것도 이유다 — 추론만 하는 서버에 학습 프레임워크가 들어가면 이미지가
GB 단위로 커진다.

<h2>모델 파일은 이미지에 함께 넣는다</h2>
런타임에 받아오지 않는다. 모델이 바뀌면 절단점이 바뀌는데, 그게 배포와 무관하게 조용히
일어나면 "어제와 오늘의 블록 경계가 다른" 이유를 아무도 못 찾는다. 경로는 설정으로 받되
없으면 명확히 실패한다.

<h2>state 를 이어서 넘긴다</h2>
silero v5 는 RNN 이라 프레임 사이에 상태가 흐른다. 매 프레임 0 으로 초기화하면 모델이
직전 맥락을 잃고, 말 중간의 짧은 숨을 전부 무음으로 본다 — 그러면 700ms 조건을 만족하는
가짜 무음이 창마다 생긴다.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from app.errors import LayerError, LayerErrorKind
from app.vad.audio import FRAME_SAMPLES, SAMPLE_RATE

# silero v5 의 은닉 상태 모양. 모델이 요구하는 값이라 바꿀 수 없다.
_STATE_SHAPE = (2, 1, 128)

_lock = threading.Lock()
_session = None
_session_path: str | None = None


def _load(model_path: str):
    """세션을 한 번만 만든다.

    요청마다 만들면 20초 창 하나 처리하는 데 모델 로딩이 매번 붙는다. onnxruntime 세션은
    스레드 안전하므로 공유해도 된다 — 상태는 세션이 아니라 우리가 넘기는 `state` 에 있다.
    """
    global _session, _session_path

    with _lock:
        if _session is not None and _session_path == model_path:
            return _session

        if not Path(model_path).is_file():
            raise LayerError(
                LayerErrorKind.PERMANENT,
                "VAD_MODEL_MISSING",
                f"silero-vad 모델을 찾을 수 없습니다: {model_path}",
            )

        try:
            import onnxruntime
        except ImportError as exc:  # pragma: no cover - 의존성 누락은 배포 문제다
            raise LayerError(
                LayerErrorKind.PERMANENT,
                "VAD_RUNTIME_MISSING",
                "onnxruntime 이 설치되어 있지 않습니다.",
            ) from exc

        options = onnxruntime.SessionOptions()
        # CPU 한 장으로 20초를 도는 작업이다. 스레드를 늘리면 동시 요청끼리 코어를 뺏는다.
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1

        _session = onnxruntime.InferenceSession(
            model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        _session_path = model_path
        return _session


def speech_probs(pcm: bytes, model_path: str) -> list[float]:
    """s16le PCM → 프레임별 발화 확률.

    프레임을 못 채우는 꼬리는 버린다. 0 으로 채워 넣으면 그 패딩이 무음으로 잡혀서,
    창 끝에 실제로 없는 무음이 항상 하나 생긴다.
    """
    session = _load(model_path)

    samples = np.frombuffer(pcm, dtype=np.int16)
    usable = (len(samples) // FRAME_SAMPLES) * FRAME_SAMPLES
    if usable == 0:
        return []

    # int16 → float32 [-1, 1]. 모델이 정규화된 파형을 전제한다.
    audio = samples[:usable].astype(np.float32) / 32768.0
    frames = audio.reshape(-1, FRAME_SAMPLES)

    state = np.zeros(_STATE_SHAPE, dtype=np.float32)
    sample_rate = np.array(SAMPLE_RATE, dtype=np.int64)

    probs: list[float] = []
    for frame in frames:
        outputs = session.run(
            None,
            {"input": frame.reshape(1, -1), "state": state, "sr": sample_rate},
        )
        probs.append(float(outputs[0][0][0]))
        state = outputs[1]

    return probs
