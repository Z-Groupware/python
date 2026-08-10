"""silero ONNX 계약 — 모델 파일이 코드가 기대하는 모양인가.

<h2>왜 이 테스트가 있나</h2>
`app/vad/silero.py` 는 텐서 **이름과 모양**을 직접 안다(`input`·`state`·`sr` → 확률·다음 state).
그 가정이 틀리면 첫 호출에서 깨지는데, 나머지 테스트는 전부 순수 함수 위에서 돌기 때문에
아무도 잡지 못한다.

특히 위험한 순간은 **모델을 새 버전으로 갈아끼울 때**다. v4 계열은 state 가 h·c 두 개로
나뉘어 있었고, 그런 변화는 파일만 바꿔도 조용히 들어온다. 그때 여기서 빨간불이 떠야 한다.

<h2>모델이 없으면 건너뛴다</h2>
파일은 이미지에 함께 넣는 것이라 개발 환경에 없을 수 있다. 없는 것을 실패로 만들면 사람들이
`-k` 로 빼기 시작하고, 그러면 있을 때도 안 돌게 된다.
"""

from pathlib import Path

import numpy as np
import pytest

from app.vad.audio import FRAME_SAMPLES, SAMPLE_RATE
from app.vad.silero import speech_probs

MODEL = Path(__file__).resolve().parent.parent / "models" / "silero_vad.onnx"

pytestmark = pytest.mark.skipif(not MODEL.is_file(), reason=f"모델 파일이 없다: {MODEL}")


def test_모델이_코드가_아는_이름과_모양을_요구한다():
    import onnxruntime

    session = onnxruntime.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    inputs = {i.name: i for i in session.get_inputs()}

    # silero.py 가 이 이름들로 넘긴다. 하나라도 다르면 첫 호출에서 깨진다.
    assert set(inputs) == {"input", "state", "sr"}
    # state 는 [2, batch, 128]. 가운데가 동적이라 우리가 1 로 쓴다.
    assert inputs["state"].shape[0] == 2
    assert inputs["state"].shape[2] == 128
    # 출력 순서에 기대고 있다 — [0] 확률, [1] 다음 state.
    assert len(session.get_outputs()) == 2


def test_프레임_수만큼_확률이_나온다():
    """플러밍 검증이다 — state 를 이어 넘기는 것까지 포함해 끝까지 도는지 본다.

    ⚠ **판정 품질은 검증하지 않는다.** 합성 신호는 사람 목소리가 아니라 silero 가 무음으로
    보는 것이 정상이고, 여기서 확률의 절대값을 못박으면 모델을 올릴 때마다 의미 없이 깨진다.
    실제 음성으로 확인하는 것은 gold set 회의를 돌려 보는 자리다.
    """
    frames = 30
    pcm = (np.zeros(FRAME_SAMPLES * frames) * 32767).astype(np.int16).tobytes()

    probs = speech_probs(pcm, str(MODEL))

    assert len(probs) == frames
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_프레임을_못_채우면_빈_결과다():
    # 0 으로 채워 넣으면 그 패딩이 무음으로 잡혀 없는 무음이 하나 생긴다.
    pcm = (np.zeros(FRAME_SAMPLES - 1)).astype(np.int16).tobytes()

    assert speech_probs(pcm, str(MODEL)) == []


def test_모델이_없으면_명확히_거절한다():
    from app.errors import LayerError, LayerErrorKind

    with pytest.raises(LayerError) as caught:
        speech_probs(b"\x00" * (FRAME_SAMPLES * 2), str(MODEL.parent / "없는파일.onnx"))

    # 조용히 통과시키면 무음을 하나도 못 찾아 모든 블록이 FALLBACK_OVERLAP 이 되고,
    # 그게 "절단이 잘 안 되는 회의"로 위장된다.
    assert caught.value.code == "VAD_MODEL_MISSING"
    assert caught.value.kind is LayerErrorKind.PERMANENT


def test_샘플레이트가_16k_로_고정돼_있다():
    # 8k 를 쓰면 자음 구분이 나빠져 무음 판정이 흔들린다. 값이 바뀌면 프레임 길이(32ms)도
    # 함께 어긋나 절단점이 통째로 밀린다.
    assert SAMPLE_RATE == 16_000
    assert FRAME_SAMPLES == 512
