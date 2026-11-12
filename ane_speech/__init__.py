"""Speech recognition on the Apple Neural Engine.

    import ane_speech

    asr = ane_speech.load("qwen3_asr_1_7b")     # downloads the model on first use
    print(asr.generate("speech.wav").text)

    live = ane_speech.load("confucius4_r2t2")
    session = live.stream_session(chunk_ms=480)
    for pcm in microphone:                       # float32, 16 kHz mono
        print(session.feed(pcm).committed)
    print(session.finalize().text)
"""

from importlib.metadata import version

from ane_speech.hub import CHECKPOINTS, DEFAULT, pull
from ane_speech.model import Model, StreamSession, StreamUpdate, Transcript, load

__version__ = version("ane-speech")
__all__ = [
    "CHECKPOINTS", "DEFAULT", "Model", "StreamSession", "StreamUpdate", "Transcript", "load", "pull",
]
