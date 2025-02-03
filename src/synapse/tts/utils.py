
import numpy as np

def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """
    Convert float32 [-1.0, 1.0] waveform to int16 PCM bytes (mono).
    """
    # shape could be (channels, samples)
    if audio.ndim == 2:
        audio = audio[0]  # take first channel for mono
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767.0).astype(np.int16)
    return audio_int16.tobytes()
