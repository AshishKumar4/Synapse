from queue import Queue
from typing import Any, Dict, Callable
import threading
from termcolor import colored

from synapse.pipeline.streamers.common import CancellableText2SpeechStreamer
from .utils import float32_to_pcm16
from synapse.utils.stream2sentence import generate_sentences

class KokoroTTS(CancellableText2SpeechStreamer):
    
    """
    Drop-in replacement for ElevenLabsTTS_WS, but uses Kokoro TTS locally.
    Extends CancellableText2SpeechStreamer to integrate with your pipeline.
    """

    def __init__(
        self,
        sample_rate=24000,
        lang_code='a',
        voice_id='af_heart',
        speed=1.1,
    ):
        """
        :param sample_rate: Bark/Kokoro typically use 24k. 
                           Adjust to match your LocalSpeaker if needed.
        :param lang_code:  Passed to KPipeline (e.g. 'en', 'a', etc).
        :param voice_id:   The voice name or file path (like 'af_heart').
        :param speed:      Playback speed factor, used by Kokoro engine.
        # :param split_pattern: Regex for chunk-splitting input text in Kokoro.
        """
        from kokoro import KPipeline

        super(KokoroTTS, self).__init__()
        self.sample_rate = sample_rate
        self.lang_code = lang_code
        self.voice_id = voice_id
        self.speed = speed
        # self.split_pattern = split_pattern

        # Kokoro pipeline instance
        self.pipeline = KPipeline(lang_code=self.lang_code)

    def __call__(self, data: str):
        generator = self.pipeline(
            data,
            voice=self.voice_id,
            speed=self.speed,
            # split_pattern=self.split_pattern
        )
        for i, (gs, ps, audio_tensor) in enumerate(generator):
            if self.interrupted or self.is_closed:
                break
            pcm_bytes = float32_to_pcm16(audio_tensor.numpy())
            self.commit(pcm_bytes)

    def close(self):
        """
        Closes the pipeline, the background thread, etc.
        """
        super(KokoroTTS, self).close()
        print(colored("((KokoroTTS closed))", "light_red"))

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()