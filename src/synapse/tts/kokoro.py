from queue import Queue
from typing import Any, Dict, Callable
import threading
from termcolor import colored

from synapse.pipeline.streamers.common import CancellableText2SpeechStreamer
from .utils import float32_to_pcm16

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

        # Queue for text chunks from __call__()
        self.text_queue = Queue()

        # Start background thread to produce PCM frames
        self.read_from_thread = threading.Thread(target=self._audio_generator, daemon=True)
        self.read_from_thread.start()

    def __call__(self, data: str):
        """
        Receives text from the AITranscriptIterator. 
        If data == SPEECH_END_TOKEN, we flush. Otherwise, queue text for TTS.
        """
        if data == "<$AI_SPEECH_END>":
            print(colored(f"((Sending end))", "light_yellow"), end='')
            # Enqueue a sentinel to mark flush or speech-end
            self.text_queue.put(None)  # or some marker
        else:
            # Normal text chunk
            # self.text_queue.put(data)
            for char in data:
                self.text_queue.put(char)

    def _audio_generator(self):
        import stream2sentence as s2s
        """
        Continuously reads text chunks from text_queue, calls Kokoro TTS,
        and commits frames to the DataStreamer. 
        Mimics the structure of __fetch_frames__ + audio generator in ElevenLabsTTS_WS.
        """
        def char_iterator():
            while True:
                char = self.text_queue.get()
                if char is None or self.interrupted or self.is_closed:
                    break
                yield char
                self.text_queue.task_done()
        while True:
            for text in s2s.generate_sentences(char_iterator()):
                # print(colored(f"((Got text: {text}))", "light_yellow"), end='')
                # If we got None => flush / speech end
                if text is None:
                    # Mark speech done, you can reset self.iter = 0, etc.
                    # For now, we do nothing special besides printing.
                    print(colored(f"((Got speech-end sentinel))", "light_yellow"), end='')
                    continue

                if self.is_closed:
                    # If the streamer has closed, drain queue & exit
                    continue

                # Actually do TTS if not interrupted
                if not self.interrupted and text.strip():
                    # Generate partial audio in a streaming manner
                    generator = self.pipeline(
                        text,
                        voice=self.voice_id,
                        speed=self.speed,
                        # split_pattern=self.split_pattern
                    )
                    # generator yields tuples like (graphemes, phonemes, audio_tensor)
                    # We feed them chunk by chunk
                    for i, (gs, ps, audio_tensor) in enumerate(generator):
                        if self.interrupted or self.is_closed:
                            break
                        pcm_bytes = float32_to_pcm16(audio_tensor.numpy())
                        self.commit(pcm_bytes)

                # If interrupted in the middle, drain leftover items from generator
                # or do any cleanup as needed
        
    def clear(self):
        with self.text_queue.mutex:
            self.text_queue.queue.clear()
            # Reset the unfinished tasks count to zero so that join() doesn’t block forever
            self.text_queue.unfinished_tasks = 0
            self.text_queue.all_tasks_done.notify_all()
        return super().clear()
    
    def close(self):
        """
        Closes the pipeline, the background thread, etc.
        """
        super(KokoroTTS, self).close()
        self.is_closed = True
        # Put enough sentinels so that the worker definitely ends
        self.text_queue.put(None)
        self.text_queue.put(None)
        self.read_from_thread.join(timeout=2.0)
        print(colored("((KokoroTTS closed))", "light_red"))

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()