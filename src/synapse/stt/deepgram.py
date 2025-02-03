from typing import Any, Callable, Dict, Iterator, AsyncGenerator
from elevenlabs import stream as elevenlabs_stream
import threading
import time
from termcolor import colored
from deepgram import (
    DeepgramClient,
    LiveTranscriptionEvents,
    LiveOptions,
)
from deepgram.clients.live.client import LiveClient, LiveResultResponse
from deepgram.clients.live.v1.response import Word, Alternative
from typing import Any, Callable, Iterator, AsyncGenerator, Dict, List
from abc import ABC, abstractmethod

from synapse.config import config
from synapse.pipeline.streamers.common import SpeechToTextStreamer

deepgram_client = DeepgramClient(config.DEEPGRAM_API_KEY)

def createDeepgramSocket(on_message_callback, channels, sample_rate):
    dg_connection : LiveClient = deepgram_client.listen.live.v("1")
    # Define event handlers
    def on_message(self, result:LiveResultResponse, **kwargs):
        arrived_time = time.time()
        on_message_callback(result, arrived_time)

    def on_metadata(self, metadata, **kwargs):
        print(f"Metadata: {metadata}")
    def on_error(self, error, **kwargs):
        print(f"Errors: {error}")
    def on_warning(self, warning, **kwargs):
        print(f"Warnings: {warning}")

    # Register event handlers
    dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
    dg_connection.on(LiveTranscriptionEvents.Metadata, on_metadata)
    dg_connection.on(LiveTranscriptionEvents.Error, on_error)
    dg_connection.on(LiveTranscriptionEvents.Warning, on_warning)

    # Configure Deepgram options for live transcription
    options = LiveOptions(
        model="nova-2-conversationalai", 
        language="en-IN", 
        smart_format=True,
        interim_results=True,
        channels=channels,
        sample_rate=sample_rate,
        encoding="linear16",
        utterance_end_ms="1000",
        endpointing=300,
        # vad_events=True,
    )
    # Start the connection
    dg_connection.start(options)
    return dg_connection

def measure_overlap(referenceWord: Word, targetWord: Word):
    reference_start = referenceWord.start
    reference_end = referenceWord.end
    target_start = targetWord.start
    target_end = targetWord.end
    overlap = max(0, min(reference_end, target_end) - max(reference_start, target_start))
    return overlap / (target_end - target_start)

class DeepgramTranscriptManager(ABC):
    def __init__(self):
        super(DeepgramTranscriptManager, self).__init__()
        self.uncommitted_words = []
        self.words_since_last_speech = []
        self._on_new_words_cb = lambda *x: print(colored(x, "yellow"), end='')
        self._on_sentence_end_cb = lambda: print(colored("<$S_END>", "yellow"), end='')
        self._on_speech_final_cb = lambda x: print(colored("<$S_FINAL{x}>", "red"), end='')
        self.transcript_response_lock = threading.Lock()
        
    @abstractmethod
    def commit_text(self, text: str, speaker="Ashish", arrived_time: float=None):
        pass
        
    def finalize_sentence(self):
        if len(self.uncommitted_words) == 0:
            return
        self.uncommitted_words = []
        if self._on_sentence_end_cb is not None:
            self._on_sentence_end_cb()
            # self.thread_pool.submit(self.on_sentence_end)
            
    def finalize_speech(self):
        words = self.words_since_last_speech
        self.words_since_last_speech = []
        self._on_speech_final_cb(words)
    
    def handle_new_words(self, words: list[str], speaker="Ashish", arrived_time: float=None):
        self.words_since_last_speech += words
        self.commit_text(words, speaker, arrived_time)
        if self._on_new_words_cb is not None:
            self._on_new_words_cb(words, speaker, arrived_time)
        
    def handle_transcript_response(self, result: LiveResultResponse, arrived_time: float):
        self.transcript_response_lock.acquire()
        try:
            if result.speech_final:
                print(colored("<$$final$$>", "red"), end='')
            words = result.channel.alternatives[0].words
            if len(words) == 0:
                self.finalize_sentence()
                self.transcript_response_lock.release()
                return
            
            identity_length = 0
            for i in range(min(len(words), len(self.uncommitted_words))):
                if words[i].word == self.uncommitted_words[i].word:
                    identity_length += 1
                else:
                    break
                # TODO: Handle past mispredictions and corrections using some events, identified using overlap
                
            new_words = words[identity_length:]
            # print(f"<identiy_length: {identity_length}, new_words: {new_words}, uncommitted_words: {self.uncommitted_words}>")
            if len(new_words) > 0:
                self.handle_new_words([i.punctuated_word for i in new_words], arrived_time=arrived_time)
            mispredicted_words = self.uncommitted_words[identity_length:]
            if len(mispredicted_words) > 0:
                # print(f"<!MISTAKE{' '.join([i.punctuated_word for i in mispredicted_words])}>", end='')
                self.handle_new_words([f"<!{' '.join([i.punctuated_word for i in mispredicted_words])}, iter={identity_length}>"])
            self.uncommitted_words = words
            
            if result.is_final:
                self.finalize_sentence()
            if result.speech_final:
                print(colored("<$$speech-final$$>", "red"), end='')
                self.finalize_speech()
        except Exception as e:
            print(colored(f"<!ERROR {e}, {result}>", "red"))
        self.transcript_response_lock.release()
        
    def on_new_words(self, callback):
        self._on_new_words_cb = callback
        
    def on_sentence_end(self, callback):
        self._on_sentence_end_cb = callback
        
    def on_speech_final(self, callback):
        self._on_speech_final_cb = callback

class DeepgramSTTStreamer(SpeechToTextStreamer, DeepgramTranscriptManager):
    def __init__(self, channels, sample_rate):
        super(DeepgramSTTStreamer, self).__init__()
        self.channels = channels
        self.sample_rate = sample_rate
        def on_message(result, arrived_time):
            self.handle_transcript_response(result, arrived_time)
        self.dg_connection = createDeepgramSocket(on_message, channels, sample_rate)
        self.on_sentence_end(lambda: self.trigger("sentence_end"))
        def on_speech_final(*x):
            self.trigger("speech_end", *x)
            self.speech_end()
        self.on_speech_final(on_speech_final)
        def on_new_words(*x):
            self.trigger("new_words", *x)
        self.on_new_words(on_new_words)
        
    def commit_text(self, text: str, speaker="Ashish", arrived_time: float = None):
        self.commit((text, speaker, arrived_time))
        
    def __call__(self, frame: bytes):
        self.dg_connection.send(frame)
        
    def close(self):
        super(DeepgramSTTStreamer, self).close()
        self.dg_connection.finish()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        