from typing import Callable, Dict, Any
from queue import Queue
import threading
import time
from termcolor import colored
from synapse.utils import DataFrame
from synapse.pipeline.streamers.common import DataStreamer, EventDrivenDataStreamer
import traceback
from synapse.utils import AI_SPEECH_END_TOKEN

SPEAKER_COLOR_MAP = {
    "user": "blue",
    "assistant": "green"
}

SPEAKER_TYPES_MAP = {
}

class GlobalTranscript(EventDrivenDataStreamer):
    def __init__(self, transcript_file=None) -> None:
        super(GlobalTranscript, self).__init__()
        # self.transcript = []
        self.past_transcripts = []
        self.current_text = ""
        self.current_speaker = None
        word_queue = Queue()
        self.word_queue = word_queue
        self.lock = threading.Lock()
        self.speaker_change_timings = []
        self.last_commit_at = time.time()
        self.transcript_file = open(transcript_file, "w") if transcript_file is not None else None
        
    def event_handlers(self) -> Dict[str, Callable[..., Any]]:
        """
        Simply forward the events to the triggers
        """
        return {
            "new_words": lambda *args, **kwargs: self.trigger("new_words", *args, **kwargs),
            "sentence_end": lambda *args, **kwargs: self.trigger("sentence_end", *args, **kwargs),
            "speech_end": lambda *args, **kwargs: self.trigger("speech_end", *args, **kwargs)
        }
        
    def commit_word(self, word, speaker, arrived_time=None, is_ai=False):
        with self.lock:
            self.__commit_word(word, speaker, arrived_time, is_ai)
    
    def __commit_word(self, word, speaker, arrived_time=None, is_ai=False):
        if speaker not in SPEAKER_TYPES_MAP:
            SPEAKER_TYPES_MAP[speaker] = "user" if not is_ai else "assistant"
        self.commit((word, speaker))
        self.last_commit_at = time.time()
        
    def commit(self, data: DataFrame):
        try:
            text, speaker = data
            speaker_type = SPEAKER_TYPES_MAP.get(speaker, "assistant")
            color = SPEAKER_COLOR_MAP.get(speaker_type, "green")
            text_to_render = ''
            if speaker != self.current_speaker:
                # Speaker changed, commit a new line and add current text to past transcripts
                speaker_change_time = time.time() - self.last_commit_at
                self.speaker_change_timings.append({
                    "last_speaker": self.current_speaker,
                    "new_speaker": speaker,
                    "time": speaker_change_time,
                })
                if self.current_speaker is not None:
                    self.past_transcripts.append({ "role": speaker_type, "content": self.current_text })
                        
                self.trigger(
                    "speaker_change",
                    old_speaker=self.current_speaker, 
                    old_speaker_type=SPEAKER_TYPES_MAP.get(self.current_speaker, "assistant"), 
                    new_speaker=speaker, 
                    new_speaker_type=speaker_type, 
                    time=speaker_change_time
                )
                self.current_speaker = speaker
                self.current_text = ""
                text_to_render = colored(f'\n{speaker}:', color)
                
            if text != AI_SPEECH_END_TOKEN:
                text_to_render += colored(text, color)
                print(text_to_render, end="")
                if self.transcript_file is not None:
                    self.transcript_file.write(text_to_render)
                    self.transcript_file.flush()
                self.current_text += text
            return super().commit(data)
        except Exception as e:
            print(f"Error while committing {data}: {e}", traceback.format_exc())
        
    def __call__(self, data: DataFrame):
        self.commit_word(*data)
    
    def commit_punctuation(self, punctuation):
        self.commit_word(punctuation, self.current_speaker)
        
    def sync(self):
        self.word_queue.join()
        
    def get_transcript(self):
        past_transcripts = list(self.past_transcripts)
        # print("transcripts", past_transcripts)
        return past_transcripts + [{ "role": SPEAKER_TYPES_MAP[self.current_speaker], "content": self.current_text }]
    
    def get_speaker_change_timings(self):
        return self.speaker_change_timings
    
    def close(self):
        super(GlobalTranscript, self).close()
        self.word_queue.put(None)
        self.word_queue.join(timeout=1)
        self.sync()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        