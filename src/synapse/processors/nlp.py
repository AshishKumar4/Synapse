from termcolor import colored
from queue import Queue
from typing import Any, Dict, Callable
from concurrent.futures import Future

from synapse.pipeline.streamers.common import InterruptCascadeStreamer
from synapse.utils import GLOBAL_THREAD_POOL

import synapse.utils.stream2sentence as s2s
from synapse.utils import logger, AI_SPEECH_END_TOKEN
import traceback

from .ai_transcript import AITranscriptIterator

class Stream2Sentence(InterruptCascadeStreamer):
    """
    Processes incoming text chunks and generates sentences.
    """
    def __init__(self) -> None:
        super(Stream2Sentence, self).__init__()
        self.text_queue = Queue()
        s2s.initialize_nltk()
        logger.info("Stream2Sentence initialized")
        def char_iterator():
            try:
                while True:
                    char = self.text_queue.get()
                    if char is None or self.interrupted or self.is_closed:
                        self.text_queue.task_done()
                        break
                    yield char
                    self.text_queue.task_done()
            except Exception as e:
                logger.error(f"Error in char_iterator: {e}")
                traceback.print_exc()

        def sentence_generator():
            try:
                while True:
                    for sentence in s2s.generate_sentences(char_iterator(), log_characters=False):
                        print(colored(f"((Sentence: {sentence}))", "light_cyan"), end="")
                        if sentence is None:
                            continue
                        
                        if self.is_closed:
                            return
                        
                        if not self.interrupted:
                            self.commit(sentence)
            except Exception as e:
                logger.error(f"Error in sentence_generator: {e}")
                traceback.print_exc()
                        
        self.sentence_thread = GLOBAL_THREAD_POOL.submit(sentence_generator)

    def __call__(self, data):
        print(colored(f"((Stream2Sentence: {data}))", "light_cyan"), end="")
        if data == AI_SPEECH_END_TOKEN:
            self.text_queue.put(None)
        else:
            for char in data:
                self.text_queue.put(char)
    
    def handle_interrupt(self):
        print(colored("((Stream2Sentence: Interrupt))", "light_red"), end="")
        self.text_queue.put(None)
        return super().handle_interrupt()
    
    def handle_end(self):
        print(colored("((Stream2Sentence: End))", "light_red"), end="")
        # self.text_queue.put(None)
        return super().handle_end()
        
    def clear(self):
        print(colored("((Clearing TTS))", "light_red"), end="")
        with self.text_queue.mutex:
            self.text_queue.queue.clear()
            # Reset the unfinished tasks count to zero so that join() doesn’t block forever
            self.text_queue.unfinished_tasks = 0
            self.text_queue.all_tasks_done.notify_all()
        print(colored("((TTS cleared))", "light_red"), end="")
        return super().clear()
    
    def close(self):
        self.text_queue.put(None)
        return super().close()