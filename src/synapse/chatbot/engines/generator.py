from transformers import PreTrainedModel, PreTrainedTokenizerFast
from transformers import TextIteratorStreamer
import threading
from termcolor import colored
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from .openai import OpenAIInferenceRun
from .utils import InterruptibleStoppingCriteria
from .types import InferenceRun

from synapse.utils import GLOBAL_THREAD_POOL

class LLMGenerator:
    def __init__(self, model:PreTrainedModel | str, tokenizer:PreTrainedTokenizerFast = None, max_tokens=1000):
        self.lock = threading.Lock()
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.thread_pool = GLOBAL_THREAD_POOL
        self.current_run: InferenceRun = None
        self.need_to_start_new_run = False
        
    def __start_new_run__(self, prior_fetcher: Callable):
        self.need_to_start_new_run = False
        if isinstance(self.model, str):
            # Use OpenAI API
            self.current_run = OpenAIInferenceRun(model=self.model, prior_fetcher=prior_fetcher, thread_pool=self.thread_pool, max_tokens=self.max_tokens)
        else:
            # Use Huggingface model
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            stopper = InterruptibleStoppingCriteria()
            self.current_run = InferenceRun(model=self.model, tokenizer=self.tokenizer, 
                                            streamer=streamer, stopper=stopper, 
                                            prior_fetcher=prior_fetcher, thread_pool=self.thread_pool, 
                                            max_tokens=self.max_tokens)
        return self.current_run
        
    def generate(self, prior_fetcher: Callable, on_run_start:Callable[[InferenceRun], None]=None):
        with self.lock:
            if self.current_run is not None:
                self.current_run.cancel()
            # if self.current_run is not None:
            #     if self.current_run.is_cancelled():
            #         if not self.current_run.run_future.done():
            #             # Do not start a new run if the current run is cancelled but not done
            #             self.need_to_start_new_run = True
            #             def __maybe_start_new_run():
            #                 self.lock.acquire()
            #                 if self.need_to_start_new_run:
            #                     print(colored(f'<@@gen debounced>', "red"), end='')
            #                     run = self.__start_new_run__(prior_fetcher)
            #                     if on_run_start is not None:
            #                         on_run_start(run)
            #                 self.lock.release()
                                
            #             self.current_run.run_future.add_done_callback(__maybe_start_new_run)
            #             print(colored(f'<@@gen skipped, last run didnt end>', "red"), end='')
            #             self.lock.release()
            #             return
            #     else:
            #         self.current_run.cancel()
                    
            run = self.__start_new_run__(prior_fetcher)
            if on_run_start is not None:
                on_run_start(run)
    
    def get_current_run(self) -> InferenceRun:
        return self.current_run
    
    def cancel_current_run(self):
        current_run = self.current_run
        if current_run is not None:
            current_run.cancel()
            current_run = None
            
    def wait_for_flush(self):
        if self.current_run is not None:
            self.current_run.wait_for_flush()
            
    def exit(self):
        if self.current_run is not None:
            self.current_run.cancel()
        # self.thread_pool.shutdown()