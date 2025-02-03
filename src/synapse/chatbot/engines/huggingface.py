from typing import Any, Callable, Dict
from openai import Stream
from openai.resources.chat.completions import ChatCompletionChunk
from transformers import StoppingCriteria
import torch
import threading
from termcolor import colored
import openai
from concurrent.futures import ThreadPoolExecutor
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from transformers import AutoTokenizer, TextStreamer, PreTrainedModel, PreTrainedTokenizerFast, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList

from .types import InferenceRun
from .utils import InterruptibleStoppingCriteria

class LLMInferenceRun(InferenceRun):
    global_run_id: int = 0
    def __init__(self, 
                 model:PreTrainedModel, 
                 tokenizer:PreTrainedTokenizerFast, 
                 streamer:TextStreamer,
                 stopper:InterruptibleStoppingCriteria,
                 prior_fetcher: Callable[[], str], 
                 thread_pool:ThreadPoolExecutor,
                 max_tokens=1000):
        super(LLMInferenceRun, self).__init__(model, prior_fetcher, thread_pool, max_tokens)
        self.streamer = streamer
        self.stopper = stopper
        self.tokenizer = tokenizer
        self.__run__() 
        self.lock.release()
        
    def __run__(self):
        def __generate_fn():
            print(colored(f'<@@gen starting run:{self.run_id}>', "yellow"), end='')
            prior = self.prior_fetcher()
            messages = prior
            try:
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                model_inputs = self.tokenizer([text], return_tensors="pt").to("cuda")
                if self.run_future.cancelled() or self.cancelled:
                    print(colored(f'<@@gen precancelled {self.run_id}>', "light_red"), end='')
                    return
                print(colored(f'<@@gen started {self.run_id}>', "yellow"), end='')
                _ = self.model.generate(**model_inputs, do_sample=True, penalty_alpha=0.6, top_k=5, max_new_tokens=self.max_tokens, streamer=self.streamer, tokenizer=self.tokenizer, stopping_criteria=StoppingCriteriaList([self.stopper]))
                print(colored(f'<@@gen done {self.run_id}>', "yellow"), end='')
            except Exception as e:
                print(colored(f'<@@gen failed {self.run_id}, {e}>', "yellow"), end='')
        self.run_future = self.thread_pool.submit(__generate_fn)
    
    def flush(self, on_start_callback=None, on_end_callback=None, on_word_callback=None):
        with self.lock:
            if on_start_callback is not None:
                on_start_callback()
            def __flush_fn():
                print(colored(f'<@@flush started {self.run_id}>', "yellow"), end='')
                for word in self.streamer:
                    if self.cancelled:
                        print(colored(f'<@@flush cancelled {self.run_id}>', "yellow"), end='')
                        if on_end_callback is not None:
                            on_end_callback()
                        return
                    if on_word_callback is not None:
                        on_word_callback(word)#' '.join(processed))
                if on_end_callback is not None:
                    on_end_callback()
                print(colored(f'<@@flush done {self.run_id}>', "yellow"), end='')
                
            self.flush_future = self.thread_pool.submit(__flush_fn)
    
    def cancel(self):
        if self.cancelled:
            return
        self.cancelled = True
        self.stopper.interrupt()
        if self.run_future is not None and not self.run_future.done():
            print(colored(f'<@@intr old gen {self.run_id}>', 'red'), end='')
            self.run_future.cancel()
        if self.flush_future is not None and not self.flush_future.done():
            print(colored(f'<@@intr old flush {self.run_id}>', 'red'), end='')
            self.flush_future.cancel()
        def __cancel_fn():
            if self.run_future is not None and not self.run_future.done():
                try:
                    self.run_future.result()
                except Exception as e:
                    pass
                print(colored(f'<@@intr gen done {self.run_id}>', 'red'), end='')
            if self.flush_future is not None and not self.flush_future.done():
                try:
                    self.flush_future.result()
                except Exception as e:
                    pass
                print(colored(f'<@@intr flush done {self.run_id}>', 'red'), end='')
        self.thread_pool.submit(__cancel_fn)
        
    def wait_for_flush(self):
        self.lock.acquire()
        if self.flush_future is not None:
            self.flush_future.result()
        else:
            self.lock.release()
            raise Exception("No flush future to wait for")
        self.lock.release()
        
    def is_cancelled(self):
        return self.cancelled
