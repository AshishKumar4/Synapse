from typing import Any, Callable, Dict
from openai import Stream
from openai.resources.chat.completions import ChatCompletionChunk
from transformers import StoppingCriteria
import torch
import threading
from termcolor import colored
import openai
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from .types import InferenceRun

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
vllm_client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

class OpenAIInferenceRun(InferenceRun):
    global_run_id: int = 0
    def __init__(
        self,
        model:str,
        prior_fetcher: Callable[[], str], 
        thread_pool:ThreadPoolExecutor,
        max_tokens=1000,
        flush_rate=3,
    ):
        """
        :param messages: A list of message dicts in the OpenAI chat format.
        :param on_token_callback: Called with each token (string) as it is streamed.
        :param on_complete_callback: Called when the generation is complete.
        :param on_error_callback: Called in case of an error.
        """
        super(OpenAIInferenceRun, self).__init__(model, prior_fetcher, thread_pool, max_tokens)
        self.response: Stream[ChatCompletionChunk] = None
        print(colored(f'<@@gen run setup:{self.run_id}>', "yellow"), end='')
        self.flush_rate = flush_rate
        self.__run__()
        self.lock.release()

    def __run__(self):
        def __generate_fn():
            print(colored(f'<@@gen starting run:{self.run_id}>', "yellow"), end='')
            prior = self.prior_fetcher()
            messages = prior
            try:
                client = openai if self.model.startswith("gpt") else vllm_client
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    stream=True,  # important for streaming
                )
                self.response = response
                print(colored(f'<@@gen started {self.run_id}>', "yellow"), end='')
            except Exception as e:
                print(colored(f"[OpenAIInferenceRun Error]: {e}", "red"))
        self.run_future = self.thread_pool.submit(__generate_fn)
        
    def flush(self, on_start_callback=None, on_end_callback=None, on_word_callback=None):
        with self.lock:
            if on_start_callback is not None:
                on_start_callback()
            def __flush_fn():
                print(colored(f'<@@flush started {self.run_id}>', "yellow"), end='')
                self.run_future.result()
                if self.response is not None:
                    buffer = ""
                    try:
                        for message in self.response:
                            if self.cancelled:
                                print(colored(f'<@@flush cancelled {self.run_id}>', "yellow"), end='')
                                if on_end_callback is not None:
                                    on_end_callback()
                                return
                            delta = message.choices[0].delta.content or ""
                            buffer += delta
                            if buffer.count(" ") >= self.flush_rate:
                                # Flush buffer if it has more than flush_rate words
                                if on_word_callback is not None:
                                    on_word_callback(buffer)
                                buffer = ""
                        if buffer:
                            if on_word_callback is not None:
                                on_word_callback(buffer)
                        if on_end_callback is not None:
                            on_end_callback()
                        print(colored(f'<@@flush done {self.run_id}>', "yellow"), end='')
                    except Exception as e:
                        print(colored(f'<@@flush failed {self.run_id}, {e}>', "yellow"), end='')
                        if on_end_callback is not None:
                            on_end_callback()
                else:
                    print(colored(f'<@@flush no response {self.run_id}>', "yellow"), end='')
            self.flush_future = self.thread_pool.submit(__flush_fn)
    
    def cancel(self):
        if self.cancelled:
            return
        self.cancelled = True
        try:
            if self.response is not None:
                self.response.close()
        except Exception as e:
            pass
        if self.run_future is not None and not self.run_future.done():
            self.run_future.cancel()
        if self.flush_future is not None and not self.flush_future.done():
            self.flush_future.cancel()