import threading
from abc import ABC, abstractmethod
from typing import Callable
from concurrent.futures import ThreadPoolExecutor
from transformers import PreTrainedModel


class InferenceRun(ABC):
    global_run_id = 0
    def __init__(
                 self,
                 model:PreTrainedModel | str,
                 prior_fetcher: Callable[[], str],
                 thread_pool:ThreadPoolExecutor,
                 max_tokens=1000,):
        self.lock = threading.Lock()
        self.lock.acquire()
        self.run_id = InferenceRun.global_run_id
        InferenceRun.global_run_id += 1
        self.thread_pool = thread_pool
        self.max_tokens = max_tokens
        self.model = model
        self.prior_fetcher = prior_fetcher
        self.cancelled = False
        self.flush_future = None
        
    @abstractmethod
    def flush(self, on_start_callback=None, on_end_callback=None, on_word_callback=None):
        pass
    
    @abstractmethod
    def cancel(self):
        pass
        
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