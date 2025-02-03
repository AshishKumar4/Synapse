from transformers import StoppingCriteria
import torch
import threading
from termcolor import colored

class InterruptibleStoppingCriteria(StoppingCriteria):
    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._should_interrupt = False

    def interrupt(self):
        with self._lock:
            self._should_interrupt = True

    def reset(self):
        with self._lock:
            self._should_interrupt = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        with self._lock:
            if self._should_interrupt:
                # Reset here so if generation tries again we don't keep interrupting
                self._should_interrupt = False
                print(colored(f'<@@int>', 'red'), end='')
                return True
        return False