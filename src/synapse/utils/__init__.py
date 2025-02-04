from typing import Any, Callable, Dict, List
from abc import ABC
from concurrent.futures import ThreadPoolExecutor
from .logger import *

GLOBAL_THREAD_POOL = ThreadPoolExecutor(max_workers=32)
AI_SPEECH_END_TOKEN = "<$AI_SPEECH_END>"

class DataFrame(Any):
    pass

class EventEmitter(ABC):
    def __init__(self) -> None:
        super(EventEmitter, self).__init__()
        self.handlers : Dict[str, List[Callable]] = {}
    
    def events(self) -> List[str]:
        return list(self.handlers.keys())
    
    def on(self, event: str, handler: Callable):
        updated = self.handlers.get(event, [])
        updated.append(handler)
        self.handlers[event] = updated
        
    def trigger(self, event: str, *args, **kwargs):
        handlers = self.handlers.get(event, [])
        for handler in handlers:
            handler(*args, **kwargs)
