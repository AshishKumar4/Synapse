from typing import Any, Callable, Iterator, AsyncGenerator, Dict, List
from abc import ABC, abstractmethod
from synapse.utils import DataFrame

class DataSink(ABC):    
    def __init__(self) -> None:
        super(DataSink, self).__init__()
        pass
    
    def event_handlers(self) -> Dict[str, Callable]:
        return {}
    
    @abstractmethod
    def __call__(self, data: DataFrame):
        """ To be called on new data """
        pass

    @abstractmethod
    def close(self):
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()