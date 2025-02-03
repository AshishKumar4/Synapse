from typing import Any, Callable, Iterator, AsyncGenerator, Dict, List
from abc import ABC, abstractmethod
from synapse.utils import DataFrame, EventEmitter
from synapse.pipeline.sinks import DataSink

class DataSource(Iterator[DataFrame]):
    def __init__(self) -> None:
        super(DataSource, self).__init__()
        pass
    
    @abstractmethod
    def __next__(self) -> DataFrame:
        """ To be called to fetch next data """
        pass
    
    def __iter__(self) -> DataFrame:
        return self
    
    @abstractmethod
    def close(self):
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    
class EventDrivenDataSource(DataSource, EventEmitter):
    def __init__(self) -> None:
        super(EventDrivenDataSource, self).__init__()
        pass
    
    def drive(self, sink: DataSink):
        # Link the event handlers of the sink to the events of the source
        # print(f"Driving {sink} from {self}")
        handlers = sink.event_handlers()
        for event, handler in handlers.items():
            self.on(event, handler)