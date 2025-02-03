from typing import List, Dict, Iterator
from loguru import logger
from queue import Queue
from termcolor import colored
import threading
import time
from abc import ABC, abstractmethod

from synapse.utils import DataFrame
from .types import DataStreamer, EventDrivenDataStreamer

class DataAggregator(EventDrivenDataStreamer):
    def __init__(self,) -> None:
        super(DataAggregator, self).__init__()
        
class StreamerGenerator():
    def __init__(self, streamerClass: DataStreamer, *args, **kwargs) -> None:
        super(StreamerGenerator, self).__init__()
        self.streamerClass = streamerClass
        self.args = args
        self.kwargs = kwargs
        pass
    
    def generate(self) -> DataStreamer:
        return self.streamerClass(*self.args, **self.kwargs)
        
class PipelineOnEveryFrame(EventDrivenDataStreamer):
    """
    Its a conditional data source
    """
    def __init__(self, 
                 *generators: StreamerGenerator) -> None:
        super(PipelineOnEveryFrame, self).__init__()
        self.interrupt = False
        self.generators = generators
        self.current_result_queue = Queue()
    
    def __next__(self) -> DataFrame:
        data, is_final = self.current_result_queue.get()
        self.msg_queue.task_done()
        if is_final:
            raise StopIteration
        return data
    
    def __call__(self, data: DataFrame):
        # Create new chain on every frame
        streamers = self.launch_chain()
        
    def launch_chain(self) -> List[DataStreamer]:
        for generator in self.generators:
            streamer = generator.generate()
            self

class GluedStreamer(DataStreamer):
    def __call__(self, data):
        self.commit(data)
        
    def __next__(self):
        data = self.msg_queue.get()
        self.msg_queue.task_done()
        return data
    
class TextPrinter(DataStreamer):
    def __init__(self, color_map: Dict[str, str]):
        super(TextPrinter, self).__init__()
        self.color_map = color_map
    
    def __call__(self, data: DataFrame):
        print(colored(data, color='magenta'), end="")
        self.commit(data)
        
class InterruptibleStreamer(EventDrivenDataStreamer):
    def __init__(self):
        super(InterruptibleStreamer, self).__init__()
        self.interrupt = False
        self.interrupt_lock = threading.Lock()
    
    def event_handlers(self):
        return {
            "start": self.handle_start,
            "interrupt": self.handle_interrupt,
            "end": self.handle_end,
        }
    
    def start(self):
        self.trigger("start")
        
    def interrupt(self):
        self.trigger("interrupt")
        
    def end(self):
        self.trigger("end")
                
    def handle_start(self):
        with self.interrupt_lock:
            self.interrupted = False
            # Clear all pending stuff from the buffer
            # self.clear()
            self.start_time = time.time()

    def handle_interrupt(self):
        with self.interrupt_lock:
            self.interrupted = True
            # Clear all pending stuff from the buffer
            self.clear()
    
    def handle_end(self):
        pass
    
class InterruptCascadeStreamer(InterruptibleStreamer):
    """
    This class cascades the interrupt signal to further such streamers in the pipeline.
    """
    def __init__(self):
        super(InterruptCascadeStreamer, self).__init__()
    
    def handle_start(self):
        with self.interrupt_lock:
            self.interrupted = False
            # Clear all pending stuff from the buffer
            # self.clear()
            self.start_time = time.time()
            self.start()
        
    def handle_interrupt(self):
        with self.interrupt_lock:
            self.interrupted = True
            # Clear all pending stuff from the buffer
            self.clear()
            self.interrupt()
        
    def handle_end(self):
        super(InterruptCascadeStreamer, self).handle_end()
        self.end()

class SpeechToTextStreamer(EventDrivenDataStreamer):
    SPEECH_END_TOKEN = "<$SPEECH_END>"
    def __init__(self):
        super(SpeechToTextStreamer, self).__init__()
        self._on_message_cb = lambda: None
        
    def speech_end(self):
        self.commit(SpeechToTextStreamer.SPEECH_END_TOKEN)

class TextToSpeechStreamer(EventDrivenDataStreamer):
    def __init__(self):
        super(TextToSpeechStreamer, self).__init__()
        
class CancellableText2SpeechStreamer(TextToSpeechStreamer, InterruptCascadeStreamer):
    def __init__(self):
        super(CancellableText2SpeechStreamer, self).__init__()
    
    @abstractmethod
    def __call__(self, text_iterator: Iterator[str]):
        pass
    
    def handle_interrupt(self):
        print(colored("((Interrupting TTS))", "light_cyan"), end="")
        super(CancellableText2SpeechStreamer, self).handle_interrupt()
        print(colored("((Interrupted TTS))", "light_cyan"), end="")
        
    def handle_start(self):
        print(colored("((Starting TTS))", "light_cyan"), end="")
        super(CancellableText2SpeechStreamer, self).handle_start()
        print(colored("((Started TTS))", "light_cyan"), end="")
    
    def close(self):
        super(CancellableText2SpeechStreamer, self).close()
        self.is_closed = True
        self.cancel()