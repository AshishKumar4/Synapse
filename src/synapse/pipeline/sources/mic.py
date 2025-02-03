from queue import Queue
from typing import Any
import pyaudio

from synapse.pipeline.sources import DataSource

class LocalMicrophone(DataSource):
    def __init__(self, format=pyaudio.paInt16, channels=1, sample_rate=16000, frames_per_buffer=1024) -> None:
        super(LocalMicrophone, self).__init__()
        self.p = pyaudio.PyAudio()
        # Callback function to send audio data to Deepgram
        self.queue = Queue()
        def callback(in_data, frame_count, time_info, status):
            self.queue.put((in_data, False))
            return (in_data, pyaudio.paContinue)
        self.stream = self.p.open(format=format, channels=channels, rate=sample_rate, input=True, frames_per_buffer=frames_per_buffer, stream_callback=callback)

    def __next__(self) -> Any:
        data, is_final = self.queue.get()
        self.queue.task_done()
        if is_final:
            raise StopIteration
        return data
    
    def close(self):
        super(LocalMicrophone, self).close()
        self.queue.put((None, True))
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
     