from typing import Any
import pyaudio

from synapse.pipeline.sinks import DataSink

class LocalSpeaker(DataSink):
    def __init__(self, format=pyaudio.paInt16, channels=1, sample_rate=16000, frames_per_buffer=1024) -> None:
        super(LocalSpeaker, self).__init__()
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=format, channels=channels, rate=sample_rate, output=True, frames_per_buffer=frames_per_buffer)
    
    def __call__(self, data: Any):
        self.stream.write(data)
        
    def close(self):
        super(LocalSpeaker, self).close()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()