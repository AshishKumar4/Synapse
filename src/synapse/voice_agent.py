import pyaudio
import sys

from synapse.utils import logger
from synapse.processors import AITranscriptIterator, Stream2Sentence
from synapse.chatbot.simple import ChatBot
from synapse.pipeline.sources import LocalMicrophone
from synapse.pipeline.sinks import LocalSpeaker
from synapse.stt.deepgram import DeepgramSTTStreamer
from synapse.tts.kokoro import KokoroTTS

class LocalVoiceAgent:
    def __init__(self, chatbot:ChatBot,
                 channels=1 if sys.platform == 'darwin' else 2, sample_rate=24000, format=pyaudio.paInt16, frames_per_buffer=pyaudio.paFramesPerBufferUnspecified):
        mic = LocalMicrophone(format=format, channels=channels, sample_rate=sample_rate, frames_per_buffer=frames_per_buffer)
        stt = DeepgramSTTStreamer(channels, sample_rate)
        # ai_iter = AITranscriptIterator()
        s2s = Stream2Sentence()
        tts = KokoroTTS(sample_rate=sample_rate)
        speaker = LocalSpeaker(format=format, channels=1, sample_rate=sample_rate, frames_per_buffer=frames_per_buffer)
        
        stt.read_from(mic)
        chatbot.read_from(stt)
        s2s.read_from(chatbot)
        tts.read_from(s2s)
        tts.write_to(speaker)
        
        self.mic = mic
        self.stt = stt
        self.s2s = s2s
        self.tts = tts
        self.speaker = speaker
        
        # Start PyAudio stream
        logger.info("Recording...")
    
    def close(self):
        logger.info("Recording finished.")
        self.mic.close()
        self.stt.close()
        # self.ai_iter.close()
        self.tts.close()
        self.speaker.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        
def local_voice_bot(chatbot:ChatBot=None):
    agent = LocalVoiceAgent(chatbot=chatbot)
    return agent