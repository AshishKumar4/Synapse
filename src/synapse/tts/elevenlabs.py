from queue import Queue
from typing import Any, Dict, Callable
import threading
from termcolor import colored
from elevenlabs import stream as elevenlabs_stream
import websockets.sync.client
import json
import time
import base64

from synapse.pipeline.streamers.common import CancellableText2SpeechStreamer
from synapse.processors import AITranscriptIterator
from synapse.utils import logger
from synapse.config import config

class ElevenLabsTTS_WS(CancellableText2SpeechStreamer):
    def __init__(self, sample_rate=16000, voice_id=config.ELEVENLABS_VOICE_ID, output_format="pcm_16000"):
        """Send text to ElevenLabs API and stream the returned audio."""
        self.uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id=eleven_turbo_v2&output_format={output_format}&optimize_streaming_latency=4"
        self.websocket = websockets.sync.client.connect(self.uri)
        super(ElevenLabsTTS_WS, self).__init__()
        
        def keep_alive():
            self.websocket.send(json.dumps({
                "text": " ",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
                "xi_api_key": config.ELEVENLABS_API_KEY,
                 "generation_config": {
                    "chunk_length_schedule": [50, 160, 250, 290]
                },
            }))
            while not self.is_closed:
                try:
                    self.websocket.send(json.dumps({"text": " "}))
                except Exception as e:
                    print(f"Error in keep alive: {e}")
                    self.websocket = websockets.sync.client.connect(self.uri)
                # self.lock.release()
                time.sleep(15)
                
        self.keep_alive_thread = threading.Thread(target=keep_alive)
        self.keep_alive_thread.start()
        self.iter = 0
        
        def audio_generator():
            while True:
                frame, is_final = self.__fetch_frames__()
                if is_final:
                    break
                self.commit(frame)
        self.read_from_thread = threading.Thread(target=audio_generator)
        self.read_from_thread.start()
        
    def __call__(self, data: str):
        text = f'{data} '
        if data == AITranscriptIterator.SPEECH_END_TOKEN:
            print(colored(f"((Sending end))", "light_yellow"), end='')
            self.iter = 0   # Reset the counter
            req = {"flush": True}
        else:
            # print(f"Sending text: {text}")
            req = {"text": text, "try_trigger_generation": True}
            if self.iter == 0:
                req["flush"] = True

        self.websocket.send(json.dumps(req))
        self.iter += 1
    
    def __fetch_frames__(self) -> tuple[bytes, bool]:
        try:
            print(colored(f"((Waiting for audio at {time.time()-self.start_time}))", "light_yellow"), end='')
            message = self.websocket.recv()
            data = json.loads(message)
            if data.get("audio"):
                print(colored(f"((got audio at {time.time()-self.start_time}))", "light_yellow"), end='')
                output = base64.b64decode(data["audio"])
                if output != None and len(output) > 0 and not self.interrupted:
                    return output, False
                else:
                    return None, False
            elif data.get('isFinal'):
                print(colored(f"((got final))", "light_yellow"), end='')
                return None, True
        except Exception as e:
            logger.error(f"<<@@ERROR {e}>>")
            return None, True
        
    def close(self):
        self.websocket.close()
        self.keep_alive_thread.join(timeout=2.0)
        super().close()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        
