from dataclasses import dataclass
from dotenv import load_dotenv
import os

@dataclass
class Config:
    DEEPGRAM_API_KEY: str
    ELEVENLABS_API_KEY: str
    ELEVENLABS_VOICE_ID: str
    OPENAI_API_KEY: str
    
def load_config():
    load_dotenv()
    return Config(
        DEEPGRAM_API_KEY=os.getenv("DEEPGRAM_API_KEY"),
        ELEVENLABS_API_KEY=os.getenv("ELEVENLABS_API_KEY"),
        ELEVENLABS_VOICE_ID=os.getenv("ELEVENLABS_VOICE_ID"),
        OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
    )
    
config = load_config()