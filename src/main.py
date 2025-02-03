import time
from synapse.chatbot.simple import ChatBot
from synapse.voice_agent import local_voice_bot

if __name__ == "__main__":
    # Local voice bot
    with ChatBot("Qwen/Qwen2.5-14B-Instruct-1M", infer_on_new_words=True) as chatbot:
    # with ChatBot("gpt-4o", infer_on_new_words=True) as chatbot:
        with local_voice_bot(chatbot) as agent:
            time.sleep(300)