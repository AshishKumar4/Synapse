import time
from typing import Any, Callable, Dict
from termcolor import colored
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from .states import GlobalTranscript
from .engines.generator import LLMGenerator

from synapse.utils import DataFrame, AI_SPEECH_END_TOKEN
from synapse.pipeline.streamers.common import InterruptCascadeStreamer, SpeechToTextStreamer

class ChatBot(InterruptCascadeStreamer):
    def __init__(
        self, 
        model: PreTrainedModel | str, 
        tokenizer: PreTrainedTokenizerFast = None, 
        infer_on_new_words=True, 
        bot_name="Ratchel", 
        human_names=["Ashish"],
        transcript_log_path=None
    ) -> None:
        super(ChatBot, self).__init__()
        self.llm_generator = LLMGenerator(model, tokenizer)
        
        self.initial_prompt = self.get_default_prompt()
        
        self._on_ai_speech_start_cb = lambda: self.handle_start()
        self._on_ai_speech_end_cb = lambda: self.handle_end()
        self._on_ai_words_cb = lambda word: self.commit(word)
        
        self.bot_name = bot_name
        self.human_names = human_names
        
        self.infer_on_new_words = infer_on_new_words
        
        self.global_transcript = GlobalTranscript(transcript_log_path)
        def speaker_change_handler(old_speaker, old_speaker_type, new_speaker, new_speaker_type, time):
            print(colored(f'<@@gen speaker change {old_speaker}->{new_speaker} at {time}>', "yellow"), end='')
            # If Old speaker was a bot, call self.cancel()
            if old_speaker == self.bot_name:
                self.handle_interrupt()
            
        self.global_transcript.on("speaker_change", speaker_change_handler)
        
    def commit(self, data: DataFrame):
        self.global_transcript((data, self.bot_name, time.time(), True))
        super().commit(data)
        
    def __call__(self, data) -> Any:
        if data == SpeechToTextStreamer.SPEECH_END_TOKEN:
            print(colored(f'<@@user-speech end>', "red"), end='')
            if not self.infer_on_new_words:
                self.__generate_response__()
            self.__start_flushing__()
            return
        words, speaker, arrival_time = data
        if len(words) == 0:
            print(colored(f'<@@gen skipped, no words>', "red"), end='')
            return
        
        text = ' '.join(words)
        if text.strip() == "":
            return
        
        self.llm_generator.cancel_current_run()
        
        self.global_transcript((text, speaker, arrival_time, False))
        if self.infer_on_new_words:
            self.__generate_response__(text)
        pass
        
    def __generate_response__(self, words=[]):
        print(colored(f'<@@gen queuing {words}>', "yellow"), end='')
        self.llm_generator.generate(self.get_full_context)
        
    def __start_flushing__(self):
        current_run = self.llm_generator.get_current_run()
        if current_run is not None and not current_run.is_cancelled() and current_run.flush_future is None:
            current_run.flush(on_start_callback=self._on_ai_speech_start_cb, 
                              on_end_callback=self._on_ai_speech_end_cb, 
                              on_word_callback=self._on_ai_words_cb)
        else:
            # start a new run
            print(colored(f'<@@re-gen>', "red"), end='')
            self.llm_generator.generate(self.get_full_context, 
                                        on_run_start=lambda run: run.flush(on_start_callback=self._on_ai_speech_start_cb, 
                                                                           on_end_callback=self._on_ai_speech_end_cb, 
                                                                           on_word_callback=self._on_ai_words_cb))
        
    def wait_for_flush(self):
        self.llm_generator.wait_for_flush()
    
    def close(self):
        self.llm_generator.exit()
        return super().close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        
    # def predict_next_few_tokens(self):
    #     if self.llm_generator.tokenizer is None:
    #         raise Exception("Tokenizer not set")
    #     context = self.get_full_context()
    #     batch = [
    #         context,
    #     ]
    #     model_inputs = self.llm_generator.tokenizer(batch, return_tensors="pt", padding=True).to("cuda")
    #     # First try generate up until bob's response
    #     generated_ids = self.llm_generator.model.generate(**model_inputs, penalty_alpha=0.6, top_k=5, max_new_tokens=5, stop_strings=["Ratchel:"], tokenizer=self.tokenizer)
    #     output = self.llm_generator.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0][len(context):]
    #     return output
    
    def get_default_prompt(self) -> str:
    #     return """
    # This is a transcript of a natural voice conversation between 2 indivisuals: Ashish and his colleague Ratchel. Ashish is a machine learning engineer who is working on designing Voice bot with advanced intelligence with his colleage Ratchel.
    # Ashish has a poor microphone and thus his transcripts are not generated properly. The transcripts were generated using a voice to text model which tries to guess the spoken text in realtime, but 
    # if after a certain time it thinks that some part of the text was mispredicted, it would add a text string of type "<!{' '.join([i.punctuated_word for i in mispredicted_words]), iter={correction_position}}>".
    # Ashish is a man of few words and his transcripts are very very short. The conversation is as follows:
    # ------------------------------------------------------------------------------------------------------------------------------
    # """
        return """
    You are Ratchel, a super intelligent AI Assistant made by Ashish. The User is Ashish.
    The User interacts with you via voice, which is converted to text and provided to you. But the speech to text conversion may be imperfect, and the way it works is as follows:
    -   The text were generated using a voice to text model which tries to guess the spoken text in realtime, but if after a certain time it thinks that some part of the text was mispredicted, 
        it would add a text string of type f`<!{' '.join([i.punctuated_word for i in mispredicted_words]), iter={correction_position}}>` into the transcript. For example: 
            Hi.How are you?<!Hi., iter=0>
        Here, the actual text should have been "How are you?" and 'Hi.' was the mispredicted text. Another Example:
            ThePlease stop.<!The, iter=0>
        Here, the actual text should have been "Please stop." and 'The' was the mispredicted text.
    -   Thus you need to account for such errors and still respond accurately to the user.
    **INSTRUCTIONS:**
    - Your task is to converse with him while keeping in mind that the text you see is not always accurate. Thus you need to understand the context and respond accordingly.
    - Your responses need to be interesting, enthusiastic, accurate and engaging. Please do not use any emojis or emoticons in your responses.
    - Your output needs to be in the form of text that can be spoken out by a text to speech model.
    - Do not use lists, bullet points or any other formatting in your responses directly. Formatted text cant be spoken out by the text to speech model.
    - If you were interrupted in the middle of a response by the user, Please stop immediately and DO NOT respond and wait for the user to finish speaking. You may only say things like 'ahmm', 'okay', 'got it', 'go on' etc. in such cases.
    - To provide a formatted text or code response, please encapsulate the text in a code block using triple backticks as shown below:
    This is normal text
    ```
    This is a code block or a formatted text with bullet points or lists or any other formatting or markdown
    ```
    """
    
    def get_full_context(self) -> str:
        # First fetch the global transcript
        # self.global_transcript.sync() # sync the transcript
        transcript = self.global_transcript.get_transcript()
        initial_prompt = {
            "role": "system",
            "content": self.initial_prompt
        }
        # initial_prompt = self.initial_prompt
        context = [initial_prompt] + transcript
        return context