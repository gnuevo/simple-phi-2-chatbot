import gradio as gr
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    StoppingCriteria, 
    StoppingCriteriaList, 
    TextIteratorStreamer
    )
from threading import Thread
import re
from typing import List
from details import DETAILS

LANG = "EN" # either EN and ES are valid codes
CHAT_NAME_PATTERN = r'\n.+:$' # matches substrings like `\nUser:` at the end

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Your device is", device)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2", 
    device_map="auto", 
    torch_dtype="auto" if device == "cuda" else torch.float, 
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/phi-2", trust_remote_code=True)

if LANG == "ES":
    HUMAN_NAME = "Usuario"
    BOT_NAME = "Asistente"
else:
    HUMAN_NAME = "User"
    BOT_NAME = "Assistant"

if LANG == "ES":
    CONTEXT = f"El siguiente texto es una conversación amistosa \
    entre {HUMAN_NAME} y {BOT_NAME} en español."
else:
    CONTEXT = f"The following is a friendly conversation \
    between {HUMAN_NAME} and {BOT_NAME} in English."


class StopOnTokens(StoppingCriteria):
    """Stops the model if it produces an 'end of text' token"""
    def __call__(self, input_ids: torch.LongTensor, 
                 scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50256, 198] # <|endoftext|> and EOL
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class StopOnNames(StoppingCriteria):
    """
    Stops the model when it starts hallucinating future turns of the 
    conversation.

    It stops the token generation when we find a token sequence of the form 
    "\n<name>:", for example "\nUser:" or "\nAssistant:".
    """
    EOL_TOKEN = 198
    COLON_TOKEN = 25
    
    def __init__(self, tokenized_names: List[List[int]]):
        self.tokenized_names = tokenized_names
    
    def __call__(self, input_ids: torch.LongTensor, 
                 scores: torch.FloatTensor, **kwargs) -> bool:
        for tokens in self.tokenized_names:
            template = [self.EOL_TOKEN, *tokens, self.COLON_TOKEN]
            if input_ids[0][-len(template):].tolist() == template:
                return True
        return False


def predict(message, history):
    history_transformer_format = history + [[message, ""]]
    stop_on_tokens = StopOnTokens()
    stop_on_names = StopOnNames(
        [tokenizer.encode(HUMAN_NAME), tokenizer.encode(BOT_NAME)])

    messages = "".join(["".join(
        [f"\n{HUMAN_NAME}: "+item[0], f"\n{BOT_NAME}:"+item[1]]
    ) for item in history_transformer_format]).strip()
    messages = CONTEXT + '\n' + messages

    model_inputs = tokenizer([messages], return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, 
                                    skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=256,
        do_sample=True,
        top_p=0.95,
        top_k=1000,
        temperature=1.0,
        num_beams=1,
        stopping_criteria=StoppingCriteriaList([stop_on_tokens, stop_on_names])
        )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    partial_message = ""
    for new_token in streamer:
        partial_message += new_token
        match = re.search(CHAT_NAME_PATTERN, partial_message)
        if match:
            partial_message = partial_message[:-len(match.group())]
        yield partial_message


gr.ChatInterface(predict,**DETAILS[LANG]).queue().launch()