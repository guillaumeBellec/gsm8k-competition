"""
GSM8K Competition Agent — starter template.

Implement `answer(questions)`: it receives a list of question strings (a
batch from Env, up to 10 at a time) and must return a list of floats of
the same length — one numeric answer per question. Return `float('nan')`
(or any non-matching number) if your pipeline fails to produce one;
Env counts those as format_errors.

The default implementation loads HuggingFaceTB/SmolLM3-3B, prompts it to
solve each problem step by step ending with `#### N`, and parses N out
of the model's text. You are free to change the model, the prompt, the
decoding strategy, or replace the whole pipeline.
"""

import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)*")


def _parse_final_number(text):
    """Pull the number after the final `####` marker, or return NaN."""
    if "####" not in text:
        return float("nan")
    tail = text.rsplit("####", 1)[-1].strip()
    matches = _NUMBER_RE.findall(tail)
    if not matches:
        return float("nan")
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return float("nan")

#Choose the model you want to use. SmolLM3-3B is a good default for evaluation, but you can experiment with other models if you like. Just make sure to update the prompt and parsing logic in answer() if your model's output format differs from the expected "step by step solution ending with '#### N'".

MODEL_NAME = "HuggingFaceTB/SmolLM3-3B" #"Qwen/Qwen2.5-0.5B-Instruct" #"Qwen/Qwen3-1.7B"

SYSTEM_PROMPT = """You are a careful math tutor. Solve the problem step by step, then write the final numeric answer on a new line prefixed by '####'."""


class Agent:
    def __init__(self):

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

  
        # Trigger CUDA kernel compilation now, before any timeout window opens.
        if torch.cuda.is_available():
            dummy = self.tokenizer("warm-up", return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                self.model.generate(**dummy, max_new_tokens=1,
                                    pad_token_id=self.tokenizer.eos_token_id)
            torch.cuda.synchronize()

    def answer(self, questions: list[str]) -> tuple[list[float], list[str]]:
        prompts = []

        for q in questions:
            ## TODO: generate the prompt with tokenizer.apply_chat_template(...)
            ##  include a SYSTEM_PROMPT and the appropriate question.
            prompt = ...
            prompts.append(prompt)

        ## TODO: define the inputs tokens
        ##  inputs = self.tokenizer(...).to(self.model.device)

        inputs = ...
        # We move the input tokens to the GPU
        inputs = inputs.to(self.model.device)
        prefix_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            ## TODO: run the LLM
            ##  outputs = self.model.generate(...).
            pass

        # The function answer() return two lists : 
            # A list of floats, one per question, anwering the question or NaN if parsing failed (This is used for evaluation)
            # A list of strings, one per question, containing the full text output of the model (This is only used for renderring, just because it's fancy)

        floats = []
        thinking_traces = []

        for i in range(len(questions)):
            ## TODO: define the reply string.
            ##  reply = self.tokenizer.decode(...).to(self.model.device)
            reply = ...
            thinking_traces.append(reply)
            floats.append(_parse_final_number(reply))

        return floats, thinking_traces
