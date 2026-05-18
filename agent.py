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


MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"

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

    def answer(self, questions: list[str]) -> list[float]:
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

        replies = []

        for i in range(len(questions)):
            ## TODO: define the reply string.
            ##  reply = self.tokenizer.decode(...).to(self.model.device)
            reply = ...
            replies.append(_parse_final_number(reply))

        return replies
