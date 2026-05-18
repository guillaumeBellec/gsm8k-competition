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


HF_CACHE_DIR = os.environ.get("HF_HOME", "/opt/hf_cache")


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

SYSTEM_PROMPT = """You are a careful math tutor. Solve the problem step by step, then write the final numeric answer on a new line prefixed by '####'.

Here are some worked examples:

Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
#### 72

Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10

Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
Answer: In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.
Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30.
This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.
#### 5

Now solve the next problem in the same format."""


class Agent:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, cache_dir=HF_CACHE_DIR,
        )
        # Left padding is required for batched causal generation: every
        # sample's "next token" position must align at the right edge.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=HF_CACHE_DIR,
        )
        self.model.eval()

    def answer(self, questions: list[str]) -> list[float]:
        prompts = []

        for q in questions:
            ## TODO: generate the prompt with toknizer.apply_chat_template(...)
            ##  include a SYSTEM_PROMPT and the appropriate question.
            prompt = ...
            prompts.append(prompt)

        ## TODO: define the inputs tokens
        ##  inputs = self.tokeizer(...).to(self.model.device)

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
            reply = ///
            replies.append(_parse_final_number(reply))

        return replies
