# GSM8K Competition

[GSM8K](https://github.com/openai/grade-school-math) is a benchmark of ~8.5k grade-school-level math word problems (7.5k train / 1.3k test) released by OpenAI in 2021. Each problem requires 2–8 steps of natural-language reasoning over basic arithmetic. Gold answers are step-by-step with the final number on its own line after `####`. Example:

> **Q:** Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
>
> **A:** Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips altogether in April and May. `#### 72`

In this competition your `Agent` is evaluated against a private question set in the same format. At test time `Env` draws **K=5 disjoint subsets of 10 questions** each (5 easy + 5 hard, shuffled with a fixed seed). Each subset is sent to your agent as a single batch via `agent.answer(questions: list[str]) -> list[float]` — one numeric answer per question. Each subset has a **60-second timeout**; if a batch fails or times out, no further subsets are sent. Score = number of replies matching the gold answer to within `1e-6`, summed across all delivered subsets. Format errors (replies that are not coercible to float) are reported separately. See `agent_template.py` for the minimal interface and `Agent.py` for a SmolLM3-3B baseline with batched generation and 3-shot prompting.

**HF model cache.** The evaluation platform pre-populates a Hugging Face cache so popular checkpoints don't need to be re-downloaded inside your 60-second batch budget. Point HF at it by passing `cache_dir=os.environ.get("HF_HOME", "/opt/hf_cache")` to `AutoModelForCausalLM.from_pretrained` and `AutoTokenizer.from_pretrained` (the local default `/opt/hf_cache` is what the platform mounts; the env var fallback lets the same code work on your laptop).

# Competition link:  

Enroll with this ml-arena competition ID, and submit your agent on this link:  
[https://ml-arena.com/enroll/05c13e15008e4167b6f91fe67abff6f2](https://ml-arena.com/enroll/05c13e15008e4167b6f91fe67abff6f2)   

Available models on the platform are:  

  ┌────────────────────────────┬───────┐
  │           Model            │ Size  │
  ├────────────────────────────┼───────┤
  │ HuggingFaceTB/SmolLM3-3B   │ 5.8 G │
  ├────────────────────────────┼───────┤
  │ Qwen/Qwen2.5-0.5B-Instruct │ 954 M │
  ├────────────────────────────┼───────┤
  │ Qwen/Qwen3-1.7B            │ 3.8 G │
  └────────────────────────────┴───────┘


# External infos:  

HF LLM course: https://huggingface.co/learn/llm-course/chapter1/1  
HF Agent course: https://huggingface.co/learn/agents-course/unit0/introduction  