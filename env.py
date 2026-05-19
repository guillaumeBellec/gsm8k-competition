"""GSM8K Competition Environment (standalone).

At test time we draw K disjoint subsets of 10 questions each. Every subset
contains 5 easy questions (drawn from items with difficulty "easy" or
"medium") and 5 hard questions, shuffled. Each subset is sent to the agent
as a single batch via `agent.answer(questions: list[str]) -> list[float]`:
the agent must return one numeric answer per question. A subset has
BATCH_TIMEOUT seconds to complete; if it fails or times out, no further
subsets are sent.

Questions and gold answers come from `question_set.json` next to this file.

For each agent we return:
  - score:               total correct across all delivered subsets
  - format_errors:       agent replies that were not coercible to float
  - questions_received:  number of questions actually sent
"""

import json
import math
import os
import random
import re


DATA_PATH = os.path.join(os.path.dirname(__file__), "question_set_private.json")
K = 5                       # number of subsets per agent
EASY_PER_SUBSET = 5         # easy/medium picks per subset
HARD_PER_SUBSET = 5         # hard picks per subset
BATCH_TIMEOUT = 60.0        # seconds per subset
SEED = 0                    # fixed sample is the same for every agent

_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)*")


def _gold_from_answer(answer_field):
    """Parse the gold number from a GSM8K answer string ending in `#### N`."""
    if answer_field is None or "####" not in str(answer_field):
        return None
    tail = str(answer_field).rsplit("####", 1)[-1].strip()
    matches = _NUMBER_RE.findall(tail)
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def _to_float(x):
    """Coerce an agent reply to float, or return None if it cannot be parsed."""
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return None if (isinstance(x, float) and math.isnan(x)) else float(x)
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


class Env:
    def __init__(self):
        try:
            with open(DATA_PATH) as f:
                items = json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Question set not found at {DATA_PATH!r}. "
                f"Place question_set.json next to env.py."
            ) from e
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Question set at {DATA_PATH!r} is not valid JSON: {e}"
            ) from e

        if not isinstance(items, list):
            raise TypeError(
                f"Question set at {DATA_PATH!r} must be a JSON list; "
                f"got {type(items).__name__}."
            )

        easy = [it for it in items if it.get("difficulty", "easy") != "hard"]
        hard = [it for it in items if it.get("difficulty") == "hard"]
        need_easy = K * EASY_PER_SUBSET
        need_hard = K * HARD_PER_SUBSET
        if len(easy) < need_easy:
            raise ValueError(
                f"Question set has {len(easy)} easy/medium items; "
                f"need at least K * EASY_PER_SUBSET = {need_easy}."
            )
        if len(hard) < need_hard:
            raise ValueError(
                f"Question set has {len(hard)} hard items; "
                f"need at least K * HARD_PER_SUBSET = {need_hard}."
            )

        rng = random.Random(SEED)
        rng.shuffle(easy)
        rng.shuffle(hard)

        self.subsets = []
        for k in range(K):
            chunk = (
                easy[k * EASY_PER_SUBSET:(k + 1) * EASY_PER_SUBSET]
                + hard[k * HARD_PER_SUBSET:(k + 1) * HARD_PER_SUBSET]
            )
            rng.shuffle(chunk)
            self.subsets.append(chunk)

        # Flat view for callers that want all questions in delivery order.
        self.questions = [it["question"] for s in self.subsets for it in s]
        self.gold = [_gold_from_answer(it["answer"])
                     for s in self.subsets for it in s]

    def evaluate(self, agents, agent_infos):
        results = []
        for i, agent in enumerate(agents):
            correct = 0
            received = 0
            format_errors = 0
            first_error = None

            for subset in self.subsets:
                q_batch = [it["question"] for it in subset]
                g_batch = [_gold_from_answer(it["answer"]) for it in subset]

                replies = agent.call(
                    "answer", q_batch,
                    timeout=BATCH_TIMEOUT,
                    catch_errors=True,
                )
                
                if agent.last_error is not None:
                    # Proxy already typed the exception (e.g. "TimeoutError: ...");
                    # surface verbatim and stop sending more subsets.
                    first_error = first_error or agent.last_error
                    break
                if not isinstance(replies, list):
                    first_error = first_error or (
                        f"TypeError: agent.answer must return a list of floats; "
                        f"got {type(replies).__name__}."
                    )
                    break
                if len(replies) != len(q_batch):
                    first_error = first_error or (
                        f"ValueError: agent.answer returned {len(replies)} replies "
                        f"for a batch of {len(q_batch)} questions."
                    )
                    break

                for reply, gold in zip(replies, g_batch):
                    received += 1
                    pred = _to_float(reply)
                    if pred is None:
                        format_errors += 1
                        continue
                    if gold is not None and abs(pred - gold) < 1e-6:
                        correct += 1

            total = K * (EASY_PER_SUBSET + HARD_PER_SUBSET)
            entry = {
                "agent_index": i,
                "score": correct / K,
                "format_errors": format_errors,
                "questions_received": received,
                "info_message": (
                    f"{correct} correct, {format_errors} formatting errors, "
                    f"{received}/{total} received"
                ),
            }
            if first_error is not None:
                entry["agent_code_error_message"] = first_error
                entry["is_agent_code_error"] = (correct == 0)
            results.append(entry)

        return {"agent_results": results}
