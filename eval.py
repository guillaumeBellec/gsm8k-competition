"""Minimal local evaluator: load Agent from agent_llm, run Env.evaluate, print."""

import traceback

from env import Env
from agent_llm import Agent


class LocalProxy:
    def __init__(self, agent):
        self._agent = agent
        self.last_error = None

    def call(self, method, *args, timeout=None, default=None,
             catch_errors=False, **kwargs):
        self.last_error = None
        try:
            return getattr(self._agent, method)(*args, **kwargs)
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            if default is not None or catch_errors:
                return default
            raise


def main():
    env = Env()
    proxy = LocalProxy(Agent())
    result = env.evaluate([proxy], [{"agent_index": 0}])
    print(result["agent_results"][0])


if __name__ == "__main__":
    main()
