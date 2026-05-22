"""Minimal local evaluator: load Agent from agent_llm, run Env.evaluate, print."""

import concurrent.futures
import traceback

from env import Env
#from agent_llm import Agent
from agent_with_python import Agent


class LocalProxy:
    """Shim that mimics the platform's AgentProxy for local runs.

    Timeouts are enforced with a worker thread. Caveat: `torch.generate`
    can't be interrupted from another thread, so on timeout we *abandon*
    the call (parent moves on, returns `default`) while the worker thread
    keeps running until the process exits. Fine for one-shot eval scripts.
    """

    def __init__(self, agent):
        self._agent = agent
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.last_error = None

    def call(self, method, *args, timeout=None, default=None,
             catch_errors=False, **kwargs):
        self.last_error = None
        fn = getattr(self._agent, method)
        try:
            if timeout is None:
                return fn(*args, **kwargs)
            future = self._executor.submit(fn, *args, **kwargs)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError as e:
                raise TimeoutError(
                    f"agent.{method} exceeded timeout={timeout}s"
                ) from e
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
