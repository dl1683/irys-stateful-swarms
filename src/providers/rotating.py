from __future__ import annotations

import threading

from ..swarm.models import ModelCaller, ModelResult


class RotatingCaller:
    """Cycles between multiple ModelCallers in a repeating pattern.

    Example: RotatingCaller([fable, flash], pattern=[0, 1, 1])
    produces: fable, flash, flash, fable, flash, flash, ...
    """

    def __init__(self, callers: list[ModelCaller],
                 pattern: list[int] | None = None):
        self._callers = callers
        self._pattern = pattern or list(range(len(callers)))
        self._idx = 0
        self._lock = threading.Lock()

    def complete(self, prompt: str, *, max_tokens: int = 8192,
                 temperature: float = 0.05, json_mode: bool = True) -> ModelResult:
        with self._lock:
            caller_idx = self._pattern[self._idx % len(self._pattern)]
            self._idx += 1
        return self._callers[caller_idx].complete(
            prompt, max_tokens=max_tokens, temperature=temperature,
            json_mode=json_mode,
        )

    def __repr__(self) -> str:
        return (
            f"RotatingCaller("
            f"callers={len(self._callers)}, "
            f"pattern={self._pattern}, "
            f"idx={self._idx})"
        )
