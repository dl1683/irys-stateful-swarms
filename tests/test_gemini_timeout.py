import time

from src.providers.gemini import GeminiCaller
from src.providers.gemini import _RequestLimiter


class HangingModels:
    calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        time.sleep(5)


class HangingClient:
    models = HangingModels()


def test_gemini_watchdog_times_out_hanging_sdk_call():
    caller = GeminiCaller.__new__(GeminiCaller)
    caller.client = HangingClient()
    caller.model = "fake-gemini"
    caller.timeout_s = 0.01

    started = time.perf_counter()
    try:
        caller._generate_content_with_timeout("prompt", object())
    except TimeoutError as exc:
        elapsed = time.perf_counter() - started
        assert elapsed < 1
        assert "watchdog timeout" in str(exc)
    else:
        raise AssertionError("Expected watchdog timeout")


def test_gemini_complete_does_not_retry_timeout_by_default():
    models = HangingModels()

    class Client:
        pass

    client = Client()
    client.models = models
    caller = GeminiCaller.__new__(GeminiCaller)
    caller.client = client
    caller.model = "fake-gemini"
    caller.timeout_s = 0.01
    caller.timeout_attempts = 1

    try:
        caller.complete("prompt", max_tokens=8, json_mode=False)
    except RuntimeError as exc:
        assert "timed out after 1 attempts" in str(exc)
    else:
        raise AssertionError("Expected complete() timeout")

    assert models.calls == 1


def test_gemini_complete_retries_deadline_exceeded(monkeypatch):
    class Usage:
        prompt_token_count = 3
        candidates_token_count = 4

    class Response:
        text = '{"ok": true}'
        usage_metadata = Usage()

    class DeadlineThenSuccessModels:
        def __init__(self):
            self.calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("504 DEADLINE_EXCEEDED")
            return Response()

    class Client:
        pass

    models = DeadlineThenSuccessModels()
    client = Client()
    client.models = models
    caller = GeminiCaller.__new__(GeminiCaller)
    caller.client = client
    caller.model = "fake-gemini"
    caller.timeout_s = 1
    caller.timeout_attempts = 1
    monkeypatch.setattr("src.providers.gemini.time.sleep", lambda seconds: None)

    result = caller.complete("prompt", max_tokens=8, json_mode=True)

    assert models.calls == 2
    assert result.text == '{"ok": true}'
    assert result.tokens_total == 7


def test_request_limiter_enforces_hard_cap(monkeypatch):
    limiter = _RequestLimiter(60_000, max_requests=2)
    monkeypatch.setattr("src.providers.gemini.time.sleep", lambda seconds: None)
    limiter.wait()
    limiter.wait()
    try:
        limiter.wait()
    except RuntimeError as exc:
        assert "cap reached" in str(exc)
    else:
        raise AssertionError("Expected request cap")
