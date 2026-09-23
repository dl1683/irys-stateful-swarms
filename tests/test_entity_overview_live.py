import pytest

from scripts.run_entity_overview_live import RecordingCaller


class FailingCaller:
    provider_request_count = 2

    def complete(self, *_args, **_kwargs):
        raise RuntimeError("provider unavailable")


def test_recording_caller_preserves_failed_provider_call():
    caller = RecordingCaller(FailingCaller())

    with pytest.raises(RuntimeError, match="provider unavailable"):
        caller.complete("sensitive prompt", max_tokens=8192)

    assert caller.calls == [{
        "prompt": "sensitive prompt", "error": "provider unavailable",
        "requested_output_tokens": 8192, "provider_requests_total": 2,
    }]


def test_recording_caller_caps_logical_calls_not_provider_retries():
    caller = RecordingCaller(FailingCaller(), max_calls=1)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        caller.complete("first")
    with pytest.raises(RuntimeError, match="logical model-call cap reached"):
        caller.complete("second")
