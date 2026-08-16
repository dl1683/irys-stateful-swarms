import pytest

from src.providers.gemini import _RequestLimiter


def test_request_limiter_uses_the_configured_requests_per_minute():
    limiter = _RequestLimiter(10)

    assert limiter.interval == pytest.approx(6.0)


def test_request_limiter_rejects_non_positive_rate():
    with pytest.raises(ValueError, match="requests_per_minute must be positive"):
        _RequestLimiter(0)
