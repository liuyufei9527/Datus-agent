"""Tests for datus/storage/document/fetcher/rate_limiter.py"""

import time

import pytest

from datus.storage.document.fetcher.rate_limiter import (
    RateLimitConfig,
    RateLimitState,
    RateLimiter,
    get_rate_limiter,
)


class TestRateLimitConfig:
    def test_defaults(self):
        config = RateLimitConfig()
        assert config.requests_per_hour == 60
        assert config.min_interval == 0.5
        assert config.burst_size == 10

    def test_custom_values(self):
        config = RateLimitConfig(requests_per_hour=5000, min_interval=0.1, burst_size=50)
        assert config.requests_per_hour == 5000
        assert config.min_interval == 0.1
        assert config.burst_size == 50


class TestRateLimitState:
    def test_defaults(self):
        state = RateLimitState()
        assert state.request_count == 0
        assert state.last_request == 0.0
        assert state.remaining is None
        assert state.reset_time is None


class TestRateLimiter:
    def test_init(self):
        limiter = RateLimiter()
        assert "api.github.com" in limiter._configs
        assert "github.com" in limiter._configs
        assert "default" in limiter._configs

    def test_configure(self):
        limiter = RateLimiter()
        config = RateLimitConfig(requests_per_hour=100, min_interval=1.0)
        limiter.configure("example.com", config)
        assert limiter._configs["example.com"].requests_per_hour == 100

    def test_configure_github_authenticated(self):
        limiter = RateLimiter()
        limiter.configure_github_authenticated(5000)
        config = limiter._configs["api.github.com"]
        assert config.requests_per_hour == 5000
        assert config.min_interval == 0.05

    def test_wait_returns_float(self):
        limiter = RateLimiter()
        result = limiter.wait("example.com")
        assert isinstance(result, float)

    def test_wait_second_request_respects_interval(self):
        limiter = RateLimiter()
        limiter.configure("test.com", RateLimitConfig(min_interval=0.01, requests_per_hour=3600))
        limiter.wait("test.com")
        start = time.time()
        limiter.wait("test.com")
        elapsed = time.time() - start
        # Should have waited at least the min_interval
        assert elapsed >= 0.005  # Small margin for timing

    def test_update_from_headers_remaining(self):
        limiter = RateLimiter()
        headers = {"x-ratelimit-remaining": "50"}
        limiter.wait("api.github.com")  # Create state
        limiter.update_from_headers("api.github.com", headers)
        assert limiter.get_remaining("api.github.com") == 50

    def test_update_from_headers_reset(self):
        limiter = RateLimiter()
        reset_time = str(time.time() + 3600)
        headers = {"x-ratelimit-reset": reset_time}
        limiter.wait("api.github.com")
        limiter.update_from_headers("api.github.com", headers)
        state = limiter._states["api.github.com"]
        assert state.reset_time is not None

    def test_update_from_headers_invalid_values(self):
        limiter = RateLimiter()
        headers = {
            "x-ratelimit-remaining": "invalid",
            "x-ratelimit-reset": "invalid",
        }
        limiter.wait("api.github.com")
        limiter.update_from_headers("api.github.com", headers)
        state = limiter._states["api.github.com"]
        assert state.remaining is None

    def test_get_remaining_unknown_domain(self):
        limiter = RateLimiter()
        assert limiter.get_remaining("unknown.com") is None


class TestCalculateWaitTime:
    def test_no_wait_for_first_request(self):
        limiter = RateLimiter()
        config = RateLimitConfig()
        state = RateLimitState()
        wait = limiter._calculate_wait_time(config, state)
        assert wait == 0.0

    def test_min_interval_wait(self):
        limiter = RateLimiter()
        config = RateLimitConfig(min_interval=1.0)
        state = RateLimitState(last_request=time.time())
        wait = limiter._calculate_wait_time(config, state)
        assert wait > 0

    def test_hourly_limit_reached(self):
        limiter = RateLimiter()
        config = RateLimitConfig(requests_per_hour=10)
        state = RateLimitState(request_count=10, window_start=time.time())
        wait = limiter._calculate_wait_time(config, state)
        assert wait > 0

    def test_api_remaining_zero(self):
        limiter = RateLimiter()
        config = RateLimitConfig()
        state = RateLimitState(remaining=0, reset_time=time.time() + 60)
        wait = limiter._calculate_wait_time(config, state)
        assert wait > 0

    def test_burst_limit_delay(self):
        limiter = RateLimiter()
        config = RateLimitConfig(burst_size=10)
        state = RateLimitState(request_count=10)
        wait = limiter._calculate_wait_time(config, state)
        assert wait >= 1.0

    def test_max_wait_cap(self):
        limiter = RateLimiter()
        config = RateLimitConfig(requests_per_hour=1)
        state = RateLimitState(request_count=1, window_start=time.time())
        wait = limiter._calculate_wait_time(config, state, max_wait=5.0)
        assert wait <= 5.0


class TestGetRateLimiter:
    def test_returns_limiter(self):
        limiter = get_rate_limiter()
        assert isinstance(limiter, RateLimiter)

    def test_returns_same_instance(self):
        limiter1 = get_rate_limiter()
        limiter2 = get_rate_limiter()
        assert limiter1 is limiter2
