"""Unit tests for the in-process TTL cache (backend/app/services/ttl_cache.py).

The cache exists because three endpoints recomputed >65 s of artifact
aggregation per request on the deployed t3.micro. These tests cover the
helper itself only — no FastAPI, no DB, no research imports — and inject a
fake clock instead of sleeping, so expiry is tested deterministically.
"""

from __future__ import annotations

import threading

import pytest

from backend.app.services.ttl_cache import TTLCache, cached


class FakeClock:
    """A controllable monotonic clock."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# ---------------------------------------------------------------------------
# TTLCache.get_or_compute
# ---------------------------------------------------------------------------


def test_computes_once_within_ttl():
    clock = FakeClock()
    cache = TTLCache(clock=clock)
    calls = []

    def compute():
        calls.append(1)
        return "payload"

    assert cache.get_or_compute("k", 300.0, compute) == "payload"
    clock.advance(299.0)
    assert cache.get_or_compute("k", 300.0, compute) == "payload"
    assert len(calls) == 1


def test_recomputes_after_expiry():
    clock = FakeClock()
    cache = TTLCache(clock=clock)
    calls = []

    def compute():
        calls.append(1)
        return len(calls)

    assert cache.get_or_compute("k", 60.0, compute) == 1
    clock.advance(60.0)  # exactly at TTL counts as expired (strict '<')
    assert cache.get_or_compute("k", 60.0, compute) == 2
    assert len(calls) == 2


def test_keys_are_independent():
    clock = FakeClock()
    cache = TTLCache(clock=clock)

    assert cache.get_or_compute(("gate", 0), 300.0, lambda: "a") == "a"
    assert cache.get_or_compute(("gate", 5), 300.0, lambda: "b") == "b"
    # Each key keeps its own value.
    assert cache.get_or_compute(("gate", 0), 300.0, lambda: "WRONG") == "a"
    assert cache.get_or_compute(("gate", 5), 300.0, lambda: "WRONG") == "b"


def test_cache_result_false_is_returned_but_not_stored():
    """Degraded/fallback payloads must not outlive the condition causing them."""
    clock = FakeClock()
    cache = TTLCache(clock=clock)
    results = iter(["unavailable", "healthy"])

    def compute():
        return next(results)

    is_healthy = lambda value: value == "healthy"  # noqa: E731
    assert cache.get_or_compute("k", 300.0, compute, cache_result=is_healthy) == "unavailable"
    # No time passes: the bad result was not cached, so we recompute at once...
    assert cache.get_or_compute("k", 300.0, compute, cache_result=is_healthy) == "healthy"
    # ...and the good result IS cached.
    assert cache.get_or_compute("k", 300.0, compute, cache_result=is_healthy) == "healthy"


def test_exceptions_propagate_and_are_not_cached():
    clock = FakeClock()
    cache = TTLCache(clock=clock)
    attempts = []

    def compute():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("boom")
        return "ok"

    with pytest.raises(RuntimeError):
        cache.get_or_compute("k", 300.0, compute)
    assert cache.get_or_compute("k", 300.0, compute) == "ok"
    assert len(attempts) == 2


def test_invalidate_forces_recompute():
    clock = FakeClock()
    cache = TTLCache(clock=clock)
    calls = []

    def compute():
        calls.append(1)
        return len(calls)

    assert cache.get_or_compute("k", 300.0, compute) == 1
    cache.invalidate("k")
    assert cache.get_or_compute("k", 300.0, compute) == 2
    cache.invalidate()  # clear-all form
    assert cache.get_or_compute("k", 300.0, compute) == 3


def test_concurrent_callers_share_one_computation():
    """Single-flight: N concurrent callers must not trigger N computations."""
    cache = TTLCache()
    calls = []
    results = []

    def compute():
        calls.append(1)
        return "shared"

    def worker():
        results.append(cache.get_or_compute("k", 300.0, compute))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == ["shared"] * 8
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# @cached decorator
# ---------------------------------------------------------------------------


def test_decorator_caches_per_arguments_and_expires():
    clock = FakeClock()
    calls = []

    @cached(300.0, clock=clock)
    def build(limit: int = 200) -> tuple[int, int]:
        calls.append(1)
        return (limit, len(calls))

    assert build(limit=100) == (100, 1)
    assert build(limit=100) == (100, 1)  # cached
    assert build(limit=50000) == (50000, 2)  # different key
    clock.advance(300.0)
    assert build(limit=100) == (100, 3)  # expired


def test_decorator_custom_key_ignores_unhashable_arguments():
    """The session-style argument is excluded from the key on purpose."""
    clock = FakeClock()
    calls = []

    @cached(60.0, key=lambda db: "fixed", clock=clock)
    def payload(db: dict) -> int:
        calls.append(1)
        return len(calls)

    assert payload({"an": "unhashable session"}) == 1
    assert payload({"another": "session"}) == 1  # same key -> cached
    clock.advance(60.0)
    assert payload({"a": "third"}) == 2


def test_decorator_exposes_cache_for_invalidation():
    calls = []

    @cached(300.0)
    def value() -> int:
        calls.append(1)
        return len(calls)

    assert value() == 1
    assert value() == 1
    value.cache.invalidate()
    assert value() == 2
