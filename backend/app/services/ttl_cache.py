"""Tiny in-process TTL cache. Standard library only, by design.

WHY THIS EXISTS
---------------
Measured on the deployed t3.micro (2026-10): three endpoints recomputed
expensive work on every request and timed out under modest load —

- ``GET /api/relearning/status`` re-read ``backend/artifacts/
  prediction_outputs.csv`` (36 MB, 307k rows) and re-ran the 200-trial gate
  evaluation per call (>65 s),
- ``GET /api/public-metrics`` and ``GET /api/model-analysis`` rebuilt
  full-artifact aggregates per call (>65 s each),

while light endpoints answered in <1 s. The underlying inputs are a static
committed CSV plus cheap DB counts, so repeat calls within a short window can
legitimately serve the previous result.

WHAT THIS IS NOT
----------------
Not a shared cache (per-process only), not an LRU (the key spaces here are a
handful of entries), and not a general dependency — keep it for service-layer
payloads whose inputs are immutable-per-deployment artifacts or metrics where
a stale-by-TTL readout is explicitly acceptable.

CONCURRENCY
-----------
``get_or_compute`` holds the cache's lock *during* the compute. That is
deliberate single-flight behaviour: on a 1-vCPU box, ten concurrent requests
must produce one 65-second computation, not ten. The lock is re-entrant so a
cached function may call back into its own cache without deadlocking.
"""

from __future__ import annotations

import threading
import time
from functools import wraps
from typing import Any, Callable, Hashable

__all__ = ["TTLCache", "cached"]

_UNSET = object()


def _always(_value: Any) -> bool:
    return True


class TTLCache:
    """Maps key -> (value, computed_at) with per-entry time-to-live.

    ``clock`` is injectable (tests pass a fake monotonic clock instead of
    sleeping). It must be monotonic-like: a float that never decreases.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.RLock()
        self._store: dict[Hashable, tuple[Any, float]] = {}

    def get_or_compute(
        self,
        key: Hashable,
        ttl_seconds: float,
        compute: Callable[[], Any],
        *,
        cache_result: Callable[[Any], bool] = _always,
    ) -> Any:
        """Return the cached value for ``key`` if younger than ``ttl_seconds``,
        otherwise call ``compute()`` and store its result.

        ``cache_result`` decides whether a freshly computed value is worth
        keeping. Returning False (e.g. for an error/empty fallback payload)
        means the next call recomputes immediately, so degraded answers never
        outlive the condition that caused them.

        Exceptions from ``compute`` propagate unchanged and nothing is stored,
        preserving the caller's existing error contract.
        """
        with self._lock:
            entry = self._store.get(key, _UNSET)
            if entry is not _UNSET:
                value, computed_at = entry
                if (self._clock() - computed_at) < ttl_seconds:
                    return value
            value = compute()
            if cache_result(value):
                self._store[key] = (value, self._clock())
            else:
                # Drop any expired entry rather than refreshing its timestamp.
                self._store.pop(key, None)
            return value

    def invalidate(self, key: Hashable | None = None) -> None:
        """Forget one key, or everything when called with no argument."""
        with self._lock:
            if key is None:
                self._store.clear()
            else:
                self._store.pop(key, None)


def cached(
    ttl_seconds: float,
    *,
    key: Callable[..., Hashable] | None = None,
    cache_result: Callable[[Any], bool] = _always,
    clock: Callable[[], float] = time.monotonic,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator form: ``@cached(60)`` memoises a function for 60 seconds.

    ``key`` maps the call's arguments to a hashable cache key; by default the
    positional args and sorted kwargs are the key, so every argument must be
    hashable. Pass an explicit ``key`` to ignore unhashable arguments (such as
    a DB session) — with the correctness caveat that ignored arguments must
    not change what the function returns within one TTL window.

    The wrapped function exposes its cache as ``fn.cache`` so tests and
    operators can ``fn.cache.invalidate()``.
    """

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        cache = TTLCache(clock=clock)

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if key is not None:
                cache_key: Hashable = key(*args, **kwargs)
            else:
                cache_key = (args, tuple(sorted(kwargs.items())))
            return cache.get_or_compute(
                cache_key,
                ttl_seconds,
                lambda: fn(*args, **kwargs),
                cache_result=cache_result,
            )

        wrapper.cache = cache  # type: ignore[attr-defined]
        return wrapper

    return decorate
