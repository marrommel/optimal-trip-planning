import logging
import threading
import time


class ProgressLogger:
    """Emit a cumulative INFO summary every `every` queries or `interval_sec`, whichever comes first."""

    def __init__(self, logger: logging.Logger, label: str, unit: str, every: int = 25, interval_sec: float = 60.0):
        self._logger = logger
        self._label = label
        self._unit = unit
        self._every = every
        self._interval = interval_sec
        self._lock = threading.Lock()
        self._queries = 0
        self._empty = 0
        self._items = 0
        self._last_log = time.monotonic()

    def record(self, items: int) -> None:
        with self._lock:
            self._queries += 1
            self._items += items
            self._empty += items == 0
            now = time.monotonic()
            if self._queries % self._every and now - self._last_log < self._interval:
                return
            self._last_log = now
            self._logger.info(
                "%s progress: %d queries done, %d %s found (%d queries without results)",
                self._label, self._queries, self._items, self._unit, self._empty,
            )
