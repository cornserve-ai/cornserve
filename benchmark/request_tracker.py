"""Track request lifecycle for steady-state detection during MLLM benchmarks.

Steady-state window:
  - **Start**: ``max_num_seqs`` requests have received their first output
    token, meaning the server has ramped up to full batch occupancy.
  - **End**: ``num_requests - max_num_seqs - 1`` requests have fully
    completed, meaning the server is about to enter its ramp-down phase.

Inspired by ml-energy/mlenergy/llm/benchmark.py ``RequestTracker``.
"""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class RequestTracker:
    """Track request progress and delimit the steady-state window.

    The tracker exposes two ``asyncio.Event`` objects:
      * ``start_event`` — set when ``max_num_seqs`` requests have started
        producing tokens (i.e. received their first output token).
      * ``end_event`` — set when ``num_requests - max_num_seqs - 1``
        requests have finished, signalling the end of steady state.

    Attributes:
        steady_state_start: ``perf_counter`` timestamp when steady state begins.
        steady_state_end: ``perf_counter`` timestamp when steady state ends.
        tokens_at_start: Cumulative generated tokens when steady state begins.
        tokens_at_end: Cumulative generated tokens when steady state ends.
        requests_at_start: Number of finished requests when steady state begins.
        requests_at_end: Number of finished requests when steady state ends.
    """

    def __init__(self, max_num_seqs: int, num_requests: int) -> None:
        self._start_event = asyncio.Event()
        self._end_event = asyncio.Event()

        self._max_num_seqs = max_num_seqs
        self._num_requests = num_requests

        # Counters
        self._num_started: int = 0        # requests that received first token
        self._num_finished: int = 0       # requests that fully completed (success + failure)
        self._num_succeeded: int = 0      # requests that completed successfully
        self._num_generated_tokens: int = 0

        # Steady-state boundary snapshots (filled when events fire)
        self.steady_state_start: float = 0.0
        self.steady_state_end: float = 0.0
        self.tokens_at_start: int = 0
        self.tokens_at_end: int = 0
        self.requests_at_start: int = 0   # successful requests at steady-state start
        self.requests_at_end: int = 0     # successful requests at steady-state end

    # -- Waiting ----------------------------------------------------------

    async def wait_start(self) -> None:
        """Block until the steady-state start event fires."""
        await self._start_event.wait()

    async def wait_end(self) -> None:
        """Block until the steady-state end event fires."""
        await self._end_event.wait()

    # -- Notifications ----------------------------------------------------

    def notify_first_token(self) -> None:
        """Notify that a request has received its first output token."""
        self._num_started += 1
        if self._num_started >= self._max_num_seqs and not self._start_event.is_set():
            self.steady_state_start = time.perf_counter()
            self.tokens_at_start = self._num_generated_tokens
            self.requests_at_start = self._num_succeeded
            self._start_event.set()
            logger.info(
                "Steady state started (after %d first-tokens, %d tokens generated so far)",
                self._num_started,
                self._num_generated_tokens,
            )

    def notify_tokens_generated(self, count: int) -> None:
        """Add *count* newly generated tokens to the cumulative total."""
        self._num_generated_tokens += count

    def notify_request_finished(self, success: bool = True) -> None:
        """Notify that a request has fully completed."""
        self._num_finished += 1
        if success:
            self._num_succeeded += 1
        threshold = self._num_requests - self._max_num_seqs - 1
        if self._num_finished >= threshold and not self._end_event.is_set():
            # Also set start in case all requests finished before reaching
            # steady state (e.g. all failed quickly).
            if not self._start_event.is_set():
                self.steady_state_start = time.perf_counter()
                self.tokens_at_start = self._num_generated_tokens
                self.requests_at_start = self._num_succeeded
                self._start_event.set()
            self.steady_state_end = time.perf_counter()
            self.tokens_at_end = self._num_generated_tokens
            self.requests_at_end = self._num_succeeded
            self._end_event.set()
            logger.info(
                "Steady state ended (after %d/%d finished, %d tokens generated)",
                self._num_finished,
                self._num_requests,
                self._num_generated_tokens,
            )

    # -- Read-only accessors ----------------------------------------------

    @property
    def num_started(self) -> int:
        return self._num_started

    @property
    def num_finished(self) -> int:
        return self._num_finished

    @property
    def num_succeeded(self) -> int:
        return self._num_succeeded

    @property
    def num_generated_tokens(self) -> int:
        return self._num_generated_tokens

    @property
    def steady_state_tokens(self) -> int:
        """Tokens generated during the steady-state window."""
        return self.tokens_at_end - self.tokens_at_start

    @property
    def steady_state_requests(self) -> int:
        """Requests completed during the steady-state window."""
        return self.requests_at_end - self.requests_at_start

    @property
    def steady_state_duration(self) -> float:
        """Duration (seconds) of the steady-state window."""
        return self.steady_state_end - self.steady_state_start
