"""Centralized API gateway for all external LLM calls.

This module provides a single entry point for performing LLM queries,
including basic rate limiting, caching, retries with exponential backoff,
and lightweight logging. The default transport simulates an LLM response
so the system can run offline without external dependencies.
"""

from __future__ import annotations

from collections import deque
import logging
import math
import random
import re
import time
from typing import Callable, Dict, Optional


logger = logging.getLogger("ai_household.gateway")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class APIGateway:
    """Centralized handler for LLM API calls.

    Responsibilities:
    - Rate limiting using a sliding 60s window of request timestamps
    - Response caching keyed by (model, prompt)
    - Retries with exponential backoff on transport exceptions
    - Basic logging of prompts and responses

    Notes:
    - The default `transport` is a simulated LLM that returns deterministic,
      heuristic-based responses to enable repeatable experiments offline.
    - To integrate a real provider, pass a callable as `transport` that
      accepts `(prompt: str, model: str) -> str` and raises on transient
      failures; the gateway will retry according to `max_retries`.
    """

    def __init__(
        self,
        requests_per_minute: int = 30,
        max_retries: int = 3,
        initial_backoff_seconds: float = 0.5,
        transport: Optional[Callable[[str, str], str]] = None,
    ) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")

        self._requests_per_minute = requests_per_minute
        self._request_interval_seconds = 60.0 / float(requests_per_minute)
        self._timestamps: deque[float] = deque()
        self._cache: Dict[tuple[str, str], str] = {}

        self._max_retries = max_retries
        self._initial_backoff_seconds = initial_backoff_seconds
        self._transport = transport or self._default_simulated_transport

    # ---------------------------- Public API ---------------------------- #
    def query(self, prompt: str, model: str = "simulated", use_cache: bool = True) -> str:
        """Query the LLM via the configured transport.

        Args:
            prompt: The prompt to send
            model: Provider/model identifier
            use_cache: Whether to return cached response for identical inputs

        Returns:
            Raw response string
        """
        cache_key = (model, prompt)
        if use_cache and cache_key in self._cache:
            logger.debug("Cache hit for model=%s", model)
            return self._cache[cache_key]

        self._enforce_rate_limit()

        attempt = 0
        backoff = self._initial_backoff_seconds
        while True:
            try:
                logger.info("Dispatching prompt via gateway | model=%s", model)
                logger.debug("Prompt:\n%s", prompt)

                response = self._transport(prompt, model)

                # Record timestamp after a successful call
                now = time.time()
                self._timestamps.append(now)

                logger.debug("Response:\n%s", response)
                if use_cache:
                    self._cache[cache_key] = response
                return response
            except Exception as exc:  # noqa: BLE001 - transport can raise arbitrary exceptions
                if attempt >= self._max_retries:
                    logger.error("Gateway transport failed after %s attempts: %s", attempt + 1, exc)
                    raise
                sleep_seconds = backoff * (2 ** attempt) * (1.0 + random.uniform(-0.1, 0.1))
                sleep_seconds = max(0.05, min(10.0, sleep_seconds))
                logger.warning(
                    "Transport error on attempt %s/%s. Retrying in %.2fs. Error: %s",
                    attempt + 1,
                    self._max_retries + 1,
                    sleep_seconds,
                    exc,
                )
                time.sleep(sleep_seconds)
                attempt += 1

    # -------------------------- Rate Limiting -------------------------- #
    def _enforce_rate_limit(self) -> None:
        """Ensure requests do not exceed the configured requests/minute."""
        now = time.time()

        # Evict timestamps older than 60 seconds
        while self._timestamps and (now - self._timestamps[0]) > 60.0:
            self._timestamps.popleft()

        if len(self._timestamps) >= self._requests_per_minute:
            # Sleep until we can place the next request respecting spacing
            time_since_last = now - self._timestamps[-1]
            sleep_time = max(0.0, self._request_interval_seconds - time_since_last)
            if sleep_time > 0:
                time.sleep(sleep_time)

    # -------------------------- Default Transport ---------------------- #
    @staticmethod
    def _default_simulated_transport(prompt: str, model: str) -> str:  # noqa: ARG004 - model currently unused in simulation
        """A deterministic, heuristic-based simulation of an LLM response.

        The goal is to provide stable, offline behavior to enable tests and
        experiments without external services. The heuristic parses a few
        common scenario cues and produces consistent dollar allocations.
        """
        text = prompt.lower()

        # Identify total windfall amount if present
        amount_match = re.search(r"windfall[^\d$]*(\$?([\d,]+)(?:\.\d{1,2})?)", prompt, flags=re.IGNORECASE)
        total_amount = None
        if amount_match:
            raw = amount_match.group(2).replace(",", "")
            try:
                total_amount = float(raw)
            except ValueError:
                total_amount = None

        # Persona cues
        is_frugal = "frugal" in text or "risk-averse" in text or "saver" in text
        is_spendthrift = "spendthrift" in text or "impulsive" in text or "shopper" in text
        uses_cot = "let's think step by step" in text

        # Decide allocation shares
        save_share = 0.5
        spend_share = 0.5
        if uses_cot:
            save_share, spend_share = 0.7, 0.3
        if is_frugal:
            save_share, spend_share = 0.85, 0.15
        if is_spendthrift:
            save_share, spend_share = 0.2, 0.8

        # If both persona and CoT apply, blend (average) the shares to be moderate
        if (is_frugal or is_spendthrift) and uses_cot:
            save_share = (save_share + 0.7) / 2.0
            spend_share = (spend_share + 0.3) / 2.0

        if total_amount is None:
            # Fallback generic response
            return (
                "I will allocate resources prudently based on needs and savings goals. "
                "Decision: Save $600 and spend $400."
            )

        # Compute dollar amounts and round to nearest dollar for readability
        amount_saved = int(math.floor(total_amount * save_share))
        amount_spent = int(round(total_amount - amount_saved))

        return (
            f"Reasoning: Based on my preferences and context, I will save {save_share:.0%} "
            f"and spend {spend_share:.0%} of the windfall. "
            f"Decision: I will save ${amount_saved:,} and spend ${amount_spent:,}."
        )


__all__ = ["APIGateway"]


