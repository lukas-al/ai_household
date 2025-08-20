"""Centralized API gateway for all external LLM calls.

This module provides a single entry point for performing LLM queries. It
supports:
- A simple text API (`query`) that uses a simulated transport by default
  for fully offline operation
- A structured API (`query_structured`) powered by the `instructor` package
  for Pydantic-validated responses with automatic retries, when available

If the `instructor` package or an API client (e.g., OpenAI) is not available,
the structured API falls back to a deterministic local simulation.
"""

from __future__ import annotations

from collections import deque
import json
import logging
import math
import random
import re
import time
from typing import Callable, Dict, Optional, Type, Any


logger = logging.getLogger("ai_household.gateway")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class APIGateway:
    """Centralized handler for LLM API calls.

    Responsibilities:
    - Rate limiting using a sliding 60s window of request timestamps
    - Response caching keyed by (model, prompt)
    - Retries (text API only); structured API delegates retries to instructor
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

        # Lazy-initialized Instructor/OpenAI client
        self._instructor_client = None

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

    # ------------------------ Structured Responses --------------------- #
    def query_structured(
        self,
        prompt: str,
        response_model: Type[Any],
        model: str = "gpt-4o-mini",
        use_cache: bool = True,
    ) -> Any:
        """Return a Pydantic model using Instructor if available, else simulate.

        When `instructor` and an API backend (OpenAI) are available, this uses
        `response_model=` to obtain validated objects with built-in retry logic.
        Otherwise, it falls back to the local deterministic simulation.
        """
        cache_key = (f"structured:{model}:{response_model.__name__}", prompt)
        if use_cache and cache_key in self._cache:
            logger.debug("Structured cache hit for model=%s", model)
            return self._cache[cache_key]

        try:
            client = self._get_instructor_client()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Instructor client unavailable, falling back to simulation: %s", exc)
            client = None

        if client is not None:
            logger.info("Dispatching structured prompt via Instructor | model=%s", model)
            # Instructor handles retry; we still log
            result = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_model=response_model,
                max_retries=self._max_retries,
            )
            if use_cache:
                self._cache[cache_key] = result
            return result

        # Fallback to local simulation for structured outputs
        logger.info("Using local simulation for structured output")
        simulated = self._simulate_structured(prompt, response_model)
        if use_cache:
            self._cache[cache_key] = simulated
        return simulated

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

        # Determine whether the caller requested structured JSON output
        wants_json = ("valid json object" in text) or bool(re.search(r"```json", prompt, flags=re.IGNORECASE))

        if total_amount is None:
            # Fallback generic response
            if wants_json:
                # Return a deterministic JSON fallback
                return json.dumps({"amount_spent": 400, "amount_saved": 600})
            return (
                "I will allocate resources prudently based on needs and savings goals. "
                "Decision: Save $600 and spend $400."
            )

        # Compute dollar amounts and round to nearest dollar for readability
        amount_saved = int(math.floor(total_amount * save_share))
        amount_spent = int(round(total_amount - amount_saved))

        if wants_json:
            return json.dumps({"amount_spent": amount_spent, "amount_saved": amount_saved})

        return (
            f"Reasoning: Based on my preferences and context, I will save {save_share:.0%} "
            f"and spend {spend_share:.0%} of the windfall. "
            f"Decision: I will save ${amount_saved:,} and spend ${amount_spent:,}."
        )

    # -------------------------- Instructor Setup ----------------------- #
    def _get_instructor_client(self):
        """Create and cache an Instructor-wrapped OpenAI client if possible.

        We import lazily so that the package remains importable without
        optional dependencies or credentials.
        """
        if self._instructor_client is not None:
            return self._instructor_client

        try:
            import instructor  # type: ignore
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("instructor/OpenAI not available") from exc

        # Configure Instructor in JSON-Schema mode for reliable structured IO
        try:
            client = instructor.from_openai(OpenAI())  # defaults to JSON_SCHEMA mode
        except AttributeError:
            # Older versions use `patch` API
            client = instructor.patch(OpenAI())

        self._instructor_client = client
        return self._instructor_client

    # -------------------------- Structured Fallback -------------------- #
    @staticmethod
    def _simulate_structured(prompt: str, response_model: Type[Any]) -> Any:
        """Heuristic simulation that instantiates the provided `response_model`.

        This mirrors the allocation logic from `_default_simulated_transport`.
        """
        import re
        import math
        text = prompt.lower()

        amount_match = re.search(r"windfall[^\d$]*(\$?([\d,]+)(?:\.\d{1,2})?)", prompt, flags=re.IGNORECASE)
        total_amount = None
        if amount_match:
            raw = amount_match.group(2).replace(",", "")
            try:
                total_amount = float(raw)
            except ValueError:
                total_amount = None

        is_frugal = "frugal" in text or "risk-averse" in text or "saver" in text
        is_spendthrift = "spendthrift" in text or "impulsive" in text or "shopper" in text
        uses_cot = "let's think step by step" in text

        save_share = 0.5
        spend_share = 0.5
        if uses_cot:
            save_share, spend_share = 0.7, 0.3
        if is_frugal:
            save_share, spend_share = 0.85, 0.15
        if is_spendthrift:
            save_share, spend_share = 0.2, 0.8
        if (is_frugal or is_spendthrift) and uses_cot:
            save_share = (save_share + 0.7) / 2.0
            spend_share = (spend_share + 0.3) / 2.0

        if total_amount is None:
            amount_saved = 600
            amount_spent = 400
        else:
            amount_saved = int(math.floor(total_amount * save_share))
            amount_spent = int(round(total_amount - amount_saved))

        try:
            # Assume Pydantic model with these fields
            return response_model(amount_spent=amount_spent, amount_saved=amount_saved)
        except Exception:
            # If a different model is provided, return a dict as a last resort
            return {"amount_spent": amount_spent, "amount_saved": amount_saved}


__all__ = ["APIGateway"]


