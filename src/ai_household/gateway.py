"""Simplified API gateway using only Instructor for structured LLM calls."""

from __future__ import annotations

import logging
from typing import Type

import instructor
from openai import OpenAI
from pydantic import BaseModel


logger = logging.getLogger("ai_household.gateway")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class APIGateway:
    """Simplified gateway for structured LLM API calls using Instructor.
    
    This gateway provides a single interface for making structured LLM calls
    using the Instructor library, which handles validation, retries, and 
    type safety automatically.
    """

    def __init__(self, max_retries: int = 3) -> None:
        self._max_retries = max_retries
        self._client = instructor.from_openai(OpenAI())

    def query_structured(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        model: str = "gpt-4o-mini",
    ) -> BaseModel:
        """Query the LLM for structured output using Instructor.

        Args:
            prompt: The prompt to send to the LLM
            response_model: Pydantic model defining the expected response structure
            model: The model to use for the query

        Returns:
            Validated Pydantic model instance

        Raises:
            Various exceptions if the API call fails or validation fails after retries
        """
        logger.info("Making structured API call | model=%s response_model=%s", model, response_model.__name__)
        logger.debug("Prompt: %s", prompt)

        result = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_model=response_model,
            max_retries=self._max_retries,
        )

        logger.debug("Response: %s", result)
        return result


__all__ = ["APIGateway"]


