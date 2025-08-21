"""Simplified API gateway using only Instructor for structured LLM calls."""

from __future__ import annotations

import os
import logging
from typing import Type

import instructor
from openai import OpenAI
from pydantic import BaseModel


logger = logging.getLogger("ai_household.gateway")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


class APIGateway:
    """Simplified gateway for structured LLM API calls using Instructor.

    This gateway provides a single interface for making structured LLM calls
    using the Instructor library, which handles validation, retries, and
    type safety automatically.
    """

    def __init__(self, max_retries: int = 2, model: str = "openai/gpt-oss-20b", cache: bool = False) -> None:
        self._model = model
        self._max_retries = max_retries

        self._client = instructor.from_openai(
            client=OpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
            ),
            model=model,
            cache=instructor.cache.DiskCache(directory=".cache") if cache else None,
            mode=instructor.Mode.TOOLS,
        )

    def query_structured(
        self,
        prompt: str,
        pydantic_response_model: Type[BaseModel],
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
        logger.info(
            "Making structured API call | model=%s response_model=%s",
            self._model,
            pydantic_response_model.__name__,
        )
        logger.info("Prompt: %s", prompt)

        result = self._client.chat.completions.create(
            response_model=pydantic_response_model,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"provider": {"require_parameters": True}},
            max_retries=self._max_retries,
        )

        logger.info("Response: %s", result)
        return result


__all__ = ["APIGateway"]
