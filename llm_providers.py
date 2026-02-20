"""
Provider-agnostic LLM interface with Pydantic schema enforcement.
=================================================================
Supports: OpenAI, Anthropic, Google Gemini, Local (Ollama).

Each provider call returns a validated Pydantic model. If the LLM
returns malformed JSON, the provider auto-retries (up to MAX_RETRIES).

Usage:
    from llm_providers import LLMProvider
    from llm_schemas import ScreeningResult

    llm = LLMProvider(provider="openai", model="gpt-4o-mini")
    result = llm.query(system="...", user="...", schema=ScreeningResult)
    print(result.relevant, result.confidence)
"""

from __future__ import annotations

import json
import os
import time
import warnings
from typing import Optional, Type

from pydantic import BaseModel, ValidationError

from config import LLM_MAX_RETRIES, LLM_TEMPERATURE


# ── Default models per provider ──────────────────────────────────────
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
    "google": "gemini-2.0-flash",
    "ollama": "llama3.2",
}

# ── Approximate costs per 1M tokens (input, output) ──────────────────
# Used for cost estimation only — not billed through this code.
COST_PER_1M_TOKENS = {
    "openai": {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.0)},
    "anthropic": {"claude-3-5-haiku-latest": (0.80, 4.00), "claude-sonnet-4-20250514": (3.00, 15.0)},
    "google": {"gemini-2.0-flash": (0.075, 0.30), "gemini-2.5-pro-preview-06-05": (1.25, 10.0)},
    "ollama": {},  # Free
}


class LLMProvider:
    """Provider-agnostic LLM interface with structured output."""

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = LLM_TEMPERATURE,
        max_retries: int = LLM_MAX_RETRIES,
    ):
        self.provider = provider.lower()
        self.model = model or DEFAULT_MODELS.get(self.provider, "gpt-4o-mini")
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = None

        # Resolve API key
        if api_key:
            self.api_key = api_key
        else:
            env_vars = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "google": "GOOGLE_API_KEY",
            }
            env_var = env_vars.get(self.provider)
            self.api_key = os.environ.get(env_var, "") if env_var else ""

        # Track usage
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0

    def _get_client(self):
        """Lazy-initialize the provider client."""
        if self._client is not None:
            return self._client

        if self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)

        elif self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)

        elif self.provider == "google":
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model)

        elif self.provider == "ollama":
            # Ollama uses HTTP API directly
            import requests
            self._client = requests.Session()

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        return self._client

    def query(
        self,
        system: str,
        user: str,
        schema: Type[BaseModel],
    ) -> BaseModel:
        """
        Send a prompt to the LLM and parse the response into a Pydantic model.

        Args:
            system: System prompt (instructions, examples)
            user: User prompt (the paper title + abstract)
            schema: Pydantic model class for response validation

        Returns:
            Validated Pydantic model instance

        Raises:
            RuntimeError: After max_retries exhausted
        """
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                raw_json = self._call_provider(system, user, schema)
                # Parse and validate
                if isinstance(raw_json, str):
                    data = json.loads(raw_json)
                elif isinstance(raw_json, dict):
                    data = raw_json
                else:
                    raise ValueError(f"Unexpected response type: {type(raw_json)}")

                result = schema.model_validate(data)
                return result

            except (json.JSONDecodeError, ValidationError, KeyError, ValueError) as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    warnings.warn(
                        f"LLM response failed validation (attempt {attempt}/{self.max_retries}): {e}. "
                        f"Retrying in {wait}s...",
                        stacklevel=2,
                    )
                    time.sleep(wait)

            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    warnings.warn(
                        f"LLM call failed (attempt {attempt}/{self.max_retries}): {e}. "
                        f"Retrying in {wait}s...",
                        stacklevel=2,
                    )
                    time.sleep(wait)

        raise RuntimeError(
            f"LLM query failed after {self.max_retries} attempts. Last error: {last_error}"
        )

    def _call_provider(
        self,
        system: str,
        user: str,
        schema: Type[BaseModel],
    ) -> dict | str:
        """Dispatch to provider-specific implementation."""
        client = self._get_client()

        if self.provider == "openai":
            return self._call_openai(client, system, user, schema)
        elif self.provider == "anthropic":
            return self._call_anthropic(client, system, user, schema)
        elif self.provider == "google":
            return self._call_google(client, system, user, schema)
        elif self.provider == "ollama":
            return self._call_ollama(client, system, user, schema)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    # ── OpenAI ────────────────────────────────────────────────────────

    def _call_openai(self, client, system: str, user: str, schema: Type[BaseModel]) -> dict:
        """Call OpenAI API with structured output (JSON mode)."""
        json_schema = schema.model_json_schema()

        # Clean schema for OpenAI strict mode (remove unsupported keys)
        clean_schema = self._clean_schema_for_openai(json_schema)

        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": clean_schema,
                    "strict": True,
                },
            },
        )

        # Track tokens
        if response.usage:
            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens
        self.total_calls += 1

        content = response.choices[0].message.content
        return json.loads(content)

    def _clean_schema_for_openai(self, schema: dict) -> dict:
        """
        Clean a JSON schema for OpenAI strict mode.
        OpenAI requires additionalProperties: false and all properties required.
        """
        schema = schema.copy()

        # Remove $defs if present and inline them
        defs = schema.pop("$defs", {})

        def resolve_refs(obj):
            if isinstance(obj, dict):
                if "$ref" in obj:
                    ref_name = obj["$ref"].split("/")[-1]
                    if ref_name in defs:
                        return resolve_refs(defs[ref_name].copy())
                return {k: resolve_refs(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [resolve_refs(item) for item in obj]
            return obj

        schema = resolve_refs(schema)

        # Ensure additionalProperties and required
        def enforce_strict(obj):
            if isinstance(obj, dict):
                if obj.get("type") == "object" and "properties" in obj:
                    obj["additionalProperties"] = False
                    obj["required"] = list(obj["properties"].keys())
                for v in obj.values():
                    enforce_strict(v)
            elif isinstance(obj, list):
                for item in obj:
                    enforce_strict(item)

        enforce_strict(schema)
        return schema

    # ── Anthropic ─────────────────────────────────────────────────────

    def _call_anthropic(self, client, system: str, user: str, schema: Type[BaseModel]) -> dict:
        """Call Anthropic API using tool_use for structured output."""
        json_schema = schema.model_json_schema()

        # Use tool_use to enforce JSON structure
        tool_def = {
            "name": "output_result",
            "description": f"Output the {schema.__name__} result",
            "input_schema": json_schema,
        }

        response = client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[tool_def],
            tool_choice={"type": "tool", "name": "output_result"},
        )

        # Track tokens
        if response.usage:
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens
        self.total_calls += 1

        # Extract tool use result
        for block in response.content:
            if block.type == "tool_use":
                return block.input

        raise ValueError("No tool_use block in Anthropic response")

    # ── Google Gemini ─────────────────────────────────────────────────

    def _call_google(self, client, system: str, user: str, schema: Type[BaseModel]) -> dict:
        """Call Google Gemini API with JSON mode."""
        import google.generativeai as genai

        # Gemini uses generation_config for JSON mode
        generation_config = genai.types.GenerationConfig(
            temperature=self.temperature,
            response_mime_type="application/json",
            response_schema=schema,
        )

        # Combine system + user into a single prompt for Gemini
        full_prompt = f"{system}\n\n---\n\n{user}"

        response = client.generate_content(
            full_prompt,
            generation_config=generation_config,
        )

        self.total_calls += 1
        # Gemini doesn't expose token counts as easily
        if hasattr(response, "usage_metadata"):
            self.total_input_tokens += getattr(response.usage_metadata, "prompt_token_count", 0)
            self.total_output_tokens += getattr(response.usage_metadata, "candidates_token_count", 0)

        return json.loads(response.text)

    # ── Ollama (local) ────────────────────────────────────────────────

    def _call_ollama(self, session, system: str, user: str, schema: Type[BaseModel]) -> dict:
        """Call Ollama local API with JSON format."""
        json_schema = schema.model_json_schema()

        response = session.post(
            "http://localhost:11434/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "format": json_schema,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                },
            },
            timeout=120,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Ollama error {response.status_code}: {response.text}")

        result = response.json()
        self.total_calls += 1

        # Parse the content
        content = result.get("message", {}).get("content", "")
        return json.loads(content)

    # ── Cost estimation ───────────────────────────────────────────────

    def estimate_cost(self, n_papers: int, avg_input_tokens: int = 500, avg_output_tokens: int = 150) -> dict:
        """
        Estimate cost for processing n_papers.

        Args:
            n_papers: Number of papers to process
            avg_input_tokens: Average input tokens per paper (system + user prompt)
            avg_output_tokens: Average output tokens per response

        Returns:
            dict with cost breakdown
        """
        total_input = n_papers * avg_input_tokens
        total_output = n_papers * avg_output_tokens

        provider_costs = COST_PER_1M_TOKENS.get(self.provider, {})
        model_costs = provider_costs.get(self.model)

        if model_costs:
            input_cost = total_input / 1_000_000 * model_costs[0]
            output_cost = total_output / 1_000_000 * model_costs[1]
            total_cost = input_cost + output_cost
        else:
            total_cost = 0.0  # Free (Ollama) or unknown model

        return {
            "provider": self.provider,
            "model": self.model,
            "n_papers": n_papers,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "estimated_cost_usd": round(total_cost, 4),
        }

    def get_usage_summary(self) -> dict:
        """Return current session token usage and estimated cost."""
        provider_costs = COST_PER_1M_TOKENS.get(self.provider, {})
        model_costs = provider_costs.get(self.model)

        if model_costs:
            cost = (
                self.total_input_tokens / 1_000_000 * model_costs[0]
                + self.total_output_tokens / 1_000_000 * model_costs[1]
            )
        else:
            cost = 0.0

        return {
            "provider": self.provider,
            "model": self.model,
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost_usd": round(cost, 4),
        }
