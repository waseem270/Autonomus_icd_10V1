"""
LLM Provider Abstraction — routes calls to OpenAI or Gemini transparently.

All services call ``call_llm(...)`` which delegates to the active provider.
The OpenAI path converts Gemini-style parameters (GenerateContentConfig,
response_schema, safety_settings) into OpenAI Chat Completions parameters,
and wraps the OpenAI response in a Gemini-compatible object so downstream
parsing code works unchanged.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

from ..core.config import (
    settings,
    get_llm_provider,
    get_active_model,
    create_openai_client,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gemini-compatible response shim — wraps OpenAI responses
# ---------------------------------------------------------------------------
@dataclass
class _FakePart:
    text: str = ""
    thought: bool = False


@dataclass
class _FakeContent:
    parts: List[_FakePart] = field(default_factory=list)


@dataclass
class _FakeCandidate:
    content: _FakeContent = field(default_factory=_FakeContent)
    finish_reason: str = "STOP"


@dataclass
class _FakeUsage:
    prompt_token_count: int = 0
    candidates_token_count: int = 0
    total_token_count: int = 0
    thoughts_token_count: int = 0


@dataclass
class GeminiCompatResponse:
    """Wraps an OpenAI response so it looks like a Gemini response."""
    candidates: List[_FakeCandidate] = field(default_factory=list)
    usage_metadata: Optional[_FakeUsage] = None

    @property
    def text(self) -> str:
        """Convenience property matching the real Gemini response.text interface."""
        if self.candidates:
            parts = self.candidates[0].content.parts
            if parts:
                return parts[0].text
        return ""


def _schema_to_json_schema(schema_obj: Any) -> Optional[dict]:
    """Convert google.genai types.Schema to JSON Schema dict for OpenAI."""
    if schema_obj is None:
        return None
    try:
        # Manual conversion from Schema object attributes
        result = {}
        type_val = getattr(schema_obj, "type", None)
        if type_val is not None:
            # types.Type.STRING → "Type.STRING" → "string"
            raw = str(type_val).split(".")[-1].lower()
            type_map = {
                "string": "string",
                "number": "number",
                "integer": "integer",
                "boolean": "boolean",
                "array": "array",
                "object": "object",
            }
            result["type"] = type_map.get(raw, "string")

        properties = getattr(schema_obj, "properties", None)
        if properties and isinstance(properties, dict):
            result["properties"] = {}
            for k, v in properties.items():
                child = _schema_to_json_schema(v)
                if child:
                    result["properties"][k] = child

        items = getattr(schema_obj, "items", None)
        if items is not None:
            child = _schema_to_json_schema(items)
            if child:
                result["items"] = child

        required = getattr(schema_obj, "required", None)
        if required:
            result["required"] = list(required)

        return result if result else None
    except Exception as e:
        logger.debug(f"Schema conversion failed: {e}")
        return None


def _build_openai_json_schema_param(schema_obj: Any) -> Optional[dict]:
    """Build the OpenAI response_format parameter from a Gemini schema."""
    json_schema = _schema_to_json_schema(schema_obj)
    if json_schema:
        # OpenAI structured outputs format
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "clinical_analysis",
                "strict": False,
                "schema": json_schema,
            },
        }
    return {"type": "json_object"}


async def call_openai(
    model: str,
    contents: str,
    config: Any,
) -> GeminiCompatResponse:
    """Call OpenAI Chat Completions and return a Gemini-compatible response."""
    client = create_openai_client()
    if client is None:
        raise RuntimeError("OpenAI client not initialized")

    # Extract parameters from config object
    temperature = getattr(config, "temperature", 0.1)
    max_tokens = getattr(config, "max_output_tokens", 16384)
    top_p = getattr(config, "top_p", 0.95)
    system_instruction = getattr(config, "system_instruction", None) or ""

    # Build response_format — use json_object if response_mime_type is JSON
    response_mime = getattr(config, "response_mime_type", None)
    response_format = None
    if response_mime == "application/json":
        response_schema = getattr(config, "response_schema", None)
        if response_schema:
            response_format = _build_openai_json_schema_param(response_schema)
        else:
            response_format = {"type": "json_object"}

    # Build messages
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": str(system_instruction)})

    # contents can be a string or list — normalize
    user_content = contents if isinstance(contents, str) else str(contents)
    messages.append({"role": "user", "content": user_content})

    # Call OpenAI
    capped_tokens = min(max_tokens, 16384)
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": capped_tokens,
        "top_p": top_p,
    }
    if response_format:
        kwargs["response_format"] = response_format

    logger.info(f"OpenAI call: model={model}, temp={temperature}, max_completion_tokens={capped_tokens}")

    response = await client.chat.completions.create(**kwargs)

    # Convert to Gemini-compatible response
    choice = response.choices[0] if response.choices else None
    text = choice.message.content if choice else ""
    finish = choice.finish_reason if choice else "stop"

    usage = _FakeUsage()
    if response.usage:
        usage.prompt_token_count = response.usage.prompt_tokens or 0
        usage.candidates_token_count = response.usage.completion_tokens or 0
        usage.total_token_count = response.usage.total_tokens or 0

    return GeminiCompatResponse(
        candidates=[_FakeCandidate(
            content=_FakeContent(parts=[_FakePart(text=text or "")]),
            finish_reason=finish.upper() if finish else "STOP",
        )],
        usage_metadata=usage,
    )
