"""Small text-generation interface and an HTTP client for the user simulator."""

from __future__ import annotations

import json
import os
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROMPT_VERSION = "open-interview-v1"


class LLMError(RuntimeError):
    """Do not silently turn a failed LLM call into a valid patient response."""


class TextGenerator(Protocol):
    def complete(self, messages: list[dict], *, max_tokens: int) -> str: ...


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    try:
        value = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise LLMError("Expected one JSON object from the LLM") from exc
    if not isinstance(value, dict):
        raise LLMError("Expected a JSON object")
    return value


class HTTPTextGenerator:
    def __init__(self, base_url: str, model: str, api_key_env: str,
                 temperature: float = 0.0, timeout: float = 90.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, messages: list[dict], *, max_tokens: int) -> str:
        body = json.dumps({"model": self.model, "messages": messages,
                           "temperature": self.temperature, "max_tokens": max_tokens}).encode()
        headers = {"Content-Type": "application/json"}
        key = os.getenv(self.api_key_env)
        if key:
            headers["Authorization"] = "Bearer " + key
        request = Request(self.base_url + "/chat/completions", data=body, headers=headers)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = json.load(response)
            choice = data["choices"][0]
            if choice.get("finish_reason") == "length":
                raise LLMError("Simulator response exceeded max_tokens")
            text = choice["message"]["content"]
            if not isinstance(text, str) or not text.strip():
                raise LLMError("Empty simulator response")
            return text.strip()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            # Response bodies/headers may contain credentials; do not log them.
            raise LLMError(f"Simulator request failed ({type(exc).__name__})") from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError("Malformed simulator response") from exc


def public_context(observation: dict) -> str:
    """Explicit allowlist: never serialize an environment or a private scenario."""
    return json.dumps({key: observation[key] for key in
                       ("public_profile", "food_id", "messages", "asked_actions", "turn_count")},
                      ensure_ascii=False)
