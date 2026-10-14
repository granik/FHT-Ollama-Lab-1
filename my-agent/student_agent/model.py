"""Local Ollama chat adapter."""

from __future__ import annotations

import http.client
import json
import socket
import urllib.error
import urllib.request

from .runtime import TOOL_SPECS


def _default_catalog():
    return {
        name: {
            "arguments": spec["arguments"],
            "description": spec["description"],
            "argument_schema": spec.get("argument_schema", {}),
        }
        for name, spec in TOOL_SPECS.items()
    }


def _action_schema(tool_catalog=None):
    catalog = _default_catalog() if tool_catalog is None else tool_catalog
    variants = []
    for name, specification in catalog.items():
        arguments = specification["arguments"]
        argument_schema = specification.get("argument_schema", {})
        variants.append(
            {
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "enum": [name]},
                    "args": {
                        "type": "object",
                        "properties": {
                            argument: {
                                "type": "string",
                                **argument_schema.get(argument, {}),
                            }
                            for argument in arguments
                        },
                        "required": list(arguments),
                        "additionalProperties": False,
                    },
                },
                "required": ["tool", "args"],
                "additionalProperties": False,
            }
        )
    variants.append(
        {
            "type": "object",
            "properties": {"final": {"type": "string"}},
            "required": ["final"],
            "additionalProperties": False,
        }
    )
    return {"oneOf": variants}


class OllamaModel:
    """Callable adapter for Ollama's local ``/api/chat`` endpoint."""

    def __init__(
        self,
        model_name: str,
        endpoint: str = "http://127.0.0.1:11434/api/chat",
        tool_catalog=None,
    ) -> None:
        self.model_name = model_name
        self.endpoint = endpoint
        self.tool_catalog = _default_catalog() if tool_catalog is None else tool_catalog

    def set_tool_catalog(self, tool_catalog) -> None:
        """Use the runtime's single data-driven tool registry for this run."""
        self.tool_catalog = {
            name: {
                "arguments": tuple(specification["arguments"]),
                "description": specification["description"],
                "argument_schema": specification.get("argument_schema", {}),
            }
            for name, specification in tool_catalog.items()
        }

    def __call__(self, messages: list[dict[str, str]]) -> str:
        """Return ``message.content`` or raise a useful model error."""
        latest_context = messages[-1].get("content", "") if messages else ""
        has_test_failures = '"latest_unsuccessful_verification"' in latest_context
        has_rejected_edits = '"all_previous_rejected_edit_issues"' in latest_context
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "think": False,
            "format": _action_schema(self.tool_catalog),
            "options": {
                # A small amount of variation helps a local model leave a failed
                # deterministic repair after receiving concrete test evidence.
                "temperature": 0.1 if has_test_failures or has_rejected_edits else 0,
                "num_ctx": 16384,
                "num_predict": 2048,
                "repeat_last_n": 64,
                # Code repairs must often copy exact imports, identifiers, and
                # indentation from observations. Increasing this after a rejected
                # action corrupts those required repetitions on small local models.
                "repeat_penalty": 1.05,
            },
        }
        return self._request(payload)

    def _request(self, payload):
        """Send one bounded request to the local Ollama endpoint."""
        for attempt in range(2):
            request = urllib.request.Request(
                self.endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=900) as response:
                    raw = response.read(2_000_001)
                    if len(raw) > 2_000_000:
                        raise RuntimeError("Ollama response exceeded 2,000,000 bytes")
                    body = json.loads(raw.decode("utf-8"))
                    return body["message"]["content"]
            except urllib.error.HTTPError as error:
                detail = error.read(2_000).decode("utf-8", errors="replace")
                repeat_abort = (
                    error.code == 500
                    and "token repeat limit reached" in detail
                    and attempt == 0
                )
                if repeat_abort:
                    payload["options"].update(
                        {"temperature": 0.1, "num_predict": 1024, "repeat_penalty": 1.2}
                    )
                    continue
                raise RuntimeError(
                    f"Ollama returned HTTP {error.code}: {detail or error.reason}"
                ) from error
            except urllib.error.URLError as error:
                if attempt == 0:
                    payload["options"].update(
                        {"temperature": 0.1, "num_predict": 1024, "repeat_penalty": 1.2}
                    )
                    continue
                raise RuntimeError(
                    f"Cannot reach Ollama at {self.endpoint}: {error.reason}"
                ) from error
            except (http.client.RemoteDisconnected, TimeoutError, socket.timeout) as error:
                if attempt == 0:
                    payload["options"].update(
                        {"temperature": 0.1, "num_predict": 1024, "repeat_penalty": 1.2}
                    )
                    continue
                raise RuntimeError(
                    f"Ollama request failed twice: {error or type(error).__name__}"
                ) from error
            except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as error:
                raise RuntimeError(f"Invalid response from Ollama: {error}") from error

        raise RuntimeError("Ollama generation failed after retry")
