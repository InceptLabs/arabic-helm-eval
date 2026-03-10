from typing import Any, Dict

from helm.clients.openai_client import OpenAIClient
from helm.common.request import Request


class FireworksNoThinkingClient(OpenAIClient):
    """OpenAI-compatible client for Fireworks AI that always disables thinking."""

    def _make_chat_raw_request(self, request: Request) -> Dict[str, Any]:
        raw_request = super()._make_chat_raw_request(request)
        raw_request["reasoning_effort"] = "off"
        return raw_request
