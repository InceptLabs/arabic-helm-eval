from typing import Any, Dict, List, Optional

from helm.clients.openai_client import OpenAIClient
from helm.common.cache_backend_config import CacheConfig
from helm.common.request import Request
from helm.common.tokenization_request import TokenizationToken
from helm.tokenizers.tokenizer import Tokenizer

# Default system prompt for MCQ benchmarks — ensures the model outputs only the
# answer letter, preventing HELM's regex from matching letters inside repeated text.
_MCQ_SYSTEM_PROMPT = (
    "أجب عن أسئلة الاختيار من متعدد بحرف الإجابة فقط (أ، ب، ج، د، هـ) دون أي شرح أو تكرار للسؤال."
)


class FireworksNoThinkingClient(OpenAIClient):
    """OpenAI-compatible client for Fireworks AI that always disables thinking.

    Args:
        system_prompt: Optional system prompt to prepend. Defaults to the Arabic
            MCQ prompt. Pass an empty string to suppress for generation tasks.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        tokenizer_name: str,
        cache_config: CacheConfig,
        api_key: Optional[str] = None,
        org_id: Optional[str] = None,
        base_url: Optional[str] = None,
        openai_model_name: Optional[str] = None,
        system_prompt: str = _MCQ_SYSTEM_PROMPT,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            api_key=api_key,
            org_id=org_id,
            base_url=base_url,
            openai_model_name=openai_model_name,
            **kwargs,
        )
        self._system_prompt = system_prompt

    def _make_chat_raw_request(self, request: Request) -> Dict[str, Any]:
        raw_request = super()._make_chat_raw_request(request)
        raw_request["reasoning_effort"] = "off"

        if self._system_prompt:
            messages: List[Dict[str, Any]] = raw_request.get("messages", [])
            if messages and not any(m.get("role") == "system" for m in messages):
                raw_request["messages"] = [
                    {"role": "system", "content": self._system_prompt}
                ] + messages

        return raw_request
