"""AI providers. Each one is an OpenAI-compatible endpoint, so the agent never depends on a vendor."""
import os
import re
from dataclasses import dataclass

import httpx

TIMEOUT = httpx.Timeout(180.0, connect=5.0)
PROBE_TIMEOUT = httpx.Timeout(4.0, connect=1.5)
NOT_CHAT = re.compile(r"embed|rerank|whisper|tts", re.I)
# Order set by 8 plant questions with known answers: DeepSeek-V3.2 8 of 8, Kimi-K2 6, Qwen2.5-7B 0.
CLOUD_MODELS = "deepseek-ai/DeepSeek-V3.2,moonshotai/Kimi-K2-Instruct,Qwen/Qwen2.5-7B-Instruct"


class ProviderError(RuntimeError):
    """A provider could not be reached or returned an error."""


@dataclass(frozen=True)
class Provider:
    name: str
    label: str
    base_url: str
    api_key: str = ""
    default_model: str = ""
    needs_key: bool = False
    models: tuple[str, ...] = ()

    @property
    def configured(self) -> bool:
        return bool(self.api_key) or not self.needs_key

    @property
    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}


def providers() -> dict[str, Provider]:
    """Built from the environment each time, so keys never leave the server process."""
    cloud = tuple(m.strip() for m in os.environ.get("FEATHERLESS_MODELS", CLOUD_MODELS).split(",") if m.strip())
    return {
        "featherless": Provider(
            "featherless", "Featherless (cloud)", os.environ.get("FEATHERLESS_URL", "https://api.featherless.ai/v1"),
            os.environ.get("FEATHERLESS_API_KEY", ""), cloud[0], needs_key=True, models=cloud),
        "lmstudio": Provider(
            "lmstudio", "LM Studio (local)", os.environ.get("LMSTUDIO_URL", "http://127.0.0.1:1234/v1"),
            os.environ.get("LMSTUDIO_API_KEY", ""), os.environ.get("LMSTUDIO_MODEL", "")),
    }


def get_provider(name: str) -> Provider:
    found = providers().get(name)
    if found is None:
        raise ProviderError(f"unknown provider {name}")
    return found


def list_models(provider: Provider) -> list[str]:
    """Chat models the provider offers. Embedding models cannot answer, so they are left out."""
    response = httpx.get(f"{provider.base_url}/models", headers=provider.headers, timeout=PROBE_TIMEOUT)
    response.raise_for_status()
    return sorted(m["id"] for m in response.json().get("data", []) if not NOT_CHAT.search(m["id"]))


def status(provider: Provider) -> dict:
    """What the page needs to show this provider and whether it can be used."""
    base = {"name": provider.name, "label": provider.label, "default_model": provider.default_model}
    if not provider.configured:
        return {**base, "available": False, "models": [], "detail": f"Set the {provider.name.upper()}_API_KEY environment variable on the server."}
    if provider.needs_key:   # the cloud catalogue has thousands of models, so only the tested ones are offered
        return {**base, "available": True, "models": list(provider.models) or [provider.default_model], "detail": "Ready"}
    try:
        models = list_models(provider)
    except (httpx.HTTPError, ValueError):
        return {**base, "available": False, "models": [], "detail": f"Cannot reach {provider.base_url}. Is the server running?"}
    if not models:
        return {**base, "available": False, "models": [], "detail": "Server is running, but no chat model is available. Load one first."}
    return {**base, "available": True, "models": models, "detail": "Ready"}


def resolve_model(provider: Provider, model: str | None) -> str:
    if model:
        return model
    if provider.default_model:
        return provider.default_model
    models = list_models(provider)
    if not models:
        raise ProviderError(f"{provider.label} has no chat model available")
    return models[0]


def chat(provider: Provider, messages: list[dict], model: str | None = None, tools: list[dict] | None = None,
         max_tokens: int = 700, temperature: float = 0.2) -> dict:
    """One chat completion. Returns the assistant message with reasoning tags removed, and the model used."""
    if not provider.configured:
        raise ProviderError(f"{provider.label} is not configured")
    try:
        chosen = resolve_model(provider, model)
        body = {"model": chosen, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if tools:
            body["tools"] = tools
        response = httpx.post(f"{provider.base_url}/chat/completions", json=body, headers=provider.headers, timeout=TIMEOUT)
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        raise ProviderError(f"{provider.label} {error.response.status_code}: {error.response.text[:300]}") from error
    except httpx.HTTPError as error:
        raise ProviderError(f"{provider.label} could not be reached: {error}") from error
    data = response.json()
    if not data.get("choices"):
        raise ProviderError(f"{provider.label} returned no answer: {str(data)[:300]}")
    message = data["choices"][0]["message"]
    message["content"] = re.sub(r"<think>.*?</think>", "", message.get("content") or "", flags=re.S).strip()
    return {"message": message, "model": data.get("model", chosen)}
