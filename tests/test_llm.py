import httpx
import pytest

from backend import copilot, llm
from backend.llm import Provider


def respond(handler, monkeypatch):
    """Send the backend's HTTP calls to a local handler instead of a real provider."""
    transport = httpx.MockTransport(handler)
    real = httpx.Client

    def request(method):
        def call(url, **kwargs):
            kwargs.pop("timeout", None)
            with real(transport=transport) as client:
                return getattr(client, method)(url, **kwargs)
        return call

    monkeypatch.setattr(llm.httpx, "get", request("get"))
    monkeypatch.setattr(llm.httpx, "post", request("post"))


def chat_reply(content):
    return httpx.Response(200, json={"model": "test-model", "choices": [{"message": {"role": "assistant", "content": content}}]})


LOCAL = Provider("lmstudio", "LM Studio", "http://local/v1")
CLOUD = Provider("featherless", "Featherless", "http://cloud/v1", needs_key=True, default_model="m")


def test_both_providers_are_registered(monkeypatch):
    monkeypatch.delenv("FEATHERLESS_API_KEY", raising=False)
    assert set(llm.providers()) == {"featherless", "lmstudio"}


def test_cloud_needs_a_key_and_says_how_to_set_it():
    report = llm.status(CLOUD)
    assert not report["available"] and "FEATHERLESS_API_KEY" in report["detail"]


def test_local_status_hides_embedding_models(monkeypatch):
    respond(lambda r: httpx.Response(200, json={"data": [{"id": "text-embedding-nomic"}, {"id": "qwen-chat"}]}), monkeypatch)
    assert llm.status(LOCAL)["models"] == ["qwen-chat"]


def test_local_with_only_embeddings_is_not_ready(monkeypatch):
    respond(lambda r: httpx.Response(200, json={"data": [{"id": "text-embedding-nomic"}]}), monkeypatch)
    report = llm.status(LOCAL)
    assert not report["available"] and "no chat model" in report["detail"]


def test_local_server_down_is_reported(monkeypatch):
    def down(_):
        raise httpx.ConnectError("refused")

    respond(down, monkeypatch)
    assert "Cannot reach" in llm.status(LOCAL)["detail"]


def test_chat_removes_reasoning_and_keeps_the_key_in_the_header(monkeypatch):
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return chat_reply("<think>private</think>Answer")

    respond(handler, monkeypatch)
    keyed = Provider("x", "X", "http://x/v1", api_key="secret", default_model="m")
    assert llm.chat(keyed, [{"role": "user", "content": "hi"}])["message"]["content"] == "Answer"
    assert seen["auth"] == "Bearer secret"


def test_provider_error_is_wrapped(monkeypatch):
    respond(lambda r: httpx.Response(500, text="boom"), monkeypatch)
    keyed = Provider("x", "X", "http://x/v1", default_model="m")
    with pytest.raises(llm.ProviderError, match="500"):
        llm.chat(keyed, [])


def test_invented_citations_are_flagged():
    known = copilot.evidence_ids({"asset_id": "ASSET-001", "doc": "DOC-A", "events": [{"event_id": "EVT-00001"}]})
    answer = "See `EVT-00001` and `EVT-99999`, and `DOC-FIELD-SAFETY`."
    assert copilot.unknown_citations(answer, known) == ["EVT-99999"]


def test_stopped_server_is_a_provider_error_when_no_model_is_set(monkeypatch):
    def down(_):
        raise httpx.ConnectError("refused")

    respond(down, monkeypatch)
    with pytest.raises(llm.ProviderError, match="could not be reached"):
        llm.chat(LOCAL, [])


def test_cloud_offers_its_tested_models_with_the_best_first(monkeypatch):
    monkeypatch.setenv("FEATHERLESS_API_KEY", "k")
    monkeypatch.delenv("FEATHERLESS_MODELS", raising=False)
    report = llm.status(llm.providers()["featherless"])
    assert report["available"] and report["models"][0] == "deepseek-ai/DeepSeek-V3.2" and len(report["models"]) == 3


def test_answer_without_choices_is_a_provider_error(monkeypatch):
    respond(lambda r: httpx.Response(200, json={"error": "busy"}), monkeypatch)
    keyed = Provider("x", "X", "http://x/v1", default_model="m")
    with pytest.raises(llm.ProviderError, match="no answer"):
        llm.chat(keyed, [])
