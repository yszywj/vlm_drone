"""Offline checks for default loopback routing and custom transport priority."""

from io import BytesIO
import os
from unittest.mock import Mock

import pytest

from models.openai_compatible_client import OpenAICompatibleClient
from models import openai_compatible_client as client_module


@pytest.fixture
def default_transports(monkeypatch):
    direct = Mock(return_value=BytesIO(b'{"data":[]}'))
    proxied = Mock(return_value=BytesIO(b'{"data":[]}'))
    builder = Mock(return_value=Mock(open=direct))
    monkeypatch.setattr(client_module.urllib_request, "build_opener", builder)
    monkeypatch.setattr(client_module.urllib_request, "urlopen", proxied)
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    return direct, proxied, builder


@pytest.mark.parametrize("base_url", [
    "http://127.0.0.1:8000/v1",
    "http://127.12.34.56:18080/v1",
    "https://127.255.255.254:8443/v1",
    "http://localhost:8000/v1",
    "http://LOCALHOST:8000/v1",
    "http://[::1]:8000/v1",
    "http://[0:0:0:0:0:0:0:1]:8000/v1",
])
def test_actual_loopback_requests_ignore_environment_proxy(base_url, default_transports):
    direct, proxied, builder = default_transports
    environment = dict(os.environ)
    client = OpenAICompatibleClient(base_url=base_url, model="model", timeout_s=9)
    client.healthcheck()
    proxied.assert_not_called()
    builder.assert_called_once()
    handler = builder.call_args.args[0]
    assert isinstance(handler, client_module.urllib_request.ProxyHandler)
    assert handler.proxies == {}
    direct.assert_called_once()
    assert direct.call_args.args[0].full_url == client.base_url + "/models"
    assert direct.call_args.kwargs == {"timeout": 9.0}
    assert dict(os.environ) == environment


@pytest.mark.parametrize("base_url", [
    "https://model.example/v1",
    "http://192.168.1.2:8000/v1",
    "http://128.0.0.1:8000/v1",
    "http://[2001:db8::1]:8000/v1",
    "http://127.evil:8000/v1",
    "http://127.0.0.1.evil:8000/v1",
    "http://localhost.evil:8000/v1",
    "http://example.test/127.0.0.1/v1",
])
def test_other_hosts_retain_default_proxy_aware_transport(base_url, default_transports):
    direct, proxied, builder = default_transports
    OpenAICompatibleClient(base_url=base_url, model="model").healthcheck()
    proxied.assert_called_once()
    direct.assert_not_called()
    builder.assert_not_called()


@pytest.mark.parametrize("base_url", [
    "http://127.0.0.1:8000/v1", "http://model.example/v1",
])
def test_explicit_transport_keeps_priority_even_if_falsey(base_url, default_transports):
    class FalseyTransport:
        def __init__(self):
            self.requests = []

        def __bool__(self):
            return False

        def __call__(self, request, *, timeout):
            self.requests.append((request, timeout))
            return BytesIO(b'{"data":[]}')

    transport = FalseyTransport()
    client = OpenAICompatibleClient(base_url=base_url, model="model", transport=transport)
    client.healthcheck()
    assert len(transport.requests) == 1
    for mock in default_transports:
        mock.assert_not_called()


@pytest.mark.parametrize("base_url", [
    "http://user@127.0.0.1:8000/v1",
    "http://127.0.0.1@model.example/v1",
    "http://model.example@localhost:8000/v1",
])
def test_userinfo_is_rejected_before_any_transport_selection(base_url, default_transports):
    with pytest.raises(ValueError, match="credentials"):
        OpenAICompatibleClient(base_url=base_url, model="model")
    for mock in default_transports:
        mock.assert_not_called()
