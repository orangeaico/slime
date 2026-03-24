import asyncio
import threading
import weakref

from slime.utils import http_utils


def test_http_client_is_reused_within_one_event_loop(monkeypatch):
    created_clients = []

    def fake_create_http_client(_concurrency):
        client = object()
        created_clients.append(client)
        return client

    monkeypatch.setattr(http_utils, "_client_concurrency", 4)
    monkeypatch.setattr(http_utils, "_http_clients_by_loop", weakref.WeakKeyDictionary())
    monkeypatch.setattr(http_utils, "_http_client_lock", threading.Lock())
    monkeypatch.setattr(http_utils, "_create_http_client", fake_create_http_client)

    async def run_once():
        first = http_utils._get_http_client()
        second = http_utils._get_http_client()
        return first is second

    assert asyncio.run(run_once())
    assert len(created_clients) == 1


def test_http_client_is_scoped_per_event_loop(monkeypatch):
    created_clients = []

    def fake_create_http_client(_concurrency):
        client = object()
        created_clients.append(client)
        return client

    monkeypatch.setattr(http_utils, "_client_concurrency", 4)
    monkeypatch.setattr(http_utils, "_http_clients_by_loop", weakref.WeakKeyDictionary())
    monkeypatch.setattr(http_utils, "_http_client_lock", threading.Lock())
    monkeypatch.setattr(http_utils, "_create_http_client", fake_create_http_client)

    async def get_client():
        return http_utils._get_http_client()

    first = asyncio.run(get_client())
    second = asyncio.run(get_client())

    assert first is not second
    assert len(created_clients) == 2
