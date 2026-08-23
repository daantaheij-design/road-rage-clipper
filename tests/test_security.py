from __future__ import annotations

import socket

import pytest

from app.security import UnsafeURLError, resolve_and_validate_host, validate_url


def _fake_getaddrinfo(ips):
    def _impl(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0)) for ip in ips]

    return _impl


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/video.mp4",
        "http://localhost:8080/video.mp4",
        "ftp://example.com/video.mp4",
        "file:///etc/passwd",
        "http://user:pass@example.com/video.mp4",
        "http://example.com:22/video.mp4",
        "not a url",
        "",
    ],
)
def test_validate_url_rejects_obviously_unsafe(url):
    with pytest.raises(UnsafeURLError):
        validate_url(url)


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",
        "10.0.0.5",
        "172.16.0.5",
        "192.168.1.5",
        "169.254.169.254",  # cloud metadata endpoint
        "::1",
        "fe80::1",
        "fd00::1",
    ],
)
def test_resolve_rejects_private_and_special_ips(monkeypatch, ip):
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo([ip]))
    with pytest.raises(UnsafeURLError):
        resolve_and_validate_host("evil.example.com")


def test_resolve_allows_public_ip(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo(["93.184.216.34"]))
    ips = resolve_and_validate_host("example.com")
    assert ips == ["93.184.216.34"]


def test_resolve_rejects_if_any_resolved_ip_is_private(monkeypatch):
    # DNS can return multiple A records - if even one is private, block it.
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo(["93.184.216.34", "127.0.0.1"]))
    with pytest.raises(UnsafeURLError):
        resolve_and_validate_host("sneaky.example.com")


def test_validate_url_allows_public_https(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo(["93.184.216.34"]))
    assert validate_url("https://example.com/video.mp4") == "example.com"


def test_validate_url_rejects_unresolvable_host(monkeypatch):
    def _raise(*args, **kwargs):
        raise socket.gaierror("no such host")

    monkeypatch.setattr(socket, "getaddrinfo", _raise)
    with pytest.raises(UnsafeURLError):
        validate_url("https://this-does-not-exist.invalid/video.mp4")
