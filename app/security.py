"""SSRF-safe URL validation.

This tool lets the owner paste an arbitrary video URL and the server fetches
it. Without safeguards that's a classic SSRF vector (the server could be
tricked into fetching http://169.254.169.254/... cloud metadata, or
http://localhost:6379/ internal services). This module validates a URL
points at a public host before we ever open a connection to it, and is
re-used to validate every redirect hop during the actual download.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

ALLOWED_SCHEMES = {"http", "https"}

# Hostnames that are never acceptable regardless of what they resolve to.
_BLOCKED_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
    "metadata.google.internal",
}


class UnsafeURLError(ValueError):
    """Raised when a URL fails SSRF validation."""


def _is_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return False
    # IPv4-mapped IPv6 addresses (::ffff:127.0.0.1 etc.) - unwrap and recheck.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return _is_public_ip(ip.ipv4_mapped)
    # Cloud metadata endpoint, belt-and-suspenders (also caught by link-local).
    if str(ip) == "169.254.169.254":
        return False
    return True


def resolve_and_validate_host(hostname: str) -> list[str]:
    """Resolve a hostname and ensure every A/AAAA record is a public address.

    Returns the list of resolved IP strings on success. Raises
    UnsafeURLError if the hostname is blocked, unresolvable, or resolves to
    any non-public address (defense against DNS rebinding to private IPs).
    """
    if hostname.lower() in _BLOCKED_HOSTNAMES:
        raise UnsafeURLError(f"Host '{hostname}' is not allowed")

    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise UnsafeURLError(f"Could not resolve host '{hostname}': {exc}") from exc

    if not infos:
        raise UnsafeURLError(f"Host '{hostname}' did not resolve to any address")

    resolved_ips: list[str] = []
    for info in infos:
        raw_ip = info[4][0]
        # Strip IPv6 zone id if present (fe80::1%eth0).
        raw_ip = raw_ip.split("%")[0]
        try:
            ip_obj = ipaddress.ip_address(raw_ip)
        except ValueError as exc:
            raise UnsafeURLError(f"Host '{hostname}' resolved to invalid address {raw_ip}") from exc
        if not _is_public_ip(ip_obj):
            raise UnsafeURLError(
                f"Host '{hostname}' resolves to a non-public address ({raw_ip}) - blocked"
            )
        resolved_ips.append(str(ip_obj))

    return resolved_ips


def validate_url(url: str) -> str:
    """Validate a user-supplied video URL is safe to fetch from the server.

    Returns the hostname on success. Raises UnsafeURLError otherwise.
    """
    if not url or len(url) > 4096:
        raise UnsafeURLError("URL is missing or too long")

    parsed = urlparse(url)

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise UnsafeURLError(f"URL scheme '{parsed.scheme}' is not allowed (use http/https)")

    if not parsed.hostname:
        raise UnsafeURLError("URL has no hostname")

    if parsed.username or parsed.password:
        raise UnsafeURLError("URLs with embedded credentials are not allowed")

    if parsed.port is not None and parsed.port not in (80, 443) and parsed.port < 1024:
        raise UnsafeURLError("Non-standard privileged port is not allowed")

    resolve_and_validate_host(parsed.hostname)
    return parsed.hostname
