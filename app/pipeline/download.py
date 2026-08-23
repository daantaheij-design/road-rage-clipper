"""Streaming, SSRF-safe download of a user-supplied video URL to local disk."""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

import httpx

from app.security import UnsafeURLError, resolve_and_validate_host, validate_url

logger = logging.getLogger(__name__)

MAX_REDIRECTS = 5
CHUNK_SIZE = 1024 * 1024  # 1 MiB


class DownloadError(Exception):
    pass


async def download_video(
    url: str,
    dest_path: Path,
    *,
    max_bytes: int,
    timeout_seconds: float,
) -> Path:
    """Download `url` to `dest_path`, enforcing SSRF safety and a hard size cap.

    Redirects are followed manually (up to MAX_REDIRECTS) so that every hop
    is re-validated against the SSRF rules - an open redirect on an allowed
    host must not be usable to pivot to an internal address.
    """
    validate_url(url)
    current_url = url

    timeout = httpx.Timeout(timeout_seconds, connect=30.0)
    async with httpx.AsyncClient(follow_redirects=False, timeout=timeout) as client:
        for hop in range(MAX_REDIRECTS + 1):
            parsed = urlparse(current_url)
            validate_url(current_url)
            resolve_and_validate_host(parsed.hostname)  # re-check right before connecting

            async with client.stream("GET", current_url, headers={"User-Agent": "road-rage-clipper/1.0"}) as resp:
                if resp.status_code in (301, 302, 303, 307, 308):
                    location = resp.headers.get("location")
                    if not location:
                        raise DownloadError(f"Redirect ({resp.status_code}) without Location header")
                    current_url = httpx.URL(current_url).join(location).human_repr()
                    if hop == MAX_REDIRECTS:
                        raise DownloadError("Too many redirects")
                    continue

                if resp.status_code != 200:
                    raise DownloadError(f"Server returned HTTP {resp.status_code} for {current_url}")

                content_length = resp.headers.get("content-length")
                if content_length is not None and int(content_length) > max_bytes:
                    raise DownloadError(
                        f"Video is too large ({int(content_length) / 1e6:.0f} MB, "
                        f"limit is {max_bytes / 1e6:.0f} MB)"
                    )

                content_type = resp.headers.get("content-type", "")
                if content_type and not (
                    content_type.startswith("video/")
                    or content_type.startswith("application/octet-stream")
                    or content_type.startswith("binary/")
                ):
                    logger.warning("Unexpected content-type '%s' for %s - proceeding anyway", content_type, current_url)

                dest_path.parent.mkdir(parents=True, exist_ok=True)
                written = 0
                with open(dest_path, "wb") as f:
                    async for chunk in resp.aiter_bytes(CHUNK_SIZE):
                        written += len(chunk)
                        if written > max_bytes:
                            raise DownloadError(
                                f"Video exceeded the {max_bytes / 1e6:.0f} MB size limit during download"
                            )
                        f.write(chunk)

                if written == 0:
                    raise DownloadError("Downloaded file was empty")

                return dest_path

    raise DownloadError("Failed to download video (too many redirects)")


__all__ = ["download_video", "DownloadError", "UnsafeURLError"]
