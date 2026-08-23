"""Simple single-user authentication.

This is private personal software - there is exactly one "user" (you), so a
full accounts system would be overkill. Instead: one shared password
(APP_PASSWORD) gates the /upload page and its API, set via a signed,
httpOnly session cookie. The MCP server uses a separate bearer token
(MCP_API_KEY) since it's not a browser client.
"""

from __future__ import annotations

import hmac

from fastapi import Request
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from app.config import Settings, get_settings

SESSION_COOKIE = "rrc_session"
SESSION_MAX_AGE = 60 * 60 * 24 * 30  # 30 days


def _serializer(settings: Settings) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(settings.secret_key, salt="rrc-session")


def create_session_token(settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    return _serializer(settings).dumps({"authed": True})


def verify_session_token(token: str, settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    try:
        data = _serializer(settings).loads(token, max_age=SESSION_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return False
    return bool(data.get("authed"))


def check_password(password: str, settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    return hmac.compare_digest(password, settings.app_password)


def is_authed(request: Request) -> bool:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return False
    return verify_session_token(token)


def check_bearer(request: Request, expected: str) -> bool:
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        return False
    token = header[7:].strip()
    return hmac.compare_digest(token, expected)
