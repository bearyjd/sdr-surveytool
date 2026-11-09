from __future__ import annotations

import hmac
import os
from collections.abc import Callable, Mapping
from pathlib import Path

from fastapi import Header, HTTPException, status

API_TOKEN_ENV = "SURVEYTOOL_API_TOKEN"
API_TOKEN_CREDENTIAL = "api_token"
# 32 characters is 128 bits for a random hex token (`openssl rand -hex 32`).
MIN_TOKEN_LENGTH = 32


class ApiTokenError(ValueError):
    """The API token is missing or too weak; the server must not start."""


def load_api_token(environ: Mapping[str, str] = os.environ) -> str:
    """$SURVEYTOOL_API_TOKEN, else the systemd credential api_token
    (LoadCredential=). Refuses a missing, empty or short token rather than
    defaulting open. Never puts the token itself in the message."""
    token = (environ.get(API_TOKEN_ENV) or "").strip()
    credentials = environ.get("CREDENTIALS_DIRECTORY")
    if not token and credentials:
        try:
            token = (Path(credentials) / API_TOKEN_CREDENTIAL).read_text().strip()
        except FileNotFoundError:
            token = ""
    if not token:
        raise ApiTokenError(
            f"{API_TOKEN_ENV} (or the systemd credential {API_TOKEN_CREDENTIAL}) is required; "
            "generate one with `openssl rand -hex 32`"
        )
    return validate_api_token(token)


def validate_api_token(token: str) -> str:
    token = token.strip()
    if len(token) < MIN_TOKEN_LENGTH:
        raise ApiTokenError(f"API token must be at least {MIN_TOKEN_LENGTH} characters")
    return token


def bearer_dependency(token: str) -> Callable[..., None]:
    """A FastAPI dependency answering 401 unless the request carries
    `Authorization: Bearer <token>`. Compared in constant time on bytes (so a
    non-ASCII header cannot raise)."""
    expected = token.encode()

    def require_token(authorization: str | None = Header(default=None)) -> None:
        scheme, _, presented = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(
            presented.encode("latin-1"), expected
        ):
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "Unauthorized",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return require_token
