from __future__ import annotations

import hmac
import os
from collections.abc import Callable, Mapping
from pathlib import Path

from fastapi import Header, HTTPException, Request, status

API_TOKEN_ENV = "SURVEYTOOL_API_TOKEN"
API_TOKEN_CREDENTIAL = "api_token"
# 32 characters is 128 bits for a random hex token (`openssl rand -hex 32`).
MIN_TOKEN_LENGTH = 32
# Rejects "a" * 32 and repeated short words that pass the length check.
MIN_DISTINCT_CHARS = 16


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
        except OSError as exc:
            # Name the failure class only: the message may carry paths.
            raise ApiTokenError(
                f"cannot read the systemd credential {API_TOKEN_CREDENTIAL} ({type(exc).__name__})"
            ) from None
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
    if len(set(token)) < MIN_DISTINCT_CHARS:
        raise ApiTokenError(
            f"API token must contain at least {MIN_DISTINCT_CHARS} distinct characters"
        )
    return token


def bearer_dependency(token: str, public_paths: frozenset[str] = frozenset()) -> Callable[..., None]:
    """A FastAPI dependency answering 401 unless the request carries
    `Authorization: Bearer <token>`, except for the explicitly listed
    public route paths (default: none, so new routes are closed). Compared in constant time on bytes (so a
    non-ASCII header cannot raise)."""
    expected = token.encode()

    def require_token(request: Request, authorization: str | None = Header(default=None)) -> None:
        route = request.scope.get("route")
        if getattr(route, "path", None) in public_paths:
            return
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
