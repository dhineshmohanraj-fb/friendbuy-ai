"""
API key authentication — CP5.

Implements a simple Bearer-token guard for the FastAPI server.

Behaviour
---------
- If ``API_KEY`` is **not set** in ``.env`` → auth is **disabled** (open access).
  All requests pass through.  Ideal for local development.
- If ``API_KEY`` is set → every protected endpoint requires the header::

      Authorization: Bearer <your-api-key>

  Missing credentials → ``401 Unauthorized``.
  Wrong credentials  → ``403 Forbidden``.

Usage in server.py::

    from api.auth import require_api_key
    from fastapi import Depends

    @app.post("/ask")
    async def ask(body: AskRequest, _=Depends(require_api_key)):
        ...
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import get_settings

# ``auto_error=False`` means FastAPI will pass ``None`` instead of raising
# a 403 when the header is absent — we handle it ourselves so the error
# message is consistent with the "auth disabled" path.
_bearer = HTTPBearer(auto_error=False)


def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> str | None:
    """
    FastAPI dependency that enforces Bearer token auth when configured.

    Returns the token string on success, or ``None`` if auth is disabled.

    Raises:
        HTTPException 401: Header missing and API key is configured.
        HTTPException 403: Wrong token supplied.
    """
    api_key = get_settings().api_key

    # Auth disabled — let the request through
    if not api_key:
        return None

    # Auth enabled but no Authorization header
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Pass 'Authorization: Bearer <api-key>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Wrong token
    if credentials.credentials != api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    return credentials.credentials
