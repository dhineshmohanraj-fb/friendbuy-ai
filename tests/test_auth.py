"""
Tests for api/auth.py — CP5.

Tests the ``require_api_key`` dependency directly (unit-level, no HTTP server).
The FastAPI DI machinery is not invoked; we call the function with mock
credential objects and mock settings.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from api.auth import require_api_key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _creds(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


class _MockSettings:
    def __init__(self, api_key: str | None):
        self.api_key = api_key


# ===========================================================================
# Auth disabled (no API_KEY configured)
# ===========================================================================

class TestAuthDisabled:
    def test_no_key_configured_no_creds_passes(self, monkeypatch):
        monkeypatch.setattr("api.auth.get_settings", lambda: _MockSettings(api_key=None))
        result = require_api_key(credentials=None)
        assert result is None

    def test_no_key_configured_with_creds_passes(self, monkeypatch):
        monkeypatch.setattr("api.auth.get_settings", lambda: _MockSettings(api_key=None))
        result = require_api_key(credentials=_creds("anything"))
        assert result is None

    def test_empty_string_key_treated_as_disabled(self, monkeypatch):
        monkeypatch.setattr("api.auth.get_settings", lambda: _MockSettings(api_key=""))
        result = require_api_key(credentials=None)
        assert result is None


# ===========================================================================
# Auth enabled — correct key
# ===========================================================================

class TestAuthEnabled:
    def test_correct_key_returns_token(self, monkeypatch):
        monkeypatch.setattr("api.auth.get_settings", lambda: _MockSettings(api_key="secret123"))
        result = require_api_key(credentials=_creds("secret123"))
        assert result == "secret123"

    def test_correct_key_case_sensitive(self, monkeypatch):
        monkeypatch.setattr("api.auth.get_settings", lambda: _MockSettings(api_key="Secret"))
        with pytest.raises(HTTPException) as exc_info:
            require_api_key(credentials=_creds("secret"))
        assert exc_info.value.status_code == 403


# ===========================================================================
# Auth enabled — wrong or missing key
# ===========================================================================

class TestAuthErrors:
    def test_wrong_key_raises_403(self, monkeypatch):
        monkeypatch.setattr("api.auth.get_settings", lambda: _MockSettings(api_key="correct"))
        with pytest.raises(HTTPException) as exc_info:
            require_api_key(credentials=_creds("wrong"))
        assert exc_info.value.status_code == 403

    def test_missing_creds_raises_401(self, monkeypatch):
        monkeypatch.setattr("api.auth.get_settings", lambda: _MockSettings(api_key="mykey"))
        with pytest.raises(HTTPException) as exc_info:
            require_api_key(credentials=None)
        assert exc_info.value.status_code == 401

    def test_401_has_www_authenticate_header(self, monkeypatch):
        monkeypatch.setattr("api.auth.get_settings", lambda: _MockSettings(api_key="mykey"))
        with pytest.raises(HTTPException) as exc_info:
            require_api_key(credentials=None)
        assert "WWW-Authenticate" in exc_info.value.headers

    def test_403_detail_mentions_invalid(self, monkeypatch):
        monkeypatch.setattr("api.auth.get_settings", lambda: _MockSettings(api_key="correct"))
        with pytest.raises(HTTPException) as exc_info:
            require_api_key(credentials=_creds("bad"))
        assert "Invalid" in exc_info.value.detail

    def test_401_detail_mentions_required(self, monkeypatch):
        monkeypatch.setattr("api.auth.get_settings", lambda: _MockSettings(api_key="key"))
        with pytest.raises(HTTPException) as exc_info:
            require_api_key(credentials=None)
        assert "required" in exc_info.value.detail.lower() or "Authentication" in exc_info.value.detail
