"""API package exposing zero-dependency HTTP server and optional FastAPI app."""

from __future__ import annotations

from dcae.api.fastapi_app import create_fastapi_app
from dcae.api.http_server import DCAEServiceApp, make_http_server

__all__ = ["DCAEServiceApp", "create_fastapi_app", "make_http_server"]
