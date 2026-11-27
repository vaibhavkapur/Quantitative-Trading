"""Optional PostgreSQL audit store adapter (isolates psycopg dependency)."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from dcae.errors import DCAEError


class PostgresAuditStore:
    """PostgreSQL adapter reading connection DSN strictly from environment variables.

    Secrets/credentials are never stored in configuration files or source code.
    """

    def __init__(self, dsn_env_var: str = "DCAE_POSTGRES_DSN") -> None:
        dsn = os.environ.get(dsn_env_var)
        if not dsn:
            raise DCAEError(
                f"PostgreSQL DSN environment variable '{dsn_env_var}' is not set",
                code="POSTGRES_CONFIG_MISSING",
                http_status=500,
            )
        try:
            import psycopg  # type: ignore[import-not-found]
        except ImportError as exc:
            raise DCAEError(
                "Optional dependency 'psycopg' is not installed. "
                "Install with `pip install -e .[postgres]` or use SQLiteAuditStore.",
                code="OPTIONAL_DEPENDENCY_MISSING",
                http_status=500,
            ) from exc
        self._psycopg = psycopg
        self._dsn = dsn

    def health_check(self) -> Dict[str, Any]:
        with self._psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                val: Optional[tuple[Any, ...]] = cur.fetchone()
        return {"backend": "postgres", "ok": bool(val and val[0] == 1)}
