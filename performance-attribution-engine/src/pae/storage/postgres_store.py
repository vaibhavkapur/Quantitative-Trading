"""Optional PostgreSQL repository adapter matching config/postgres_schema.sql."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from pae.engine.pipeline import AttributionReport
from pae.errors import ConfigurationError


class PostgresAttributionStore:
    """PostgreSQL adapter using psycopg v3 (optional dependency; credentials via env var only)."""

    def __init__(self, dsn_env_var: str = "PAE_POSTGRES_DSN") -> None:
        self.dsn_env_var = dsn_env_var
        dsn = os.environ.get(dsn_env_var)
        if not dsn:
            raise ConfigurationError(
                f"PostgreSQL DSN environment variable {dsn_env_var!r} is not set. "
                "Store credentials outside code."
            )
        try:
            import psycopg  # type: ignore
        except ImportError as exc:
            raise ConfigurationError(
                "psycopg is not installed. Install optional extra `pip install .[postgres]`."
            ) from exc
        self._conn = psycopg.connect(dsn)

    def close(self) -> None:
        self._conn.close()

    def save_report(self, report: AttributionReport, *, author_id: str) -> str:
        payload = json.dumps(report.to_dict(), sort_keys=True)
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO pae_attribution_reports (
                    version_id, report_id, portfolio_id, version_number,
                    supersedes_version_id, correction_reason, methodology_version,
                    config_hash, input_hash, report_hash, period_start, period_end,
                    reporting_currency, environment_mode, status, author_id, report_json
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'DRAFT', %s, %s::jsonb)
                """,
                (
                    report.version_id,
                    report.report_id,
                    report.portfolio_id,
                    report.version_number,
                    report.supersedes_version_id,
                    report.correction_reason,
                    report.methodology_version,
                    report.config_hash,
                    report.input_hash,
                    report.report_hash,
                    report.period_start,
                    report.period_end,
                    report.reporting_currency,
                    report.environment_mode,
                    author_id,
                    payload,
                ),
            )
        self._conn.commit()
        return report.version_id

    def list_versions(self, report_id: str) -> List[Dict[str, Any]]:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT version_id, version_number, status, methodology_version, report_hash
                FROM pae_attribution_reports
                WHERE report_id = %s
                ORDER BY version_number DESC
                """,
                (report_id,),
            )
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
