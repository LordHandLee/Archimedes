from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

import backtest_engine.asset_provider_ingestion as asset_provider_ingestion
import backtest_engine.asset_reference as asset_reference
from backtest_engine.catalog import ResultCatalog


class AssetProviderIngestionTests(unittest.TestCase):
    def test_defeatbeta_sync_skips_price_history_dataset(self) -> None:
        dataset_codes = {spec.code for spec in asset_provider_ingestion.DEFEATBETA_DATASET_SPECS}
        dataset_tables = {spec.table_name for spec in asset_provider_ingestion.DEFEATBETA_DATASET_SPECS}

        self.assertNotIn("stock_prices", dataset_codes)
        self.assertNotIn("defeatbeta_price_history", dataset_tables)

    def test_defeatbeta_sync_marks_run_cancelled_before_connect(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "catalog.sqlite"
            with mock.patch.object(asset_provider_ingestion, "defeatbeta_available", return_value=True):
                with mock.patch.object(
                    asset_provider_ingestion,
                    "_connect_defeatbeta_duckdb",
                    side_effect=AssertionError("DuckDB should not be opened when sync is cancelled early."),
                ):
                    with self.assertRaises(InterruptedError):
                        asset_provider_ingestion.sync_defeatbeta_assets(
                            catalog_path=db_path,
                            symbols=["AAPL"],
                            stop_requested=lambda: True,
                        )

            with sqlite3.connect(db_path) as conn:
                rows = conn.execute(
                    "SELECT provider, status, asset_count, record_count FROM provider_sync_runs"
                ).fetchall()

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][0], asset_provider_ingestion.DEFEATBETA_PROVIDER)
            self.assertEqual(rows[0][1], "cancelled")
            self.assertEqual(rows[0][2], 0)
            self.assertEqual(rows[0][3], 0)

    def test_financedatabase_import_can_be_cancelled_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "catalog.sqlite"
            fake_module = SimpleNamespace()
            with mock.patch.object(asset_reference, "financedatabase_available", return_value=True):
                with mock.patch.dict(sys.modules, {"financedatabase": fake_module}):
                    with mock.patch.object(
                        asset_reference,
                        "_load_financedatabase_dataset",
                        side_effect=AssertionError("FinanceDatabase datasets should not load when import is cancelled."),
                    ):
                        with self.assertRaises(InterruptedError):
                            asset_reference.import_financedatabase_assets(
                                catalog_path=db_path,
                                stop_requested=lambda: True,
                            )

    def test_defeatbeta_news_insert_accepts_list_payload_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "catalog.sqlite"
            catalog = ResultCatalog(db_path)
            row = asset_provider_ingestion._frame_to_records(
                pd.DataFrame(
                    [
                        {
                            "symbol": "AAPL",
                            "title": "Apple headline",
                            "report_date": "2026-04-12",
                            "related_symbols": ["AAPL", "MSFT"],
                        }
                    ]
                )
            )[0]

            self.assertEqual(row["related_symbols"], ["AAPL", "MSFT"])

            with catalog.connect() as conn:
                asset_provider_ingestion._defeatbeta_insert_record(
                    conn,
                    table_name="defeatbeta_news_articles",
                    record_key="news_test_key",
                    asset_id=ResultCatalog._asset_id_from_symbol("AAPL"),
                    symbol="AAPL",
                    row=row,
                    raw_payload_id=None,
                )
                stored = conn.execute(
                    """
                    SELECT headline, payload_json
                    FROM defeatbeta_news_articles
                    WHERE record_key=?
                    """,
                    ("news_test_key",),
                ).fetchone()

            self.assertIsNotNone(stored)
            self.assertEqual(stored[0], "Apple headline")
            payload = json.loads(stored[1])
            self.assertEqual(payload["related_symbols"], ["AAPL", "MSFT"])


if __name__ == "__main__":
    unittest.main()
