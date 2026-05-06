from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from backtest_engine.catalog import ResultCatalog


class ResultCatalogSchemaTests(unittest.TestCase):
    def test_fresh_catalog_creates_trades_before_trade_migrations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "fresh.sqlite"
            ResultCatalog(db_path)
            with sqlite3.connect(db_path) as conn:
                columns = {row[1] for row in conn.execute("PRAGMA table_info(trades)").fetchall()}
            self.assertIn("run_id", columns)
            self.assertIn("dataset_id", columns)
            self.assertIn("source_dataset_id", columns)
            self.assertIn("strategy_block_id", columns)


if __name__ == "__main__":
    unittest.main()
