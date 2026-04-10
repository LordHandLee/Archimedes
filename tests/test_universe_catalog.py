from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from backtest_engine.catalog import ResultCatalog


class UniverseCatalogTests(unittest.TestCase):
    def test_save_and_load_universe_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "universes.sqlite"
            catalog = ResultCatalog(db_path)
            catalog.save_universe(
                universe_id="universe_core",
                name="Core Universe",
                description="Main watchlist",
                symbols=["spy", "qqq", "AAPL"],
                dataset_ids=["SPY_1m", "QQQ_1m"],
                source_preference="massive",
            )

            rows = catalog.load_universes()
            self.assertEqual(len(rows), 1)
            record = rows[0]
            self.assertEqual(record.universe_id, "universe_core")
            self.assertEqual(record.name, "Core Universe")
            self.assertEqual(record.description, "Main watchlist")
            self.assertEqual(json.loads(record.symbols_json), ["AAPL", "QQQ", "SPY"])
            self.assertEqual(json.loads(record.dataset_ids_json), ["QQQ_1m", "SPY_1m"])
            self.assertEqual(record.source_preference, "massive")

    def test_save_universe_requires_name_and_members(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "universes.sqlite"
            catalog = ResultCatalog(db_path)
            with self.assertRaises(ValueError):
                catalog.save_universe(universe_id="u1", name=" ", symbols=[], dataset_ids=[])


if __name__ == "__main__":
    unittest.main()
