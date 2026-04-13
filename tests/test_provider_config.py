from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backtest_engine.catalog import ResultCatalog
from backtest_engine.provider_config import (
    build_provider_runtime_environment,
    load_provider_settings,
    provider_settings_status,
    save_provider_secrets,
    save_provider_settings,
)


class ProviderConfigTests(unittest.TestCase):
    def test_alpaca_settings_round_trip_and_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "catalog.sqlite"
            secret_path = Path(tmpdir) / "provider_secrets.json"
            catalog = ResultCatalog(db_path)
            save_provider_settings(
                "alpaca",
                {"base_url": "https://paper-api.alpaca.markets", "data_url": "https://data.alpaca.markets"},
                catalog=catalog,
            )
            save_provider_secrets(
                "alpaca",
                {"api_key": "alpaca-key", "secret_key": "alpaca-secret"},
                secret_path=secret_path,
            )
            env = build_provider_runtime_environment("alpaca", catalog=catalog, secret_path=secret_path)
            self.assertEqual(env["ALPACA_API_KEY_ID"], "alpaca-key")
            self.assertEqual(env["ALPACA_API_SECRET_KEY"], "alpaca-secret")
            self.assertEqual(env["ALPACA_BASE_URL"], "https://paper-api.alpaca.markets")
            self.assertEqual(env["ALPACA_DATA_URL"], "https://data.alpaca.markets")
            self.assertEqual(
                provider_settings_status("alpaca", catalog=catalog, secret_path=secret_path),
                "Configured (https://paper-api.alpaca.markets)",
            )

    def test_interactive_brokers_settings_round_trip_and_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "catalog.sqlite"
            catalog = ResultCatalog(db_path)
            save_provider_settings(
                "interactive_brokers",
                {"host": "127.0.0.1", "port": 7497, "client_id": 9301},
                catalog=catalog,
            )
            loaded = load_provider_settings("interactive_brokers", catalog=catalog)
            self.assertEqual(loaded["host"], "127.0.0.1")
            env = build_provider_runtime_environment("interactive_brokers", catalog=catalog)
            self.assertEqual(env["IB_HOST"], "127.0.0.1")
            self.assertEqual(env["IB_PORT"], "7497")
            self.assertEqual(env["IB_CLIENT_ID"], "9301")
            self.assertEqual(provider_settings_status("interactive_brokers", catalog=catalog), "127.0.0.1:7497 client 9301")
            override_env = build_provider_runtime_environment(
                "interactive_brokers",
                catalog=catalog,
                client_id_override=9304,
            )
            self.assertEqual(override_env["IB_CLIENT_ID"], "9304")

    def test_massive_secret_round_trip_and_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "catalog.sqlite"
            secret_path = Path(tmpdir) / "provider_secrets.json"
            catalog = ResultCatalog(db_path)
            save_provider_secrets("massive", {"api_key": "secret-key"}, secret_path=secret_path)
            env = build_provider_runtime_environment("massive", catalog=catalog, secret_path=secret_path)
            self.assertEqual(env["MASSIVE_API_KEY"], "secret-key")
            self.assertEqual(provider_settings_status("massive", catalog=catalog, secret_path=secret_path), "API key saved")


if __name__ == "__main__":
    unittest.main()
