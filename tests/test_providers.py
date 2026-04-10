from __future__ import annotations

import unittest

from backtest_engine.providers import (
    DEFAULT_ACQUISITION_PROVIDER,
    available_acquisition_providers,
    build_provider_fetch_command,
    get_acquisition_provider,
    provider_display_name,
    provider_fetch_tuning,
    resolve_acquisition_source,
)


class AcquisitionProviderTests(unittest.TestCase):
    def test_default_provider_is_registered(self) -> None:
        providers = {spec.provider_id for spec in available_acquisition_providers()}
        self.assertIn(DEFAULT_ACQUISITION_PROVIDER, providers)
        self.assertIn("interactive_brokers", providers)
        self.assertIn("stooq", providers)
        spec = get_acquisition_provider(DEFAULT_ACQUISITION_PROVIDER)
        self.assertEqual(spec.provider_id, "massive")
        self.assertEqual(spec.fetch_script_relpath, "scripts/fetch_massive.py")

        ib_spec = get_acquisition_provider("interactive_brokers")
        self.assertEqual(ib_spec.label, "Interactive Brokers")
        self.assertEqual(ib_spec.fetch_script_relpath, "scripts/fetch_interactive_brokers.py")

    def test_resolve_source_prefers_explicit_over_universe_preference(self) -> None:
        self.assertEqual(resolve_acquisition_source("massive", ""), "massive")
        self.assertEqual(resolve_acquisition_source("", "massive"), "massive")
        self.assertEqual(resolve_acquisition_source("", ""), DEFAULT_ACQUISITION_PROVIDER)

    def test_build_provider_fetch_command_uses_registered_fetch_script(self) -> None:
        cmd = build_provider_fetch_command(
            "massive",
            python_executable="/usr/bin/python3",
            ticker="spy",
            out_path="data/SPY_massive_2y_1m.csv",
            resolution="1m",
            progress=True,
            resume=True,
        )
        self.assertEqual(cmd[0], "/usr/bin/python3")
        self.assertEqual(cmd[1], "scripts/fetch_massive.py")
        self.assertIn("SPY", cmd)
        self.assertIn("--progress", cmd)
        self.assertIn("--resume", cmd)
        self.assertIn("--pace", cmd)
        self.assertIn("12.5", cmd)

        stooq_cmd = build_provider_fetch_command(
            "stooq",
            python_executable="/usr/bin/python3",
            ticker="spy",
            out_path="data/SPY_stooq_max_1d.csv",
        )
        self.assertEqual(stooq_cmd[1], "scripts/fetch_stooq.py")

        ib_cmd = build_provider_fetch_command(
            "interactive_brokers",
            python_executable="/usr/bin/python3",
            ticker="spy",
            out_path="data/SPY_interactive_brokers_10y_1m.csv",
            resolution="1m",
            progress=True,
        )
        self.assertEqual(ib_cmd[1], "scripts/fetch_interactive_brokers.py")
        self.assertIn("--progress", ib_cmd)
        self.assertIn("--pace", ib_cmd)
        self.assertIn("3.0", ib_cmd)
        self.assertIn("--chunk-duration", ib_cmd)
        self.assertIn("1w", ib_cmd)

    def test_interactive_brokers_minute_fetch_tuning_prefers_weekly_chunks(self) -> None:
        tuning = provider_fetch_tuning("interactive_brokers", resolution="1 min")
        self.assertEqual(tuning.pace_seconds, "3.0")
        self.assertEqual(tuning.default_chunk_duration, "1w")

    def test_explicit_chunk_duration_overrides_interactive_brokers_default(self) -> None:
        cmd = build_provider_fetch_command(
            "interactive_brokers",
            python_executable="/usr/bin/python3",
            ticker="spy",
            out_path="data/SPY_interactive_brokers_10y_1m.csv",
            resolution="1m",
            extra_args=["--chunk-duration", "2w"],
        )
        self.assertIn("--chunk-duration", cmd)
        first_chunk_duration = cmd[cmd.index("--chunk-duration") + 1]
        self.assertEqual(first_chunk_duration, "2w")
        self.assertEqual(cmd.count("--chunk-duration"), 1)

    def test_provider_display_name_handles_unknown_values(self) -> None:
        self.assertEqual(provider_display_name("massive"), "Massive")
        self.assertEqual(provider_display_name("interactive_brokers"), "Interactive Brokers")
        self.assertEqual(provider_display_name("custom_feed"), "Custom Feed")


if __name__ == "__main__":
    unittest.main()
