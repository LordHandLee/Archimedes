import tempfile
import unittest
from pathlib import Path

import pandas as pd

import scripts.fetch_massive as fetch_massive
from scripts.fetch_massive import merge_existing_output


class MassiveFetchOutputMergeTests(unittest.TestCase):
    def test_merge_existing_output_keeps_prior_windows_and_replaces_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "SOXL_massive_2y_1m.csv"
            existing = pd.DataFrame(
                [
                    {"timestamp": "2026-04-21T13:30:00Z", "open": 1, "high": 2, "low": 1, "close": 2, "volume": 10},
                    {"timestamp": "2026-04-21T13:31:00Z", "open": 2, "high": 3, "low": 2, "close": 3, "volume": 20},
                ]
            )
            existing.to_csv(out_path, index=False)

            current = pd.DataFrame(
                [
                    {"timestamp": pd.Timestamp("2026-04-21T13:31:00Z"), "open": 4, "high": 5, "low": 4, "close": 5, "volume": 40},
                    {"timestamp": pd.Timestamp("2026-04-21T13:32:00Z"), "open": 5, "high": 6, "low": 5, "close": 6, "volume": 50},
                ]
            ).set_index("timestamp")

            merged = merge_existing_output(out_path, current)

        self.assertEqual(len(merged), 3)
        self.assertEqual(float(merged.iloc[1]["close"]), 5.0)
        self.assertEqual(float(merged.iloc[2]["close"]), 6.0)

    def test_fetch_page_honors_429_retry_after(self) -> None:
        calls = []
        sleeps = []

        class FakeResponse:
            def __init__(self, status_code: int, payload: dict | None = None, retry_after: str = "") -> None:
                self.status_code = status_code
                self.headers = {"Retry-After": retry_after} if retry_after else {}
                self._payload = payload or {}

            def raise_for_status(self) -> None:
                if self.status_code >= 400:
                    raise RuntimeError(f"HTTP {self.status_code}")

            def json(self) -> dict:
                return self._payload

        responses = [
            FakeResponse(429, retry_after="3"),
            FakeResponse(200, {"results": []}),
        ]

        original_requests = fetch_massive.requests
        original_sleep = fetch_massive.time.sleep

        class FakeRequests:
            @staticmethod
            def get(url, headers=None, timeout=30):
                calls.append((url, headers, timeout))
                return responses.pop(0)

        try:
            fetch_massive.requests = FakeRequests
            fetch_massive.time.sleep = lambda seconds: sleeps.append(seconds)
            payload = fetch_massive._fetch_page(
                "https://example.invalid",
                "key",
                max_429_retries=2,
                rate_limit_backoff_seconds=1,
                rate_limit_max_sleep_seconds=10,
            )
        finally:
            fetch_massive.requests = original_requests
            fetch_massive.time.sleep = original_sleep

        self.assertEqual(payload, {"results": []})
        self.assertEqual(len(calls), 2)
        self.assertEqual(sleeps, [3.0])

    def test_fetch_page_retries_malformed_json(self) -> None:
        calls = []
        sleeps = []

        class FakeResponse:
            status_code = 200
            headers = {"Content-Length": "123"}
            content = b'{"results": ['

            def __init__(self, payload: dict | None = None, json_error: Exception | None = None) -> None:
                self._payload = payload or {"results": []}
                self._json_error = json_error

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                if self._json_error is not None:
                    raise self._json_error
                return self._payload

        responses = [
            FakeResponse(json_error=ValueError("bad json")),
            FakeResponse({"results": [{"t": 1}]}),
        ]

        original_requests = fetch_massive.requests
        original_sleep = fetch_massive.time.sleep

        class FakeRequests:
            @staticmethod
            def get(url, headers=None, timeout=30):
                calls.append((url, headers, timeout))
                return responses.pop(0)

        try:
            fetch_massive.requests = FakeRequests
            fetch_massive.time.sleep = lambda seconds: sleeps.append(seconds)
            payload = fetch_massive._fetch_page(
                "https://example.invalid",
                "key",
                max_json_retries=2,
                json_backoff_seconds=1,
                json_max_sleep_seconds=10,
            )
        finally:
            fetch_massive.requests = original_requests
            fetch_massive.time.sleep = original_sleep

        self.assertEqual(payload, {"results": [{"t": 1}]})
        self.assertEqual(len(calls), 2)
        self.assertEqual(sleeps, [1.0])

    def test_fetch_page_reports_persistent_malformed_json(self) -> None:
        class FakeResponse:
            status_code = 200
            headers = {"Content-Length": "123"}
            content = b'{"results": ['

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                raise ValueError("bad json")

        original_requests = fetch_massive.requests
        original_sleep = fetch_massive.time.sleep

        class FakeRequests:
            @staticmethod
            def get(url, headers=None, timeout=30):
                return FakeResponse()

        try:
            fetch_massive.requests = FakeRequests
            fetch_massive.time.sleep = lambda _seconds: None
            with self.assertRaisesRegex(RuntimeError, "invalid JSON"):
                fetch_massive._fetch_page(
                    "https://example.invalid",
                    "key",
                    max_json_retries=1,
                    json_backoff_seconds=1,
                    json_max_sleep_seconds=10,
                )
        finally:
            fetch_massive.requests = original_requests
            fetch_massive.time.sleep = original_sleep


if __name__ == "__main__":
    unittest.main()
