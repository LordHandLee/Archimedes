from __future__ import annotations

import socket
import threading
import unittest

from backtest_engine.provider_validation import (
    ibapi_install_command,
    interactive_brokers_api_status,
    interactive_brokers_head_timestamp_status,
    ibapi_package_status,
    interactive_brokers_socket_status,
)


class ProviderValidationTests(unittest.TestCase):
    def test_ibapi_install_command_mentions_ibapi(self) -> None:
        self.assertIn("pip install ibapi", ibapi_install_command())

    def test_ibapi_package_status_returns_message(self) -> None:
        status = ibapi_package_status()
        self.assertIsInstance(status.ok, bool)
        self.assertIn("ibapi", status.message)

    def test_socket_status_succeeds_for_local_listener(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        host, port = server.getsockname()

        def accept_once() -> None:
            conn, _ = server.accept()
            conn.close()
            server.close()

        thread = threading.Thread(target=accept_once, daemon=True)
        thread.start()
        status = interactive_brokers_socket_status(host, port, timeout_seconds=1.0)
        self.assertTrue(status.ok)
        self.assertIn("succeeded", status.message)
        thread.join(timeout=1.0)

    def test_api_status_without_ibapi_is_clear(self) -> None:
        status = interactive_brokers_api_status("127.0.0.1", 7497, 9301, timeout_seconds=0.1)
        self.assertIsInstance(status.ok, bool)
        if not status.ok:
            self.assertTrue(bool(str(status.message).strip()))

    def test_head_timestamp_status_without_ibapi_is_clear(self) -> None:
        status = interactive_brokers_head_timestamp_status("127.0.0.1", 7497, 9301, symbol="AAPL", timeout_seconds=0.1)
        self.assertIsInstance(status.ok, bool)
        if not status.ok:
            self.assertTrue(
                "ibapi" in status.message.lower()
                or "timed out" in status.message.lower()
                or "error" in status.message.lower()
                or "failed" in status.message.lower()
                or "unavailable" in status.message.lower()
            )


if __name__ == "__main__":
    unittest.main()
