from __future__ import annotations

import importlib.util
import socket
import threading
from dataclasses import dataclass
from queue import Empty, Queue


@dataclass(frozen=True)
class PackageValidationStatus:
    ok: bool
    message: str


@dataclass(frozen=True)
class ConnectionValidationStatus:
    ok: bool
    message: str


def ibapi_package_status() -> PackageValidationStatus:
    installed = importlib.util.find_spec("ibapi") is not None
    if installed:
        return PackageValidationStatus(True, "ibapi is installed in the current Python environment.")
    return PackageValidationStatus(
        False,
        "ibapi is not installed in the current Python environment.",
    )


def interactive_brokers_socket_status(host: str, port: int, *, timeout_seconds: float = 2.0) -> ConnectionValidationStatus:
    normalized_host = str(host or "").strip()
    if not normalized_host:
        return ConnectionValidationStatus(False, "Host is missing.")
    if int(port) <= 0:
        return ConnectionValidationStatus(False, "Port must be a positive integer.")
    try:
        with socket.create_connection((normalized_host, int(port)), timeout=timeout_seconds):
            return ConnectionValidationStatus(
                True,
                f"Connection to {normalized_host}:{int(port)} succeeded.",
            )
    except TimeoutError:
        return ConnectionValidationStatus(False, f"Timed out connecting to {normalized_host}:{int(port)}.")
    except OSError as exc:
        return ConnectionValidationStatus(False, f"Could not connect to {normalized_host}:{int(port)}: {exc}")


def ibapi_install_command() -> str:
    return "source .quant_backtest_engine_venv/bin/activate && pip install ibapi"


def _load_ibapi_symbols():
    if importlib.util.find_spec("ibapi") is None:
        raise RuntimeError("ibapi is not installed in the current Python environment.")
    from ibapi.client import EClient
    from ibapi.contract import Contract
    from ibapi.wrapper import EWrapper

    return EClient, EWrapper, Contract


def interactive_brokers_api_status(
    host: str,
    port: int,
    client_id: int,
    *,
    timeout_seconds: float = 5.0,
) -> ConnectionValidationStatus:
    normalized_host = str(host or "").strip()
    if not normalized_host:
        return ConnectionValidationStatus(False, "Host is missing.")
    if int(port) <= 0:
        return ConnectionValidationStatus(False, "Port must be a positive integer.")
    if int(client_id) <= 0:
        return ConnectionValidationStatus(False, "Client ID must be a positive integer.")
    try:
        EClient, EWrapper, _ = _load_ibapi_symbols()
    except Exception as exc:
        return ConnectionValidationStatus(False, str(exc))

    class _HandshakeApp(EWrapper, EClient):
        def __init__(self) -> None:
            EWrapper.__init__(self)
            EClient.__init__(self, self)
            self.connected = threading.Event()
            self.failure: str | None = None

        def nextValidId(self, orderId: int) -> None:  # noqa: N802
            self.connected.set()

        def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):  # noqa: N802
            if int(errorCode) not in {2104, 2106, 2158}:
                self.failure = f"Interactive Brokers error {errorCode}: {errorString}"
                self.connected.set()

    app = _HandshakeApp()
    try:
        app.connect(normalized_host, int(port), int(client_id))
        thread = threading.Thread(target=app.run, daemon=True)
        thread.start()
        if not app.connected.wait(timeout_seconds):
            return ConnectionValidationStatus(
                False,
                f"Timed out waiting for Interactive Brokers API handshake at {normalized_host}:{int(port)}.",
            )
        if app.failure:
            return ConnectionValidationStatus(False, app.failure)
        return ConnectionValidationStatus(
            True,
            f"Interactive Brokers API handshake succeeded at {normalized_host}:{int(port)} with client_id {int(client_id)}.",
        )
    except Exception as exc:
        message = str(exc)
        if "NoneType" in message and "connect" in message:
            message = "Interactive Brokers API client is unavailable. Ensure ibapi is installed correctly."
        return ConnectionValidationStatus(
            False,
            f"Interactive Brokers API connection failed for {normalized_host}:{int(port)}: {message}",
        )
    finally:
        try:
            app.disconnect()
        except Exception:
            pass


def interactive_brokers_head_timestamp_status(
    host: str,
    port: int,
    client_id: int,
    *,
    symbol: str = "AAPL",
    sec_type: str = "STK",
    exchange: str = "SMART",
    currency: str = "USD",
    primary_exchange: str = "",
    what_to_show: str = "TRADES",
    use_rth: bool = False,
    timeout_seconds: float = 8.0,
) -> ConnectionValidationStatus:
    normalized_host = str(host or "").strip()
    normalized_symbol = str(symbol or "").strip().upper()
    if not normalized_symbol:
        return ConnectionValidationStatus(False, "Probe symbol is missing.")
    base_status = interactive_brokers_api_status(
        normalized_host,
        port,
        client_id,
        timeout_seconds=timeout_seconds,
    )
    if not base_status.ok:
        return base_status
    try:
        EClient, EWrapper, Contract = _load_ibapi_symbols()
    except Exception as exc:
        return ConnectionValidationStatus(False, str(exc))

    class _HeadTimestampApp(EWrapper, EClient):
        def __init__(self) -> None:
            EWrapper.__init__(self)
            EClient.__init__(self, self)
            self.connected = threading.Event()
            self.failure: str | None = None
            self.response: Queue = Queue()

        def nextValidId(self, orderId: int) -> None:  # noqa: N802
            self.connected.set()

        def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):  # noqa: N802
            if int(errorCode) not in {2104, 2106, 2158}:
                message = f"Interactive Brokers error {errorCode}: {errorString}"
                if reqId is not None and int(reqId) >= 0:
                    self.response.put(("error", message))
                else:
                    self.failure = message
                    self.connected.set()

        def headTimestamp(self, reqId, headTimestamp):  # noqa: N802
            self.response.put(("head", str(headTimestamp)))

    app = _HeadTimestampApp()
    try:
        app.connect(normalized_host, int(port), int(client_id))
        thread = threading.Thread(target=app.run, daemon=True)
        thread.start()
        if not app.connected.wait(timeout_seconds):
            return ConnectionValidationStatus(
                False,
                f"Timed out waiting for Interactive Brokers API handshake at {normalized_host}:{int(port)}.",
            )
        if app.failure:
            return ConnectionValidationStatus(False, app.failure)

        contract = Contract()
        contract.symbol = normalized_symbol
        contract.secType = str(sec_type).strip().upper()
        contract.exchange = str(exchange).strip().upper()
        contract.currency = str(currency).strip().upper()
        if primary_exchange:
            contract.primaryExchange = str(primary_exchange).strip().upper()

        req_id = 19001
        app.reqHeadTimeStamp(req_id, contract, str(what_to_show).strip().upper(), int(use_rth), 2)
        try:
            kind, payload = app.response.get(timeout=timeout_seconds)
        except Empty:
            return ConnectionValidationStatus(
                False,
                f"Timed out waiting for head timestamp for {normalized_symbol}.",
            )
        finally:
            try:
                app.cancelHeadTimeStamp(req_id)
            except Exception:
                pass
        if kind == "error":
            return ConnectionValidationStatus(False, payload)
        return ConnectionValidationStatus(
            True,
            f"Head timestamp probe for {normalized_symbol} succeeded: {payload}",
        )
    except Exception as exc:
        message = str(exc)
        if "NoneType" in message and "connect" in message:
            message = "Interactive Brokers API client is unavailable. Ensure ibapi is installed correctly."
        return ConnectionValidationStatus(
            False,
            f"Interactive Brokers head timestamp probe failed for {normalized_symbol}: {message}",
        )
    finally:
        try:
            app.disconnect()
        except Exception:
            pass
