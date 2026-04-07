from __future__ import annotations

import json
import os
import time
from pathlib import Path

from PyQt6 import QtCore, QtNetwork


DEFAULT_MAGELLAN_SERVER_NAME = "MagellanChartViewer"


class MagellanError(RuntimeError):
    """Raised when the dashboard cannot launch or reach the Magellan viewer."""


class MagellanClient(QtCore.QObject):
    def __init__(
        self,
        parent: QtCore.QObject | None = None,
        viewer_path: str | Path | None = None,
        server_name: str = DEFAULT_MAGELLAN_SERVER_NAME,
    ) -> None:
        super().__init__(parent)
        self.server_name = server_name
        self.viewer_path = self._resolve_viewer_path(viewer_path)
        self._process: QtCore.QProcess | None = None
        self._owns_process = False
        self._last_error = ""

    @property
    def last_error(self) -> str:
        return self._last_error

    @property
    def owns_process(self) -> bool:
        return self._owns_process

    def is_running(self, timeout_ms: int = 200) -> bool:
        socket = QtNetwork.QLocalSocket()
        socket.connectToServer(self.server_name, QtCore.QIODevice.OpenModeFlag.WriteOnly)
        connected = socket.waitForConnected(timeout_ms)
        if connected:
            socket.disconnectFromServer()
            if socket.state() != QtNetwork.QLocalSocket.LocalSocketState.UnconnectedState:
                socket.waitForDisconnected(50)
        return connected

    def ensure_running(self, timeout_ms: int = 5000) -> bool:
        if self.is_running():
            self._last_error = ""
            return True

        if self._process and self._process.state() != QtCore.QProcess.ProcessState.NotRunning:
            if self._wait_for_server(timeout_ms):
                self._last_error = ""
                return True

        viewer_path = self.viewer_path
        if viewer_path is None:
            self._last_error = (
                "Magellan viewer binary was not found. "
                "Set MAGELLAN_VIEWER_PATH or place the build at ~/Magellan/charting_engine/build/magellan_chart_viewer."
            )
            return False

        process = QtCore.QProcess(self)
        process.setProgram(str(viewer_path))
        process.setArguments([])
        process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        process.start()
        if not process.waitForStarted(min(timeout_ms, 1500)):
            error_text = process.errorString().strip()
            if not error_text:
                error_text = f"Unable to start Magellan at {viewer_path}."
            self._last_error = error_text
            process.deleteLater()
            return False

        self._process = process
        self._owns_process = True
        if self._wait_for_server(timeout_ms):
            self._last_error = ""
            return True

        error_text = self._read_process_output().strip()
        if not error_text:
            error_text = f"Magellan started but did not expose the {self.server_name} IPC server."
        self._last_error = error_text
        self.shutdown()
        return False

    def open_live_session(
        self,
        session_id: str,
        *,
        title: str = "",
        subtitle: str = "",
        status_text: str = "",
        snapshot_path: str | Path | None = None,
        timeout_ms: int = 1000,
    ) -> None:
        payload = {
            "type": "open_live",
            "session_id": session_id,
            "title": title,
            "subtitle": subtitle,
            "status_text": status_text,
        }
        if snapshot_path:
            payload["snapshot_path"] = str(snapshot_path)
        self._send_json(payload, timeout_ms=timeout_ms)

    def open_snapshot(self, snapshot_path: str | Path, timeout_ms: int = 1000) -> None:
        path_text = str(snapshot_path).strip()
        if not path_text:
            raise MagellanError("Snapshot path is required.")
        self._send_message(path_text.encode("utf-8"), timeout_ms=timeout_ms)

    def send_live_update(
        self,
        session_id: str,
        *,
        title: str = "",
        subtitle: str = "",
        status_text: str = "",
        bars: list[dict] | None = None,
        overlay_series: list[dict] | None = None,
        pane_series: list[dict] | None = None,
        equity_series: list[dict] | None = None,
        trade_markers: list[dict] | None = None,
        timeout_ms: int = 1000,
    ) -> None:
        payload = {
            "type": "live_update",
            "session_id": session_id,
            "title": title,
            "subtitle": subtitle,
            "status_text": status_text,
            "bars": bars or [],
            "overlay_series": overlay_series or [],
            "pane_series": pane_series or [],
            "equity_series": equity_series or [],
            "trade_markers": trade_markers or [],
        }
        self._send_json(payload, timeout_ms=timeout_ms)

    def shutdown(self, timeout_ms: int = 2000) -> None:
        if not self._owns_process or not self._process:
            return

        process = self._process
        if process.state() == QtCore.QProcess.ProcessState.NotRunning:
            self._process = None
            self._owns_process = False
            return

        process.terminate()
        if not process.waitForFinished(timeout_ms):
            process.kill()
            process.waitForFinished(timeout_ms)

        self._process = None
        self._owns_process = False

    def _send_json(self, payload: dict, timeout_ms: int = 1000) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._send_message(encoded, timeout_ms=timeout_ms)

    def _send_message(self, payload: bytes, timeout_ms: int = 1000) -> None:
        if not payload.strip():
            raise MagellanError("Refusing to send an empty Magellan IPC payload.")
        if not self.ensure_running():
            raise MagellanError(self._last_error or "Magellan is not running.")

        socket = QtNetwork.QLocalSocket()
        socket.connectToServer(self.server_name, QtCore.QIODevice.OpenModeFlag.WriteOnly)
        if not socket.waitForConnected(timeout_ms):
            self._last_error = f"Unable to connect to Magellan IPC server {self.server_name}."
            raise MagellanError(self._last_error)

        socket.write(payload)
        socket.write(b"\n")
        if not socket.waitForBytesWritten(timeout_ms):
            self._last_error = "Timed out while sending data to Magellan."
            socket.disconnectFromServer()
            raise MagellanError(self._last_error)

        socket.disconnectFromServer()
        if socket.state() != QtNetwork.QLocalSocket.LocalSocketState.UnconnectedState:
            socket.waitForDisconnected(50)

    def _wait_for_server(self, timeout_ms: int) -> bool:
        deadline = time.monotonic() + max(timeout_ms, 100) / 1000.0
        while time.monotonic() < deadline:
            if self.is_running(timeout_ms=100):
                return True
            QtCore.QCoreApplication.processEvents()
            QtCore.QThread.msleep(50)
        return False

    def _read_process_output(self) -> str:
        if not self._process:
            return ""
        data = bytes(self._process.readAllStandardOutput())
        return data.decode("utf-8", errors="ignore")

    @staticmethod
    def _resolve_viewer_path(viewer_path: str | Path | None) -> Path | None:
        candidates: list[Path] = []
        if viewer_path:
            candidates.append(Path(viewer_path).expanduser())

        env_viewer_path = os.environ.get("MAGELLAN_VIEWER_PATH", "").strip()
        if env_viewer_path:
            candidates.append(Path(env_viewer_path).expanduser())

        env_root = os.environ.get("MAGELLAN_ROOT", "").strip()
        if env_root:
            candidates.append(Path(env_root).expanduser() / "charting_engine" / "build" / "magellan_chart_viewer")

        repo_sibling = Path(__file__).resolve().parents[2] / "Magellan" / "charting_engine" / "build" / "magellan_chart_viewer"
        home_build = Path.home() / "Magellan" / "charting_engine" / "build" / "magellan_chart_viewer"
        candidates.extend([repo_sibling, home_build])

        for candidate in candidates:
            candidate = candidate.expanduser()
            if candidate.exists():
                return candidate
        return None
