from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

from .catalog import ResultCatalog


PROVIDER_SECRETS_PATH = Path("data") / "provider_secrets.json"


def _normalize_provider_id(provider_id: str | None) -> str:
    return str(provider_id or "").strip().lower()


def _as_result_catalog(catalog: object | None) -> ResultCatalog:
    if isinstance(catalog, ResultCatalog):
        return catalog
    db_path = getattr(catalog, "db_path", None) if catalog is not None else None
    return ResultCatalog(db_path or "backtests.sqlite")


def load_provider_settings(
    provider_id: str | None,
    *,
    catalog: object | None = None,
) -> dict:
    normalized = _normalize_provider_id(provider_id)
    if not normalized:
        return {}
    store = _as_result_catalog(catalog)
    record = store.load_provider_settings(normalized)
    if record is None:
        return {}
    try:
        return dict(json.loads(record.settings_json or "{}"))
    except Exception:
        return {}


def save_provider_settings(
    provider_id: str | None,
    settings: dict,
    *,
    catalog: object | None = None,
) -> None:
    normalized = _normalize_provider_id(provider_id)
    if not normalized:
        raise ValueError("Provider id is required.")
    _as_result_catalog(catalog).save_provider_settings(normalized, dict(settings or {}))


def load_provider_secrets(
    provider_id: str | None,
    *,
    secret_path: Path | str = PROVIDER_SECRETS_PATH,
) -> dict:
    normalized = _normalize_provider_id(provider_id)
    if not normalized:
        return {}
    path = Path(secret_path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload.get(normalized) or {})


def save_provider_secrets(
    provider_id: str | None,
    secrets: dict,
    *,
    secret_path: Path | str = PROVIDER_SECRETS_PATH,
) -> None:
    normalized = _normalize_provider_id(provider_id)
    if not normalized:
        raise ValueError("Provider id is required.")
    path = Path(secret_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, dict] = {}
    if path.exists():
        try:
            payload = dict(json.loads(path.read_text(encoding="utf-8")) or {})
        except Exception:
            payload = {}
    cleaned = {
        str(key).strip(): str(value).strip()
        for key, value in dict(secrets or {}).items()
        if str(key).strip() and str(value).strip()
    }
    if cleaned:
        payload[normalized] = cleaned
    else:
        payload.pop(normalized, None)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass


def build_provider_runtime_environment(
    provider_id: str | None,
    *,
    catalog: object | None = None,
    secret_path: Path | str = PROVIDER_SECRETS_PATH,
) -> dict[str, str]:
    normalized = _normalize_provider_id(provider_id)
    settings = load_provider_settings(normalized, catalog=catalog)
    secrets = load_provider_secrets(normalized, secret_path=secret_path)
    env: dict[str, str] = {}
    if normalized == "massive":
        api_key = str(secrets.get("api_key") or "").strip()
        if api_key:
            env["MASSIVE_API_KEY"] = api_key
    elif normalized == "interactive_brokers":
        host = str(settings.get("host") or "").strip()
        port = str(settings.get("port") or "").strip()
        client_id = str(settings.get("client_id") or "").strip()
        if host:
            env["IB_HOST"] = host
        if port:
            env["IB_PORT"] = port
        if client_id:
            env["IB_CLIENT_ID"] = client_id
    return env


def provider_settings_status(
    provider_id: str | None,
    *,
    catalog: object | None = None,
    secret_path: Path | str = PROVIDER_SECRETS_PATH,
) -> str:
    normalized = _normalize_provider_id(provider_id)
    settings = load_provider_settings(normalized, catalog=catalog)
    secrets = load_provider_secrets(normalized, secret_path=secret_path)
    if normalized == "massive":
        return "API key saved" if str(secrets.get("api_key") or "").strip() else "API key missing"
    if normalized == "interactive_brokers":
        host = str(settings.get("host") or "").strip()
        port = str(settings.get("port") or "").strip()
        client_id = str(settings.get("client_id") or "").strip()
        if host and port and client_id:
            return f"{host}:{port} client {client_id}"
        return "Host / port / client_id incomplete"
    return "No additional settings required"
