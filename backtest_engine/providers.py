from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_ACQUISITION_PROVIDER = "massive"


@dataclass(frozen=True)
class AcquisitionProviderSpec:
    provider_id: str
    label: str
    fetch_script_relpath: str
    default_resolution: str = "1m"
    default_history_window: str = "2y"
    description: str = ""


@dataclass(frozen=True)
class AcquisitionProviderFetchTuning:
    pace_seconds: str | None = None
    default_chunk_duration: str | None = None


_PROVIDER_REGISTRY: dict[str, AcquisitionProviderSpec] = {
    "interactive_brokers": AcquisitionProviderSpec(
        provider_id="interactive_brokers",
        label="Interactive Brokers",
        fetch_script_relpath="scripts/fetch_interactive_brokers.py",
        default_resolution="1m",
        default_history_window="10y",
        description=(
            "Historical bar downloader backed by the official Interactive Brokers TWS/Gateway API. "
            "Uses conservative chunked requests for minute history; actual earliest coverage varies by contract."
        ),
    ),
    "massive": AcquisitionProviderSpec(
        provider_id="massive",
        label="Massive",
        fetch_script_relpath="scripts/fetch_massive.py",
        default_resolution="1m",
        default_history_window="2y",
        description="Polygon/Massive minute-bar downloader currently used for interactive and scheduled acquisition.",
    ),
    "stooq": AcquisitionProviderSpec(
        provider_id="stooq",
        label="Stooq",
        fetch_script_relpath="scripts/fetch_stooq.py",
        default_resolution="1d",
        default_history_window="max",
        description="Daily EOD downloader backed by the public Stooq CSV endpoint.",
    ),
}


def available_acquisition_providers() -> tuple[AcquisitionProviderSpec, ...]:
    return tuple(_PROVIDER_REGISTRY[key] for key in sorted(_PROVIDER_REGISTRY))


def get_acquisition_provider(provider_id: str | None) -> AcquisitionProviderSpec:
    normalized = str(provider_id or "").strip().lower() or DEFAULT_ACQUISITION_PROVIDER
    if normalized not in _PROVIDER_REGISTRY:
        raise ValueError(f"Unknown acquisition provider '{provider_id}'.")
    return _PROVIDER_REGISTRY[normalized]


def provider_display_name(provider_id: str | None) -> str:
    try:
        return get_acquisition_provider(provider_id).label
    except Exception:
        return str(provider_id or DEFAULT_ACQUISITION_PROVIDER).replace("_", " ").title()


def resolve_acquisition_source(
    explicit_source: str | None = None,
    preferred_source: str | None = None,
    *,
    default_source: str = DEFAULT_ACQUISITION_PROVIDER,
) -> str:
    for candidate in (explicit_source, preferred_source, default_source):
        normalized = str(candidate or "").strip().lower()
        if normalized and normalized in _PROVIDER_REGISTRY:
            return normalized
    return default_source


def provider_fetch_script_path(provider_id: str | None) -> Path:
    return Path(get_acquisition_provider(provider_id).fetch_script_relpath)


def _normalize_resolution_alias(value: str | None) -> str:
    text = " ".join(str(value or "").strip().lower().split())
    if not text:
        return ""
    parts = text.split()
    if len(parts) == 2 and parts[0].isdigit():
        number, unit = parts
        if unit in {"min", "mins", "minute", "minutes"}:
            return f"{number}m"
        if unit in {"hour", "hours"}:
            return f"{number}h"
        if unit in {"day", "days"}:
            return f"{number}d"
        if unit in {"week", "weeks"}:
            return f"{number}w"
        if unit in {"month", "months"}:
            return f"{number}mo"
    compact = text.replace(" ", "")
    if compact.endswith("min"):
        return compact[:-3] + "m"
    if compact.endswith("mins"):
        return compact[:-4] + "m"
    if compact.endswith("minute"):
        return compact[:-6] + "m"
    if compact.endswith("minutes"):
        return compact[:-7] + "m"
    return compact


def provider_fetch_tuning(
    provider_id: str | None,
    *,
    resolution: str | None = None,
    history_window: str | None = None,
) -> AcquisitionProviderFetchTuning:
    normalized_provider = get_acquisition_provider(provider_id).provider_id
    normalized_resolution = _normalize_resolution_alias(resolution)
    if normalized_provider == "massive":
        return AcquisitionProviderFetchTuning(pace_seconds="12.5")
    if normalized_provider == "interactive_brokers":
        if normalized_resolution == "1m":
            # Live gateway benchmarking showed a sharp latency knee beyond seven-day
            # 1-minute requests, and also showed 3s pacing outperforming more
            # aggressive 0.5s/1.0s pacing for single-symbol weekly chunks.
            return AcquisitionProviderFetchTuning(pace_seconds="3.0", default_chunk_duration="1w")
        return AcquisitionProviderFetchTuning(pace_seconds="1.0")
    return AcquisitionProviderFetchTuning()


def build_provider_fetch_command(
    provider_id: str | None,
    *,
    python_executable: str,
    ticker: str,
    out_path: str | Path,
    resolution: str | None = None,
    history_window: str | None = None,
    progress: bool = False,
    resume: bool = False,
    pace_seconds: str | None = None,
    extra_args: Sequence[str] | None = None,
) -> list[str]:
    spec = get_acquisition_provider(provider_id)
    tuning = provider_fetch_tuning(spec.provider_id, resolution=resolution, history_window=history_window)
    cmd = [
        python_executable,
        str(Path(spec.fetch_script_relpath)),
        str(ticker).strip().upper(),
        "--out",
        str(out_path),
    ]
    if progress:
        cmd.append("--progress")
    if resume:
        cmd.append("--resume")
    effective_pace = str(pace_seconds) if pace_seconds is not None else tuning.pace_seconds
    if effective_pace:
        cmd.extend(["--pace", str(effective_pace)])
    explicit_args = [str(arg) for arg in (extra_args or [])]
    if tuning.default_chunk_duration and "--chunk-duration" not in explicit_args:
        cmd.extend(["--chunk-duration", tuning.default_chunk_duration])
    if explicit_args:
        cmd.extend(explicit_args)
    return cmd
