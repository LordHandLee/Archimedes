from __future__ import annotations

import importlib.util
import inspect
import json
import sqlite3
import uuid
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

from .catalog import ResultCatalog


FINANCEDATABASE_PROVIDER = "financedatabase"


@dataclass(frozen=True)
class FinanceDatabaseAssetSpec:
    code: str
    class_name: str
    asset_class: str
    security_type: str
    display_name: str


FINANCEDATABASE_ASSET_SPECS: tuple[FinanceDatabaseAssetSpec, ...] = (
    FinanceDatabaseAssetSpec("equities", "Equities", "Equities", "Equity", "Equities"),
    FinanceDatabaseAssetSpec("etfs", "ETFs", "ETFs", "ETF", "ETFs"),
    FinanceDatabaseAssetSpec("funds", "Funds", "Funds", "Fund", "Funds"),
    FinanceDatabaseAssetSpec("indices", "Indices", "Indices", "Index", "Indices"),
    FinanceDatabaseAssetSpec("currencies", "Currencies", "Forex", "Currency", "Currencies"),
    FinanceDatabaseAssetSpec("cryptos", "Cryptos", "Crypto", "Crypto", "Cryptocurrencies"),
    FinanceDatabaseAssetSpec("moneymarkets", "Moneymarkets", "Money Markets", "Money Market", "Money Markets"),
)


def financedatabase_available() -> bool:
    return importlib.util.find_spec("financedatabase") is not None


def financedatabase_install_hint() -> str:
    return "Install FinanceDatabase with `pip install financedatabase -U` or install from `requirements.txt`."


def _clean_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _clean_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return None


def _clean_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _first_text(*values: object) -> str | None:
    for value in values:
        text = _clean_text(value)
        if text:
            return text
    return None


def _json_ready(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and pd.isna(value):
            return None
        return value
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return str(value)


def _raise_if_stop_requested(stop_requested: Callable[[], bool] | None) -> None:
    if stop_requested is not None and stop_requested():
        raise InterruptedError("FinanceDatabase import cancelled.")


def _normalize_financedatabase_frame(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    if "symbol" not in working.columns:
        working = working.reset_index()
        if "symbol" not in working.columns and len(working.columns) > 0:
            working = working.rename(columns={working.columns[0]: "symbol"})
    working.columns = [str(column).strip() for column in working.columns]
    if "symbol" not in working.columns:
        raise ValueError("FinanceDatabase result did not expose a symbol column.")
    working["symbol"] = working["symbol"].astype(str).str.strip().str.upper()
    working = working.loc[working["symbol"] != ""].copy()
    working = working.drop_duplicates(subset=["symbol"], keep="first")
    return working


def _finance_dataset_columns(frame: pd.DataFrame) -> list[str]:
    return [str(column).strip() for column in frame.columns if str(column).strip()]


def _resolve_financedatabase_specs(asset_classes: Sequence[str] | None) -> list[FinanceDatabaseAssetSpec]:
    if not asset_classes:
        return list(FINANCEDATABASE_ASSET_SPECS)
    wanted = {str(item).strip().lower() for item in list(asset_classes or ()) if str(item).strip()}
    resolved = [spec for spec in FINANCEDATABASE_ASSET_SPECS if spec.code in wanted]
    if not resolved:
        valid = ", ".join(spec.code for spec in FINANCEDATABASE_ASSET_SPECS)
        raise ValueError(f"No valid FinanceDatabase asset classes were selected. Valid values: {valid}")
    return resolved


def _load_financedatabase_dataset(
    fd_module,
    spec: FinanceDatabaseAssetSpec,
    *,
    only_primary_listing: bool,
) -> pd.DataFrame:
    cls = getattr(fd_module, spec.class_name)
    instance = cls()
    select_signature = inspect.signature(instance.select)
    kwargs: dict[str, object] = {}
    if only_primary_listing and "only_primary_listing" in select_signature.parameters:
        kwargs["only_primary_listing"] = True
    frame = instance.select(**kwargs)
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    return _normalize_financedatabase_frame(frame)


def _classification_payload(spec: FinanceDatabaseAssetSpec, row: dict) -> tuple[str | None, str | None, str | None, str | None, str]:
    sector = _first_text(row.get("sector"), row.get("category_group"), row.get("category"))
    industry_group = _first_text(row.get("industry_group"), row.get("category_group"))
    industry = _first_text(row.get("industry"), row.get("category"), row.get("family"))
    theme = _first_text(row.get("category"), row.get("market_cap"))
    tags = {
        "provider": FINANCEDATABASE_PROVIDER,
        "asset_class": spec.asset_class,
        "market": _clean_text(row.get("market")),
        "state": _clean_text(row.get("state")),
        "city": _clean_text(row.get("city")),
        "zipcode": _clean_text(row.get("zipcode")),
        "category_group": _clean_text(row.get("category_group")),
        "category": _clean_text(row.get("category")),
        "family": _clean_text(row.get("family")),
        "market_cap_bucket": _clean_text(row.get("market_cap")),
    }
    return sector, industry_group, industry, theme, json.dumps(tags, sort_keys=True)


def import_financedatabase_assets(
    *,
    catalog_path: str | Path,
    asset_classes: Sequence[str] | None = None,
    only_primary_listing: bool = False,
    limit_per_class: int | None = None,
    store_raw_payloads: bool = False,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> dict:
    if not financedatabase_available():
        raise RuntimeError(f"Missing Python dependency `financedatabase`. {financedatabase_install_hint()}")

    import financedatabase as fd  # type: ignore[import-not-found]

    catalog = ResultCatalog(catalog_path)
    specs = _resolve_financedatabase_specs(asset_classes)
    imported_at = pd.Timestamp.utcnow().isoformat()
    summary: dict[str, object] = {
        "provider": FINANCEDATABASE_PROVIDER,
        "imported_at": imported_at,
        "asset_classes": {},
        "total_rows": 0,
        "total_assets": 0,
    }

    with catalog.connect() as conn:
        total_specs = max(1, len(specs))
        for spec_index, spec in enumerate(specs):
            _raise_if_stop_requested(stop_requested)
            if progress_callback is not None:
                progress_callback(
                    {
                        "provider": FINANCEDATABASE_PROVIDER,
                        "phase": "load_dataset",
                        "label": f"Loading FinanceDatabase {spec.display_name}…",
                        "percent": int((spec_index / total_specs) * 100),
                    }
                )
            frame = _load_financedatabase_dataset(
                fd,
                spec,
                only_primary_listing=only_primary_listing,
            )
            if limit_per_class is not None and int(limit_per_class) > 0:
                frame = frame.head(int(limit_per_class)).copy()

            dataset_code = spec.code
            columns = _finance_dataset_columns(frame)
            conn.execute(
                """
                INSERT INTO provider_dataset_catalog (
                    provider, dataset_code, created_at, updated_at, dataset_group,
                    display_name, description, is_enabled, is_structured,
                    supports_history, supports_incremental_refresh, supports_point_in_time,
                    last_documentation_reviewed_at
                )
                VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?, ?, 1, 1, 0, 1, 0, CURRENT_TIMESTAMP)
                ON CONFLICT(provider, dataset_code) DO UPDATE SET
                    updated_at=CURRENT_TIMESTAMP,
                    dataset_group=excluded.dataset_group,
                    display_name=excluded.display_name,
                    description=excluded.description,
                    is_enabled=excluded.is_enabled,
                    is_structured=excluded.is_structured,
                    supports_incremental_refresh=excluded.supports_incremental_refresh,
                    last_documentation_reviewed_at=CURRENT_TIMESTAMP
                """,
                (
                    FINANCEDATABASE_PROVIDER,
                    dataset_code,
                    "reference_catalog",
                    spec.display_name,
                    f"FinanceDatabase {spec.display_name} reference catalog import.",
                ),
            )

            for column in columns:
                conn.execute(
                    """
                    INSERT INTO provider_field_inventory (
                        provider, dataset_code, field_path, created_at, updated_at, field_name,
                        observed_type, first_seen_at, last_seen_at, mapping_status, notes
                    )
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?)
                    ON CONFLICT(provider, dataset_code, field_path) DO UPDATE SET
                        updated_at=CURRENT_TIMESTAMP,
                        observed_type=excluded.observed_type,
                        last_seen_at=CURRENT_TIMESTAMP
                    """,
                    (
                        FINANCEDATABASE_PROVIDER,
                        dataset_code,
                        column,
                        column,
                        "object",
                        "needs_review",
                        "Observed during FinanceDatabase import.",
                    ),
                )

            asset_rows: list[tuple] = []
            classification_rows: list[tuple] = []
            profile_rows: list[tuple] = []
            identifier_rows: list[tuple] = []
            raw_payload_rows: list[tuple] = []

            total_rows = max(1, len(frame))
            for row_index, (_, series) in enumerate(frame.iterrows(), start=1):
                _raise_if_stop_requested(stop_requested)
                row = {str(key): _json_ready(value) for key, value in series.to_dict().items()}
                symbol = _clean_text(row.get("symbol"))
                if not symbol:
                    continue
                normalized_symbol = str(symbol).upper()
                asset_id = ResultCatalog._asset_id_from_symbol(normalized_symbol)
                sector, industry_group, industry, theme, tags_json = _classification_payload(spec, row)
                raw_profile_json = json.dumps(row, sort_keys=True)

                asset_rows.append(
                    (
                        asset_id,
                        normalized_symbol,
                        normalized_symbol,
                        _first_text(row.get("name"), normalized_symbol),
                        spec.asset_class,
                        spec.security_type,
                        _clean_text(row.get("exchange")),
                        _clean_text(row.get("country")),
                        _clean_text(row.get("currency")),
                    )
                )
                classification_rows.append(
                    (
                        asset_id,
                        sector,
                        industry_group,
                        industry,
                        theme,
                        _clean_text(row.get("market")),
                        None,
                        _clean_text(row.get("country")),
                        _clean_text(row.get("exchange")),
                        _clean_text(row.get("family")),
                        _clean_text(row.get("family")),
                        tags_json,
                    )
                )
                profile_rows.append(
                    (
                        asset_id,
                        _clean_text(row.get("summary")),
                        _clean_text(row.get("website")),
                        None,
                        None,
                        _clean_float(row.get("market_cap")),
                        None,
                        None,
                        None,
                        None,
                        None,
                        _clean_text(row.get("family")),
                        _first_text(row.get("category"), row.get("market_cap")),
                        raw_profile_json,
                        FINANCEDATABASE_PROVIDER,
                        imported_at,
                    )
                )
                identifier_rows.append(
                    (
                        asset_id,
                        FINANCEDATABASE_PROVIDER,
                        normalized_symbol,
                        _clean_text(row.get("exchange")),
                        None,
                        _clean_text(row.get("isin")),
                        _clean_text(row.get("cusip")),
                        _clean_text(row.get("figi")),
                        _clean_text(row.get("composite_figi")),
                        _clean_text(row.get("shareclass_figi")),
                        _clean_text(row.get("composite_figi")),
                        1,
                    )
                )
                if store_raw_payloads:
                    raw_payload_id = f"fd_{uuid.uuid4().hex}"
                    raw_payload_rows.append(
                        (
                            raw_payload_id,
                            FINANCEDATABASE_PROVIDER,
                            dataset_code,
                            asset_id,
                            normalized_symbol,
                            normalized_symbol,
                            raw_profile_json,
                            sha1(raw_profile_json.encode("utf-8")).hexdigest(),
                            imported_at,
                        )
                    )
                if progress_callback is not None and (row_index == total_rows or row_index % 250 == 0):
                    completed = spec_index + (row_index / total_rows)
                    progress_callback(
                        {
                            "provider": FINANCEDATABASE_PROVIDER,
                            "phase": "write_rows",
                            "label": f"Importing {spec.display_name}: {row_index:,} / {total_rows:,}",
                            "percent": min(99, int((completed / total_specs) * 100)),
                        }
                    )

            conn.executemany(
                """
                INSERT INTO asset_master (
                    asset_id, symbol, display_symbol, name, asset_class, security_type,
                    exchange, country, currency, first_seen_at, last_seen_at, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(asset_id) DO UPDATE SET
                    symbol=excluded.symbol,
                    display_symbol=COALESCE(NULLIF(asset_master.display_symbol, ''), excluded.display_symbol),
                    name=COALESCE(NULLIF(asset_master.name, ''), excluded.name),
                    asset_class=COALESCE(NULLIF(asset_master.asset_class, ''), excluded.asset_class),
                    security_type=COALESCE(NULLIF(asset_master.security_type, ''), excluded.security_type),
                    exchange=COALESCE(NULLIF(asset_master.exchange, ''), excluded.exchange),
                    country=COALESCE(NULLIF(asset_master.country, ''), excluded.country),
                    currency=COALESCE(NULLIF(asset_master.currency, ''), excluded.currency),
                    last_seen_at=CURRENT_TIMESTAMP,
                    updated_at=CURRENT_TIMESTAMP
                """,
                asset_rows,
            )
            conn.executemany(
                """
                INSERT INTO asset_classifications (
                    asset_id, updated_at, sector, industry_group, industry, theme, market,
                    region, country, exchange, issuer, brand, tags_json
                )
                VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(asset_id) DO UPDATE SET
                    updated_at=CURRENT_TIMESTAMP,
                    sector=COALESCE(excluded.sector, asset_classifications.sector),
                    industry_group=COALESCE(excluded.industry_group, asset_classifications.industry_group),
                    industry=COALESCE(excluded.industry, asset_classifications.industry),
                    theme=COALESCE(excluded.theme, asset_classifications.theme),
                    market=COALESCE(excluded.market, asset_classifications.market),
                    region=COALESCE(excluded.region, asset_classifications.region),
                    country=COALESCE(excluded.country, asset_classifications.country),
                    exchange=COALESCE(excluded.exchange, asset_classifications.exchange),
                    issuer=COALESCE(excluded.issuer, asset_classifications.issuer),
                    brand=COALESCE(excluded.brand, asset_classifications.brand),
                    tags_json=COALESCE(excluded.tags_json, asset_classifications.tags_json)
                """,
                classification_rows,
            )
            conn.executemany(
                """
                INSERT INTO asset_profiles (
                    asset_id, updated_at, description, website, employee_count, ipo_date,
                    market_cap, shares_outstanding, avg_volume, beta, dividend_yield,
                    expense_ratio, fund_family, category_name, raw_profile_json, source, as_of_date
                )
                VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(asset_id) DO UPDATE SET
                    updated_at=CURRENT_TIMESTAMP,
                    description=COALESCE(excluded.description, asset_profiles.description),
                    website=COALESCE(excluded.website, asset_profiles.website),
                    employee_count=COALESCE(excluded.employee_count, asset_profiles.employee_count),
                    ipo_date=COALESCE(excluded.ipo_date, asset_profiles.ipo_date),
                    market_cap=COALESCE(excluded.market_cap, asset_profiles.market_cap),
                    shares_outstanding=COALESCE(excluded.shares_outstanding, asset_profiles.shares_outstanding),
                    avg_volume=COALESCE(excluded.avg_volume, asset_profiles.avg_volume),
                    beta=COALESCE(excluded.beta, asset_profiles.beta),
                    dividend_yield=COALESCE(excluded.dividend_yield, asset_profiles.dividend_yield),
                    expense_ratio=COALESCE(excluded.expense_ratio, asset_profiles.expense_ratio),
                    fund_family=COALESCE(excluded.fund_family, asset_profiles.fund_family),
                    category_name=COALESCE(excluded.category_name, asset_profiles.category_name),
                    raw_profile_json=COALESCE(excluded.raw_profile_json, asset_profiles.raw_profile_json),
                    source=COALESCE(excluded.source, asset_profiles.source),
                    as_of_date=COALESCE(excluded.as_of_date, asset_profiles.as_of_date)
                """,
                profile_rows,
            )
            conn.executemany(
                """
                INSERT INTO asset_identifiers (
                    asset_id, provider, provider_symbol, exchange_symbol, conid, isin, cusip,
                    figi, composite_figi, shareclass_figi, composite_key, is_primary, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(asset_id, provider, provider_symbol) DO UPDATE SET
                    updated_at=CURRENT_TIMESTAMP,
                    exchange_symbol=COALESCE(excluded.exchange_symbol, asset_identifiers.exchange_symbol),
                    conid=COALESCE(excluded.conid, asset_identifiers.conid),
                    isin=COALESCE(excluded.isin, asset_identifiers.isin),
                    cusip=COALESCE(excluded.cusip, asset_identifiers.cusip),
                    figi=COALESCE(excluded.figi, asset_identifiers.figi),
                    composite_figi=COALESCE(excluded.composite_figi, asset_identifiers.composite_figi),
                    shareclass_figi=COALESCE(excluded.shareclass_figi, asset_identifiers.shareclass_figi),
                    composite_key=COALESCE(excluded.composite_key, asset_identifiers.composite_key),
                    is_primary=COALESCE(excluded.is_primary, asset_identifiers.is_primary)
                """,
                identifier_rows,
            )
            if raw_payload_rows:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO provider_raw_payloads (
                        raw_payload_id, provider, dataset_code, asset_id, provider_symbol,
                        provider_record_key, payload_json, payload_hash, fetched_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    raw_payload_rows,
                )

            conn.commit()

            summary["asset_classes"][dataset_code] = {
                "rows": int(len(frame)),
                "assets_written": int(len(asset_rows)),
            }
            summary["total_rows"] = int(summary["total_rows"]) + int(len(frame))
            summary["total_assets"] = int(summary["total_assets"]) + int(len(asset_rows))

    if progress_callback is not None:
        progress_callback(
            {
                "provider": FINANCEDATABASE_PROVIDER,
                "phase": "complete",
                "label": "FinanceDatabase import complete.",
                "percent": 100,
            }
        )
    return summary
