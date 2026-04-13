from __future__ import annotations

import contextlib
import importlib.util
import json
import sqlite3
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any, Callable, Sequence

import duckdb
import pandas as pd

from .catalog import ResultCatalog


SIMFIN_PROVIDER = "simfin"
DEFEATBETA_PROVIDER = "defeatbeta"
SIMFIN_CACHE_DIR = Path("data") / "provider_cache" / "simfin"
DEFEATBETA_BASE_URL = "https://huggingface.co/datasets/defeatbeta/yahoo-finance-data/resolve/main/data"


@dataclass(frozen=True)
class ProviderDatasetSpec:
    code: str
    display_name: str
    dataset_group: str
    description: str
    table_name: str
    loader_name: str | None = None
    variant: str | None = None
    supports_history: bool = True


SIMFIN_DATASET_SPECS: tuple[ProviderDatasetSpec, ...] = (
    ProviderDatasetSpec(
        "companies",
        "Companies",
        "reference",
        "SimFin company reference records.",
        "simfin_company_snapshots",
        loader_name="load_companies",
        supports_history=False,
    ),
    ProviderDatasetSpec(
        "shareprices_latest",
        "Share Prices (Latest)",
        "market_data",
        "Latest SimFin share-price records.",
        "simfin_shareprice_snapshots",
        loader_name="load_shareprices",
        variant="latest",
    ),
    ProviderDatasetSpec(
        "income_annual",
        "Income Statement (Annual)",
        "financial_statements",
        "Annual SimFin income statements.",
        "simfin_income_statements",
        loader_name="load_income",
        variant="annual",
    ),
    ProviderDatasetSpec(
        "income_quarterly",
        "Income Statement (Quarterly)",
        "financial_statements",
        "Quarterly SimFin income statements.",
        "simfin_income_statements",
        loader_name="load_income",
        variant="quarterly",
    ),
    ProviderDatasetSpec(
        "income_ttm",
        "Income Statement (TTM)",
        "financial_statements",
        "TTM SimFin income statements.",
        "simfin_income_statements",
        loader_name="load_income",
        variant="ttm",
    ),
    ProviderDatasetSpec(
        "balance_annual",
        "Balance Sheet (Annual)",
        "financial_statements",
        "Annual SimFin balance sheets.",
        "simfin_balance_sheets",
        loader_name="load_balance",
        variant="annual",
    ),
    ProviderDatasetSpec(
        "balance_quarterly",
        "Balance Sheet (Quarterly)",
        "financial_statements",
        "Quarterly SimFin balance sheets.",
        "simfin_balance_sheets",
        loader_name="load_balance",
        variant="quarterly",
    ),
    ProviderDatasetSpec(
        "balance_ttm",
        "Balance Sheet (TTM)",
        "financial_statements",
        "TTM SimFin balance sheets.",
        "simfin_balance_sheets",
        loader_name="load_balance",
        variant="ttm",
    ),
    ProviderDatasetSpec(
        "cashflow_annual",
        "Cash Flow (Annual)",
        "financial_statements",
        "Annual SimFin cash-flow statements.",
        "simfin_cash_flow_statements",
        loader_name="load_cashflow",
        variant="annual",
    ),
    ProviderDatasetSpec(
        "cashflow_quarterly",
        "Cash Flow (Quarterly)",
        "financial_statements",
        "Quarterly SimFin cash-flow statements.",
        "simfin_cash_flow_statements",
        loader_name="load_cashflow",
        variant="quarterly",
    ),
    ProviderDatasetSpec(
        "cashflow_ttm",
        "Cash Flow (TTM)",
        "financial_statements",
        "TTM SimFin cash-flow statements.",
        "simfin_cash_flow_statements",
        loader_name="load_cashflow",
        variant="ttm",
    ),
    ProviderDatasetSpec(
        "derived_annual",
        "Derived Metrics (Annual)",
        "metrics",
        "Annual SimFin derived metrics and ratios.",
        "simfin_metric_snapshots",
        loader_name="load_derived",
        variant="annual",
    ),
    ProviderDatasetSpec(
        "derived_quarterly",
        "Derived Metrics (Quarterly)",
        "metrics",
        "Quarterly SimFin derived metrics and ratios.",
        "simfin_metric_snapshots",
        loader_name="load_derived",
        variant="quarterly",
    ),
    ProviderDatasetSpec(
        "derived_ttm",
        "Derived Metrics (TTM)",
        "metrics",
        "TTM SimFin derived metrics and ratios.",
        "simfin_metric_snapshots",
        loader_name="load_derived",
        variant="ttm",
    ),
)


DEFEATBETA_DATASET_SPECS: tuple[ProviderDatasetSpec, ...] = (
    ProviderDatasetSpec(
        "stock_profile",
        "Company Profile",
        "reference",
        "Defeatbeta company profile records.",
        "defeatbeta_company_profiles",
        supports_history=False,
    ),
    ProviderDatasetSpec(
        "stock_officers",
        "Officers",
        "reference",
        "Defeatbeta officer roster records.",
        "defeatbeta_officers",
        supports_history=False,
    ),
    ProviderDatasetSpec(
        "stock_earning_calendar",
        "Earnings Calendar",
        "events",
        "Defeatbeta earnings calendar records.",
        "defeatbeta_earnings_calendars",
    ),
    ProviderDatasetSpec(
        "stock_statement",
        "Financial Statements",
        "financial_statements",
        "Defeatbeta financial statement records.",
        "defeatbeta_financial_statements",
    ),
    ProviderDatasetSpec(
        "stock_dividend_events",
        "Dividend Events",
        "events",
        "Defeatbeta dividend event records.",
        "defeatbeta_dividend_events",
    ),
    ProviderDatasetSpec(
        "stock_split_events",
        "Split Events",
        "events",
        "Defeatbeta stock split records.",
        "defeatbeta_split_events",
    ),
    ProviderDatasetSpec(
        "stock_news",
        "News",
        "content",
        "Defeatbeta news articles related to the asset.",
        "defeatbeta_news_articles",
    ),
    ProviderDatasetSpec(
        "stock_earning_call_transcripts",
        "Earnings Call Transcripts",
        "content",
        "Defeatbeta earnings call transcripts.",
        "defeatbeta_transcripts",
    ),
    ProviderDatasetSpec(
        "stock_revenue_breakdown",
        "Revenue Breakdown",
        "financial_statements",
        "Defeatbeta revenue breakdown records.",
        "defeatbeta_revenue_breakdowns",
    ),
    ProviderDatasetSpec(
        "stock_shares_outstanding",
        "Shares Outstanding",
        "capital_structure",
        "Defeatbeta share-count records.",
        "defeatbeta_share_counts",
    ),
    ProviderDatasetSpec(
        "stock_sec_filing",
        "SEC Filings",
        "filings",
        "Defeatbeta SEC filing records.",
        "defeatbeta_sec_filings",
    ),
)


def simfin_available() -> bool:
    return importlib.util.find_spec("simfin") is not None


def simfin_install_hint() -> str:
    return "Install SimFin with `pip install simfin`."


def defeatbeta_available() -> bool:
    return importlib.util.find_spec("duckdb") is not None


def defeatbeta_install_hint() -> str:
    return "Install DuckDB with `pip install duckdb` to query the defeatbeta parquet endpoints."


def _missing(value: object) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except Exception:
        return False
    if isinstance(missing, bool):
        return missing
    if hasattr(missing, "all"):
        try:
            return bool(missing.all())
        except Exception:
            return False
    if isinstance(missing, (list, tuple)):
        return all(bool(item) for item in missing)
    return False


def _clean_text(value: object) -> str | None:
    if _missing(value):
        return None
    text = str(value).strip()
    return text or None


def _clean_int(value: object) -> int | None:
    if _missing(value):
        return None
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return None


def _clean_float(value: object) -> float | None:
    if _missing(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _json_ready(value: object) -> object:
    if _missing(value):
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return None if pd.isna(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        converted = value.tolist()
        if converted is not value:
            return _json_ready(converted)
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return str(value)


def _json_dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True)


def _raise_if_stop_requested(stop_requested: Callable[[], bool] | None) -> None:
    if stop_requested is not None and stop_requested():
        raise InterruptedError("Provider sync cancelled.")


def _normalize_symbols(symbols: Sequence[str] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for symbol in list(symbols or []):
        cleaned = ResultCatalog._normalize_asset_symbol(symbol)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            normalized.append(cleaned)
    if not normalized:
        raise ValueError("At least one symbol is required.")
    return normalized


def _row_value(row: dict[str, object], *names: str) -> object | None:
    lowered = {str(key).strip().lower(): value for key, value in row.items()}
    for name in names:
        key = str(name).strip().lower()
        if key in lowered and not _missing(lowered[key]):
            return lowered[key]
    return None


def _frame_to_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    working = frame.copy()
    if working.empty:
        return []
    if isinstance(working.index, pd.MultiIndex) or any(name is not None for name in list(working.index.names or [])):
        working = working.reset_index()
    elif working.index.name is not None:
        working = working.reset_index()
    columns = [str(column).strip() or f"column_{idx}" for idx, column in enumerate(working.columns)]
    working.columns = columns
    return [
        {str(key): _json_ready(value) for key, value in row.items()}
        for row in working.to_dict(orient="records")
    ]


def _stable_key(provider: str, dataset_code: str, symbol: str, *parts: object) -> str:
    clean_parts = [
        str(provider).strip().lower(),
        str(dataset_code).strip().lower(),
        ResultCatalog._normalize_asset_symbol(symbol),
    ]
    clean_parts.extend(str(part).strip() for part in parts if str(part or "").strip())
    digest = sha1("|".join(clean_parts).encode("utf-8")).hexdigest()
    return digest[:32]


def _sql_text(value: str) -> str:
    return value.replace("'", "''")


def _upsert_dataset_catalog(conn: sqlite3.Connection, provider: str, spec: ProviderDatasetSpec) -> None:
    conn.execute(
        """
        INSERT INTO provider_dataset_catalog (
            provider, dataset_code, created_at, updated_at, dataset_group,
            display_name, description, is_enabled, is_structured,
            supports_history, supports_incremental_refresh, supports_point_in_time,
            last_documentation_reviewed_at
        )
        VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?, ?, 1, 1, ?, 1, 1, CURRENT_TIMESTAMP)
        ON CONFLICT(provider, dataset_code) DO UPDATE SET
            updated_at=CURRENT_TIMESTAMP,
            dataset_group=excluded.dataset_group,
            display_name=excluded.display_name,
            description=excluded.description,
            is_enabled=excluded.is_enabled,
            is_structured=excluded.is_structured,
            supports_history=excluded.supports_history,
            supports_incremental_refresh=excluded.supports_incremental_refresh,
            supports_point_in_time=excluded.supports_point_in_time,
            last_documentation_reviewed_at=CURRENT_TIMESTAMP
        """,
        (
            provider,
            spec.code,
            spec.dataset_group,
            spec.display_name,
            spec.description,
            1 if spec.supports_history else 0,
        ),
    )


def _upsert_field_inventory(
    conn: sqlite3.Connection,
    provider: str,
    dataset_code: str,
    frame: pd.DataFrame,
) -> int:
    if frame.empty:
        return 0
    columns = [str(column).strip() for column in frame.columns if str(column).strip()]
    existing = {
        str(row[0] or "")
        for row in conn.execute(
            """
            SELECT field_path
            FROM provider_field_inventory
            WHERE provider=? AND dataset_code=?
            """,
            (provider, dataset_code),
        ).fetchall()
    }
    dtype_map = {str(column): str(dtype) for column, dtype in frame.dtypes.items()}
    new_field_count = 0
    for column in columns:
        if column not in existing:
            new_field_count += 1
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
                provider,
                dataset_code,
                column,
                column,
                dtype_map.get(column, "object"),
                "needs_review",
                "Observed during provider-native asset sync.",
            ),
        )
    return new_field_count


def _store_raw_payload(
    conn: sqlite3.Connection,
    *,
    provider: str,
    dataset_code: str,
    asset_id: str,
    provider_symbol: str,
    provider_record_key: str,
    payload: dict[str, object],
    as_of_date: str | None = None,
    fiscal_year: int | None = None,
    fiscal_period: str | None = None,
    fetched_at: str | None = None,
) -> str:
    payload_json = _json_dumps(payload)
    payload_hash = sha1(payload_json.encode("utf-8")).hexdigest()
    raw_payload_id = f"raw_{_stable_key(provider, dataset_code, provider_symbol, provider_record_key, payload_hash)}"
    conn.execute(
        """
        INSERT OR REPLACE INTO provider_raw_payloads (
            raw_payload_id, created_at, provider, dataset_code, asset_id, provider_symbol,
            provider_record_key, as_of_date, fiscal_year, fiscal_period, payload_json,
            payload_hash, fetched_at
        )
        VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            raw_payload_id,
            provider,
            dataset_code,
            asset_id,
            provider_symbol,
            provider_record_key,
            as_of_date,
            fiscal_year,
            fiscal_period,
            payload_json,
            payload_hash,
            fetched_at,
        ),
    )
    return raw_payload_id


def _upsert_asset_master_row(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    display_symbol: str | None = None,
    name: str | None = None,
    asset_class: str | None = None,
    security_type: str | None = None,
    exchange: str | None = None,
    country: str | None = None,
    currency: str | None = None,
) -> str:
    normalized_symbol = ResultCatalog._normalize_asset_symbol(symbol)
    asset_id = ResultCatalog._asset_id_from_symbol(normalized_symbol)
    conn.execute(
        """
        INSERT INTO asset_master (
            asset_id, symbol, display_symbol, name, asset_class, security_type,
            exchange, country, currency, first_seen_at, last_seen_at, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ON CONFLICT(asset_id) DO UPDATE SET
            symbol=excluded.symbol,
            display_symbol=COALESCE(NULLIF(excluded.display_symbol, ''), asset_master.display_symbol),
            name=COALESCE(NULLIF(excluded.name, ''), asset_master.name),
            asset_class=COALESCE(NULLIF(excluded.asset_class, ''), asset_master.asset_class),
            security_type=COALESCE(NULLIF(excluded.security_type, ''), asset_master.security_type),
            exchange=COALESCE(NULLIF(excluded.exchange, ''), asset_master.exchange),
            country=COALESCE(NULLIF(excluded.country, ''), asset_master.country),
            currency=COALESCE(NULLIF(excluded.currency, ''), asset_master.currency),
            last_seen_at=CURRENT_TIMESTAMP,
            updated_at=CURRENT_TIMESTAMP
        """,
        (
            asset_id,
            normalized_symbol,
            str(display_symbol or "").strip() or normalized_symbol,
            str(name or "").strip() or normalized_symbol,
            str(asset_class or "").strip() or None,
            str(security_type or "").strip() or None,
            str(exchange or "").strip() or None,
            str(country or "").strip() or None,
            str(currency or "").strip() or None,
        ),
    )
    return asset_id


def _upsert_asset_identifier(
    conn: sqlite3.Connection,
    *,
    asset_id: str,
    provider: str,
    provider_symbol: str,
    exchange_symbol: str | None = None,
    isin: str | None = None,
    cusip: str | None = None,
    figi: str | None = None,
    composite_key: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO asset_identifiers (
            asset_id, provider, created_at, updated_at, provider_symbol, exchange_symbol,
            conid, isin, cusip, figi, composite_figi, shareclass_figi, composite_key, is_primary
        )
        VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?, NULL, ?, ?, ?, NULL, NULL, ?, 1)
        ON CONFLICT(asset_id, provider, provider_symbol) DO UPDATE SET
            updated_at=CURRENT_TIMESTAMP,
            exchange_symbol=COALESCE(excluded.exchange_symbol, asset_identifiers.exchange_symbol),
            isin=COALESCE(excluded.isin, asset_identifiers.isin),
            cusip=COALESCE(excluded.cusip, asset_identifiers.cusip),
            figi=COALESCE(excluded.figi, asset_identifiers.figi),
            composite_key=COALESCE(excluded.composite_key, asset_identifiers.composite_key),
            is_primary=COALESCE(excluded.is_primary, asset_identifiers.is_primary)
        """,
        (
            asset_id,
            provider,
            provider_symbol,
            exchange_symbol,
            isin,
            cusip,
            figi,
            composite_key,
        ),
    )


def _upsert_asset_classification(
    conn: sqlite3.Connection,
    *,
    asset_id: str,
    sector: str | None = None,
    industry: str | None = None,
    country: str | None = None,
    exchange: str | None = None,
    tags: dict[str, object] | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO asset_classifications (
            asset_id, updated_at, sector, industry, country, exchange, tags_json
        )
        VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?)
        ON CONFLICT(asset_id) DO UPDATE SET
            updated_at=CURRENT_TIMESTAMP,
            sector=COALESCE(excluded.sector, asset_classifications.sector),
            industry=COALESCE(excluded.industry, asset_classifications.industry),
            country=COALESCE(excluded.country, asset_classifications.country),
            exchange=COALESCE(excluded.exchange, asset_classifications.exchange),
            tags_json=COALESCE(excluded.tags_json, asset_classifications.tags_json)
        """,
        (
            asset_id,
            sector,
            industry,
            country,
            exchange,
            _json_dumps(tags or {}) if tags else None,
        ),
    )


def _upsert_asset_profile(
    conn: sqlite3.Connection,
    *,
    asset_id: str,
    description: str | None = None,
    website: str | None = None,
    employee_count: int | None = None,
    ipo_date: str | None = None,
    market_cap: float | None = None,
    shares_outstanding: float | None = None,
    beta: float | None = None,
    raw_profile_json: str | None = None,
    source: str | None = None,
    as_of_date: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO asset_profiles (
            asset_id, updated_at, description, website, employee_count, ipo_date,
            market_cap, shares_outstanding, beta, raw_profile_json, source, as_of_date
        )
        VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(asset_id) DO UPDATE SET
            updated_at=CURRENT_TIMESTAMP,
            description=COALESCE(excluded.description, asset_profiles.description),
            website=COALESCE(excluded.website, asset_profiles.website),
            employee_count=COALESCE(excluded.employee_count, asset_profiles.employee_count),
            ipo_date=COALESCE(excluded.ipo_date, asset_profiles.ipo_date),
            market_cap=COALESCE(excluded.market_cap, asset_profiles.market_cap),
            shares_outstanding=COALESCE(excluded.shares_outstanding, asset_profiles.shares_outstanding),
            beta=COALESCE(excluded.beta, asset_profiles.beta),
            raw_profile_json=COALESCE(excluded.raw_profile_json, asset_profiles.raw_profile_json),
            source=COALESCE(excluded.source, asset_profiles.source),
            as_of_date=COALESCE(excluded.as_of_date, asset_profiles.as_of_date)
        """,
        (
            asset_id,
            description,
            website,
            employee_count,
            ipo_date,
            market_cap,
            shares_outstanding,
            beta,
            raw_profile_json,
            source,
            as_of_date,
        ),
    )


def _simfin_industry_map(sf_module, *, refresh_days: int) -> dict[int, dict[str, object]]:
    try:
        frame = sf_module.load_industries(refresh_days=refresh_days)
    except Exception:
        return {}
    records = _frame_to_records(frame)
    mapping: dict[int, dict[str, object]] = {}
    for row in records:
        industry_id = _clean_int(_row_value(row, "IndustryId", "Industry ID", "industry_id"))
        if industry_id is None:
            continue
        mapping[industry_id] = row
    return mapping


@contextlib.contextmanager
def _simfin_read_csv_compat() -> Any:
    """Temporarily tolerate SimFin calling pandas with removed CSV kwargs."""

    original_read_csv = pd.read_csv
    pandas_readers = None
    original_readers_read_csv = None
    try:
        from pandas.io.parsers import readers as pandas_readers  # type: ignore

        original_readers_read_csv = getattr(pandas_readers, "read_csv", None)
    except Exception:
        pandas_readers = None

    def compat_read_csv(*args, **kwargs):
        try:
            return original_read_csv(*args, **kwargs)
        except TypeError as exc:
            message = str(exc or "")
            fallback_kwargs = dict(kwargs)
            removed = False
            for key in ("date_parser", "infer_datetime_format"):
                if key in fallback_kwargs and key in message and "unexpected keyword argument" in message:
                    fallback_kwargs.pop(key, None)
                    removed = True
            if not removed:
                raise
            return original_read_csv(*args, **fallback_kwargs)

    pd.read_csv = compat_read_csv
    if pandas_readers is not None and original_readers_read_csv is not None:
        pandas_readers.read_csv = compat_read_csv
    try:
        yield
    finally:
        pd.read_csv = original_read_csv
        if pandas_readers is not None and original_readers_read_csv is not None:
            pandas_readers.read_csv = original_readers_read_csv


def _simfin_insert_record(
    conn: sqlite3.Connection,
    *,
    table_name: str,
    record_key: str,
    asset_id: str,
    symbol: str,
    market: str,
    variant: str | None,
    row: dict[str, object],
    raw_payload_id: str | None,
) -> None:
    report_date = _clean_text(_row_value(row, "Report Date", "report_date"))
    publish_date = _clean_text(_row_value(row, "Publish Date", "publish_date"))
    restated_date = _clean_text(_row_value(row, "Restated Date", "restated_date"))
    fiscal_year = _clean_int(_row_value(row, "Fiscal Year", "FiscalYear", "fiscal_year"))
    fiscal_period = _clean_text(_row_value(row, "Fiscal Period", "FiscalPeriod", "fiscal_period"))
    payload_json = _json_dumps(row)
    if table_name == "simfin_company_snapshots":
        conn.execute(
            """
            INSERT INTO simfin_company_snapshots (
                record_key, created_at, updated_at, asset_id, simfin_ticker, simfin_id,
                market, company_name, industry_id, raw_payload_id, payload_json, as_of_date
            )
            VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(record_key) DO UPDATE SET
                updated_at=CURRENT_TIMESTAMP,
                simfin_id=excluded.simfin_id,
                market=excluded.market,
                company_name=excluded.company_name,
                industry_id=excluded.industry_id,
                raw_payload_id=excluded.raw_payload_id,
                payload_json=excluded.payload_json,
                as_of_date=excluded.as_of_date
            """,
            (
                record_key,
                asset_id,
                symbol,
                _clean_int(_row_value(row, "SimFinId", "SimFin ID", "simfinid")),
                market,
                _clean_text(_row_value(row, "Company Name", "Name", "company_name", "name")),
                _clean_int(_row_value(row, "IndustryId", "Industry ID", "industry_id")),
                raw_payload_id,
                payload_json,
                _clean_text(_row_value(row, "Date", "AsOfDate", "as_of_date")),
            ),
        )
        return
    if table_name == "simfin_shareprice_snapshots":
        conn.execute(
            """
            INSERT INTO simfin_shareprice_snapshots (
                record_key, created_at, updated_at, asset_id, simfin_ticker, market,
                variant, price_date, raw_payload_id, payload_json
            )
            VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(record_key) DO UPDATE SET
                updated_at=CURRENT_TIMESTAMP,
                market=excluded.market,
                variant=excluded.variant,
                price_date=excluded.price_date,
                raw_payload_id=excluded.raw_payload_id,
                payload_json=excluded.payload_json
            """,
            (
                record_key,
                asset_id,
                symbol,
                market,
                variant,
                _clean_text(_row_value(row, "Date", "date")),
                raw_payload_id,
                payload_json,
            ),
        )
        return
    conn.execute(
        f"""
        INSERT INTO {table_name} (
            record_key, created_at, updated_at, asset_id, simfin_ticker, market, variant,
            report_date, publish_date, restated_date, fiscal_year, fiscal_period,
            raw_payload_id, payload_json
        )
        VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(record_key) DO UPDATE SET
            updated_at=CURRENT_TIMESTAMP,
            market=excluded.market,
            variant=excluded.variant,
            report_date=excluded.report_date,
            publish_date=excluded.publish_date,
            restated_date=excluded.restated_date,
            fiscal_year=excluded.fiscal_year,
            fiscal_period=excluded.fiscal_period,
            raw_payload_id=excluded.raw_payload_id,
            payload_json=excluded.payload_json
        """,
        (
            record_key,
            asset_id,
            symbol,
            market,
            variant,
            report_date,
            publish_date,
            restated_date,
            fiscal_year,
            fiscal_period,
            raw_payload_id,
            payload_json,
        ),
    )


def sync_simfin_assets(
    *,
    catalog_path: str | Path,
    symbols: Sequence[str],
    market: str = "us",
    refresh_days: int = 30,
    store_raw_payloads: bool = True,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> dict[str, object]:
    if not simfin_available():
        raise RuntimeError(f"Missing Python dependency `simfin`. {simfin_install_hint()}")

    import simfin as sf  # type: ignore[import-not-found]

    catalog = ResultCatalog(catalog_path)
    normalized_symbols = _normalize_symbols(symbols)
    credential = catalog.load_provider_credential(SIMFIN_PROVIDER)
    api_key = str(credential.get("api_key") or "").strip() or "free"
    SIMFIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    sf.set_api_key(api_key)
    sf.set_data_dir(str(SIMFIN_CACHE_DIR))

    provider_sync_run_id = catalog.start_provider_sync_run(SIMFIN_PROVIDER)
    summary: dict[str, object] = {
        "provider": SIMFIN_PROVIDER,
        "symbols": normalized_symbols,
        "dataset_counts": {},
        "record_count": 0,
        "asset_count": 0,
        "new_field_count": 0,
    }
    touched_assets: set[str] = set()

    try:
        with _simfin_read_csv_compat():
            _raise_if_stop_requested(stop_requested)
            industry_map = _simfin_industry_map(sf, refresh_days=refresh_days)
            _raise_if_stop_requested(stop_requested)
            with catalog.connect() as conn:
                total_specs = max(1, len(SIMFIN_DATASET_SPECS))
                for spec_index, spec in enumerate(SIMFIN_DATASET_SPECS, start=1):
                    _raise_if_stop_requested(stop_requested)
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "provider": SIMFIN_PROVIDER,
                                "phase": "load_dataset",
                                "label": f"Loading SimFin {spec.display_name}…",
                                "percent": int(((spec_index - 1) / total_specs) * 100),
                            }
                        )
                    _upsert_dataset_catalog(conn, SIMFIN_PROVIDER, spec)
                    loader = getattr(sf, str(spec.loader_name or "").strip())
                    frame = loader(market=market, variant=spec.variant, refresh_days=refresh_days)
                    records = [
                        row
                        for row in _frame_to_records(frame)
                        if ResultCatalog._normalize_asset_symbol(_row_value(row, "Ticker", "ticker"))
                        in normalized_symbols
                    ]
                    summary["dataset_counts"][spec.code] = len(records)
                    if not records:
                        if progress_callback is not None:
                            progress_callback(
                                {
                                    "provider": SIMFIN_PROVIDER,
                                    "phase": "skip_dataset",
                                    "label": f"SimFin {spec.display_name}: no matching records.",
                                    "percent": int((spec_index / total_specs) * 100),
                                }
                            )
                        continue
                    dataset_frame = pd.DataFrame(records)
                    summary["new_field_count"] = int(summary["new_field_count"]) + _upsert_field_inventory(
                        conn,
                        SIMFIN_PROVIDER,
                        spec.code,
                        dataset_frame,
                    )
                    total_records = max(1, len(records))
                    for record_index, row in enumerate(records, start=1):
                        _raise_if_stop_requested(stop_requested)
                        symbol = ResultCatalog._normalize_asset_symbol(_row_value(row, "Ticker", "ticker"))
                        company_name = _clean_text(_row_value(row, "Company Name", "Name", "company_name", "name"))
                        currency = _clean_text(_row_value(row, "Currency", "currency"))
                        asset_id = _upsert_asset_master_row(
                            conn,
                            symbol=symbol,
                            display_symbol=symbol,
                            name=company_name,
                            asset_class="Equities",
                            security_type="Equity",
                            currency=currency,
                        )
                        touched_assets.add(asset_id)
                        industry_id = _clean_int(_row_value(row, "IndustryId", "Industry ID", "industry_id"))
                        industry_row = industry_map.get(int(industry_id)) if industry_id is not None else None
                        if spec.code == "companies":
                            _upsert_asset_identifier(
                                conn,
                                asset_id=asset_id,
                                provider=SIMFIN_PROVIDER,
                                provider_symbol=symbol,
                                isin=_clean_text(_row_value(row, "ISIN", "isin")),
                                composite_key=_clean_text(_row_value(row, "SimFinId", "SimFin ID", "simfinid")),
                            )
                            _upsert_asset_classification(
                                conn,
                                asset_id=asset_id,
                                sector=_clean_text(_row_value(industry_row or {}, "Sector", "sector")),
                                industry=_clean_text(
                                    _row_value(
                                        industry_row or {},
                                        "Industry",
                                        "industry",
                                        "Industry Name",
                                        "industry_name",
                                    )
                                ),
                                country=_clean_text(_row_value(row, "Country", "country")),
                                exchange=_clean_text(_row_value(row, "Exchange", "exchange", "Market Name")),
                                tags={
                                    "provider": SIMFIN_PROVIDER,
                                    "market": market,
                                    "industry_id": industry_id,
                                },
                            )
                            _upsert_asset_profile(
                                conn,
                                asset_id=asset_id,
                                website=_clean_text(_row_value(row, "Website", "website")),
                                market_cap=_clean_float(_row_value(row, "Market Cap", "market_cap")),
                                raw_profile_json=_json_dumps(row),
                                source=SIMFIN_PROVIDER,
                                as_of_date=_clean_text(_row_value(row, "Date", "AsOfDate", "as_of_date")),
                            )
                        record_key = _stable_key(
                            SIMFIN_PROVIDER,
                            spec.code,
                            symbol,
                            spec.variant or "",
                            _clean_text(_row_value(row, "Date", "Report Date", "date", "report_date")),
                            _clean_text(_row_value(row, "Fiscal Period", "fiscal_period")),
                        )
                        raw_payload_id = None
                        if store_raw_payloads:
                            raw_payload_id = _store_raw_payload(
                                conn,
                                provider=SIMFIN_PROVIDER,
                                dataset_code=spec.code,
                                asset_id=asset_id,
                                provider_symbol=symbol,
                                provider_record_key=record_key,
                                payload=row,
                                as_of_date=_clean_text(_row_value(row, "Date", "Report Date", "date", "report_date")),
                                fiscal_year=_clean_int(_row_value(row, "Fiscal Year", "fiscal_year")),
                                fiscal_period=_clean_text(_row_value(row, "Fiscal Period", "fiscal_period")),
                            )
                        _simfin_insert_record(
                            conn,
                            table_name=spec.table_name,
                            record_key=record_key,
                            asset_id=asset_id,
                            symbol=symbol,
                            market=market,
                            variant=spec.variant,
                            row=row,
                            raw_payload_id=raw_payload_id,
                        )
                        summary["record_count"] = int(summary["record_count"]) + 1
                        if progress_callback is not None and (record_index == total_records or record_index % 100 == 0):
                            percent = int((((spec_index - 1) + (record_index / total_records)) / total_specs) * 100)
                            progress_callback(
                                {
                                    "provider": SIMFIN_PROVIDER,
                                    "phase": "write_rows",
                                    "label": f"SimFin {spec.display_name}: {record_index:,} / {total_records:,}",
                                    "percent": min(99, percent),
                                }
                            )
                    conn.commit()
        summary["asset_count"] = len(touched_assets)
        if progress_callback is not None:
            progress_callback(
                {
                    "provider": SIMFIN_PROVIDER,
                    "phase": "complete",
                    "label": "SimFin sync complete.",
                    "percent": 100,
                }
            )
        catalog.finish_provider_sync_run(
            provider_sync_run_id,
            status="completed",
            asset_count=int(summary["asset_count"]),
            record_count=int(summary["record_count"]),
            new_field_count=int(summary["new_field_count"]),
        )
        return summary
    except InterruptedError as exc:
        catalog.finish_provider_sync_run(
            provider_sync_run_id,
            status="cancelled",
            asset_count=len(touched_assets),
            record_count=int(summary.get("record_count") or 0),
            new_field_count=int(summary.get("new_field_count") or 0),
            error_summary=str(exc),
        )
        raise
    except Exception as exc:
        catalog.finish_provider_sync_run(
            provider_sync_run_id,
            status="failed",
            asset_count=len(touched_assets),
            record_count=int(summary.get("record_count") or 0),
            new_field_count=int(summary.get("new_field_count") or 0),
            error_summary=str(exc),
        )
        raise


def _connect_defeatbeta_duckdb() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    try:
        conn.execute("INSTALL httpfs")
    except Exception:
        pass
    try:
        conn.execute("LOAD httpfs")
    except Exception:
        pass
    return conn


def _defeatbeta_url(dataset_code: str) -> str:
    return f"{DEFEATBETA_BASE_URL}/{dataset_code}.parquet"


def _defeatbeta_query_frame(
    conn: duckdb.DuckDBPyConnection,
    *,
    dataset_code: str,
    symbols: Sequence[str],
) -> pd.DataFrame:
    url = _defeatbeta_url(dataset_code)
    probe = conn.sql(f"SELECT * FROM '{url}' LIMIT 0").df()
    columns = {str(column).strip().lower(): str(column).strip() for column in probe.columns}
    symbol_column = columns.get("symbol") or columns.get("related_symbols")
    if not symbol_column:
        return pd.DataFrame()
    in_clause = ", ".join(f"'{_sql_text(symbol)}'" for symbol in list(symbols or []))
    query = (
        f"SELECT * FROM '{url}' "
        f"WHERE UPPER({symbol_column}) IN ({in_clause})"
    )
    return conn.sql(query).df()


def _defeatbeta_insert_record(
    conn: sqlite3.Connection,
    *,
    table_name: str,
    record_key: str,
    asset_id: str,
    symbol: str,
    row: dict[str, object],
    raw_payload_id: str | None,
) -> None:
    report_date = _clean_text(_row_value(row, "report_date", "Report Date", "date", "Date"))
    payload_json = _json_dumps(row)
    if table_name == "defeatbeta_company_profiles":
        conn.execute(
            """
            INSERT INTO defeatbeta_company_profiles (
                record_key, created_at, updated_at, asset_id, provider_symbol, report_date, raw_payload_id, payload_json
            )
            VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?)
            ON CONFLICT(record_key) DO UPDATE SET
                updated_at=CURRENT_TIMESTAMP,
                report_date=excluded.report_date,
                raw_payload_id=excluded.raw_payload_id,
                payload_json=excluded.payload_json
            """,
            (record_key, asset_id, symbol, report_date, raw_payload_id, payload_json),
        )
        return
    if table_name == "defeatbeta_officers":
        conn.execute(
            """
            INSERT INTO defeatbeta_officers (
                record_key, created_at, updated_at, asset_id, provider_symbol, report_date,
                officer_name, officer_title, raw_payload_id, payload_json
            )
            VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(record_key) DO UPDATE SET
                updated_at=CURRENT_TIMESTAMP,
                report_date=excluded.report_date,
                officer_name=excluded.officer_name,
                officer_title=excluded.officer_title,
                raw_payload_id=excluded.raw_payload_id,
                payload_json=excluded.payload_json
            """,
            (
                record_key,
                asset_id,
                symbol,
                report_date,
                _clean_text(_row_value(row, "name", "officer", "officer_name")),
                _clean_text(_row_value(row, "title", "officer_title")),
                raw_payload_id,
                payload_json,
            ),
        )
        return
    if table_name == "defeatbeta_earnings_calendars":
        conn.execute(
            """
            INSERT INTO defeatbeta_earnings_calendars (
                record_key, created_at, updated_at, asset_id, provider_symbol, report_date, raw_payload_id, payload_json
            )
            VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?)
            ON CONFLICT(record_key) DO UPDATE SET
                updated_at=CURRENT_TIMESTAMP,
                report_date=excluded.report_date,
                raw_payload_id=excluded.raw_payload_id,
                payload_json=excluded.payload_json
            """,
            (record_key, asset_id, symbol, report_date, raw_payload_id, payload_json),
        )
        return
    if table_name == "defeatbeta_price_history":
        conn.execute(
            """
            INSERT INTO defeatbeta_price_history (
                record_key, created_at, updated_at, asset_id, provider_symbol, price_date, raw_payload_id, payload_json
            )
            VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?)
            ON CONFLICT(record_key) DO UPDATE SET
                updated_at=CURRENT_TIMESTAMP,
                price_date=excluded.price_date,
                raw_payload_id=excluded.raw_payload_id,
                payload_json=excluded.payload_json
            """,
            (
                record_key,
                asset_id,
                symbol,
                _clean_text(_row_value(row, "report_date", "date", "Date")),
                raw_payload_id,
                payload_json,
            ),
        )
        return
    if table_name == "defeatbeta_financial_statements":
        conn.execute(
            """
            INSERT INTO defeatbeta_financial_statements (
                record_key, created_at, updated_at, asset_id, provider_symbol, finance_type,
                period_type, report_date, raw_payload_id, payload_json
            )
            VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(record_key) DO UPDATE SET
                updated_at=CURRENT_TIMESTAMP,
                finance_type=excluded.finance_type,
                period_type=excluded.period_type,
                report_date=excluded.report_date,
                raw_payload_id=excluded.raw_payload_id,
                payload_json=excluded.payload_json
            """,
            (
                record_key,
                asset_id,
                symbol,
                _clean_text(_row_value(row, "finance_type")),
                _clean_text(_row_value(row, "period_type")),
                report_date,
                raw_payload_id,
                payload_json,
            ),
        )
        return
    if table_name == "defeatbeta_dividend_events":
        table_sql = "defeatbeta_dividend_events"
    elif table_name == "defeatbeta_split_events":
        table_sql = "defeatbeta_split_events"
    elif table_name == "defeatbeta_share_counts":
        table_sql = "defeatbeta_share_counts"
    elif table_name == "defeatbeta_transcripts":
        table_sql = "defeatbeta_transcripts"
    else:
        table_sql = ""
    if table_sql:
        conn.execute(
            f"""
            INSERT INTO {table_sql} (
                record_key, created_at, updated_at, asset_id, provider_symbol, report_date, raw_payload_id, payload_json
            )
            VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?)
            ON CONFLICT(record_key) DO UPDATE SET
                updated_at=CURRENT_TIMESTAMP,
                report_date=excluded.report_date,
                raw_payload_id=excluded.raw_payload_id,
                payload_json=excluded.payload_json
            """,
            (record_key, asset_id, symbol, report_date, raw_payload_id, payload_json),
        )
        return
    if table_name == "defeatbeta_news_articles":
        conn.execute(
            """
            INSERT INTO defeatbeta_news_articles (
                record_key, created_at, updated_at, asset_id, provider_symbol, report_date,
                headline, raw_payload_id, payload_json
            )
            VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(record_key) DO UPDATE SET
                updated_at=CURRENT_TIMESTAMP,
                report_date=excluded.report_date,
                headline=excluded.headline,
                raw_payload_id=excluded.raw_payload_id,
                payload_json=excluded.payload_json
            """,
            (
                record_key,
                asset_id,
                symbol,
                report_date,
                _clean_text(_row_value(row, "title", "headline")),
                raw_payload_id,
                payload_json,
            ),
        )
        return
    if table_name == "defeatbeta_revenue_breakdowns":
        conn.execute(
            """
            INSERT INTO defeatbeta_revenue_breakdowns (
                record_key, created_at, updated_at, asset_id, provider_symbol, breakdown_type,
                report_date, raw_payload_id, payload_json
            )
            VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(record_key) DO UPDATE SET
                updated_at=CURRENT_TIMESTAMP,
                breakdown_type=excluded.breakdown_type,
                report_date=excluded.report_date,
                raw_payload_id=excluded.raw_payload_id,
                payload_json=excluded.payload_json
            """,
            (
                record_key,
                asset_id,
                symbol,
                _clean_text(_row_value(row, "breakdown_type")),
                report_date,
                raw_payload_id,
                payload_json,
            ),
        )
        return
    if table_name == "defeatbeta_sec_filings":
        conn.execute(
            """
            INSERT INTO defeatbeta_sec_filings (
                record_key, created_at, updated_at, asset_id, provider_symbol, report_date,
                filing_type, raw_payload_id, payload_json
            )
            VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(record_key) DO UPDATE SET
                updated_at=CURRENT_TIMESTAMP,
                report_date=excluded.report_date,
                filing_type=excluded.filing_type,
                raw_payload_id=excluded.raw_payload_id,
                payload_json=excluded.payload_json
            """,
            (
                record_key,
                asset_id,
                symbol,
                report_date,
                _clean_text(_row_value(row, "type", "filing_type", "form_type")),
                raw_payload_id,
                payload_json,
            ),
        )


def sync_defeatbeta_assets(
    *,
    catalog_path: str | Path,
    symbols: Sequence[str],
    store_raw_payloads: bool = True,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> dict[str, object]:
    if not defeatbeta_available():
        raise RuntimeError(defeatbeta_install_hint())

    catalog = ResultCatalog(catalog_path)
    normalized_symbols = _normalize_symbols(symbols)
    provider_sync_run_id = catalog.start_provider_sync_run(DEFEATBETA_PROVIDER)
    summary: dict[str, object] = {
        "provider": DEFEATBETA_PROVIDER,
        "symbols": normalized_symbols,
        "dataset_counts": {},
        "record_count": 0,
        "asset_count": 0,
        "new_field_count": 0,
    }
    touched_assets: set[str] = set()
    duck: duckdb.DuckDBPyConnection | None = None
    try:
        _raise_if_stop_requested(stop_requested)
        duck = _connect_defeatbeta_duckdb()
        with catalog.connect() as conn:
            total_specs = max(1, len(DEFEATBETA_DATASET_SPECS))
            for spec_index, spec in enumerate(DEFEATBETA_DATASET_SPECS, start=1):
                _raise_if_stop_requested(stop_requested)
                if progress_callback is not None:
                    progress_callback(
                        {
                            "provider": DEFEATBETA_PROVIDER,
                            "phase": "load_dataset",
                            "label": f"Loading defeatbeta {spec.display_name}…",
                            "percent": int(((spec_index - 1) / total_specs) * 100),
                        }
                    )
                _upsert_dataset_catalog(conn, DEFEATBETA_PROVIDER, spec)
                _raise_if_stop_requested(stop_requested)
                frame = _defeatbeta_query_frame(duck, dataset_code=spec.code, symbols=normalized_symbols)
                summary["dataset_counts"][spec.code] = int(len(frame))
                if frame.empty:
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "provider": DEFEATBETA_PROVIDER,
                                "phase": "skip_dataset",
                                "label": f"defeatbeta {spec.display_name}: no matching records.",
                                "percent": int((spec_index / total_specs) * 100),
                            }
                        )
                    continue
                records = _frame_to_records(frame)
                summary["new_field_count"] = int(summary["new_field_count"]) + _upsert_field_inventory(
                    conn,
                    DEFEATBETA_PROVIDER,
                    spec.code,
                    frame.reset_index(drop=True),
                )
                total_records = max(1, len(records))
                for record_index, row in enumerate(records, start=1):
                    _raise_if_stop_requested(stop_requested)
                    symbol = ResultCatalog._normalize_asset_symbol(
                        _row_value(row, "symbol", "related_symbols", "ticker")
                    )
                    if symbol not in normalized_symbols:
                        continue
                    profile_name = _clean_text(_row_value(row, "name", "long_name", "company_name", "longName"))
                    exchange = _clean_text(_row_value(row, "exchange", "full_exchange_name", "fullExchangeName"))
                    currency = _clean_text(_row_value(row, "financial_currency", "currency"))
                    asset_id = _upsert_asset_master_row(
                        conn,
                        symbol=symbol,
                        display_symbol=symbol,
                        name=profile_name,
                        asset_class="Equities",
                        security_type="Equity",
                        exchange=exchange,
                        currency=currency,
                    )
                    touched_assets.add(asset_id)
                    if spec.code == "stock_profile":
                        _upsert_asset_identifier(
                            conn,
                            asset_id=asset_id,
                            provider=DEFEATBETA_PROVIDER,
                            provider_symbol=symbol,
                            exchange_symbol=exchange,
                            isin=_clean_text(_row_value(row, "isin")),
                            cusip=_clean_text(_row_value(row, "cusip")),
                        )
                        _upsert_asset_classification(
                            conn,
                            asset_id=asset_id,
                            sector=_clean_text(_row_value(row, "sector")),
                            industry=_clean_text(_row_value(row, "industry")),
                            country=_clean_text(_row_value(row, "country")),
                            exchange=exchange,
                            tags={"provider": DEFEATBETA_PROVIDER},
                        )
                        _upsert_asset_profile(
                            conn,
                            asset_id=asset_id,
                            description=_clean_text(_row_value(row, "long_business_summary", "description")),
                            website=_clean_text(_row_value(row, "website")),
                            employee_count=_clean_int(_row_value(row, "full_time_employees", "employee_count")),
                            market_cap=_clean_float(_row_value(row, "market_cap", "marketCap")),
                            shares_outstanding=_clean_float(_row_value(row, "shares_outstanding", "sharesOutstanding")),
                            beta=_clean_float(_row_value(row, "beta")),
                            raw_profile_json=_json_dumps(row),
                            source=DEFEATBETA_PROVIDER,
                            as_of_date=_clean_text(_row_value(row, "report_date", "date")),
                        )
                    record_key = _stable_key(
                        DEFEATBETA_PROVIDER,
                        spec.code,
                        symbol,
                        _clean_text(_row_value(row, "report_date", "date")),
                        _clean_text(_row_value(row, "finance_type")),
                        _clean_text(_row_value(row, "period_type")),
                        _clean_text(_row_value(row, "breakdown_type")),
                    )
                    raw_payload_id = None
                    if store_raw_payloads:
                        raw_payload_id = _store_raw_payload(
                            conn,
                            provider=DEFEATBETA_PROVIDER,
                            dataset_code=spec.code,
                            asset_id=asset_id,
                            provider_symbol=symbol,
                            provider_record_key=record_key,
                            payload=row,
                            as_of_date=_clean_text(_row_value(row, "report_date", "date")),
                        )
                    _defeatbeta_insert_record(
                        conn,
                        table_name=spec.table_name,
                        record_key=record_key,
                        asset_id=asset_id,
                        symbol=symbol,
                        row=row,
                        raw_payload_id=raw_payload_id,
                    )
                    summary["record_count"] = int(summary["record_count"]) + 1
                    if progress_callback is not None and (record_index == total_records or record_index % 100 == 0):
                        percent = int((((spec_index - 1) + (record_index / total_records)) / total_specs) * 100)
                        progress_callback(
                            {
                                "provider": DEFEATBETA_PROVIDER,
                                "phase": "write_rows",
                                "label": f"defeatbeta {spec.display_name}: {record_index:,} / {total_records:,}",
                                "percent": min(99, percent),
                            }
                        )
                conn.commit()
        summary["asset_count"] = len(touched_assets)
        if progress_callback is not None:
            progress_callback(
                {
                    "provider": DEFEATBETA_PROVIDER,
                    "phase": "complete",
                    "label": "defeatbeta sync complete.",
                    "percent": 100,
                }
            )
        catalog.finish_provider_sync_run(
            provider_sync_run_id,
            status="completed",
            asset_count=int(summary["asset_count"]),
            record_count=int(summary["record_count"]),
            new_field_count=int(summary["new_field_count"]),
        )
        return summary
    except InterruptedError as exc:
        catalog.finish_provider_sync_run(
            provider_sync_run_id,
            status="cancelled",
            asset_count=len(touched_assets),
            record_count=int(summary.get("record_count") or 0),
            new_field_count=int(summary.get("new_field_count") or 0),
            error_summary=str(exc),
        )
        raise
    except Exception as exc:
        catalog.finish_provider_sync_run(
            provider_sync_run_id,
            status="failed",
            asset_count=len(touched_assets),
            record_count=int(summary.get("record_count") or 0),
            new_field_count=int(summary.get("new_field_count") or 0),
            error_summary=str(exc),
        )
        raise
    finally:
        if duck is not None:
            duck.close()
