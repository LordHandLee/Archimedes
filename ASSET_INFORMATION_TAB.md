# Asset Information Tab Design

## Purpose

This document outlines a practical design for an `Asset Information` tab that gives the platform a durable reference-data backbone.

The goal is to let us:

- browse assets by category such as equities, ETFs, crypto, forex, commodities, indices, and funds
- inspect core metadata for each asset in one place
- store normalized asset reference data in our database
- enrich that reference data from external sources when available
- use the resulting asset catalog as the foundation for a future stock and asset screener
- use screener output to build better universes for acquisition, backtesting, optimization, walk-forward, and portfolio work

This should be a reference-data and discovery layer first. It should not try to become a full research terminal in version 1.

## Why This Matters

Right now the platform is strong at:

- collecting market data
- organizing datasets
- running strategy studies
- building reusable universes

What is still missing is a strong asset master layer.

Without that layer, universe construction stays too manual because the user has to already know what symbols they want.

An `Asset Information` tab solves that by giving us:

- a clean catalog of what assets exist
- normalized metadata for filtering and grouping
- a path from asset discovery to universe construction
- a clean handoff into a future screener

## External Backbone

The proposed backbone uses three complementary sources:

### 1. FinanceDatabase

FinanceDatabase is a strong seed source for broad asset coverage and category metadata. Based on its public project documentation, it is designed around a large symbol database covering equities, ETFs, funds, indices, currencies, cryptocurrencies, and money markets.

Recommended role:

- seed the local asset master
- provide broad asset class and classification coverage
- provide exchange, country, sector, industry, and similar grouping fields where available
- give us a stable starting point for category-first browsing

### 2. defeatbeta-api

defeatbeta-api appears well suited as an enrichment layer for reference and fundamental-style metadata beyond the base symbol catalog.

Recommended role:

- enrich stored assets with profile-style metadata
- fill in issuer, summary, sector, industry, exchange, market-cap, and other descriptive fields when available
- supply price-adjacent and market/fundamental metrics such as TTM EPS, TTM P/E, market cap, P/S, P/B, PEG, ROE, ROIC, WACC, ROA, segment revenue, geography revenue, filings, transcripts, and related analytical data where available
- support later screener attributes and detail panes
- provide refreshable metadata snapshots rather than replacing the local asset master

### 3. SimFin

SimFin is a strong candidate for the deeper fundamentals layer. Based on SimFin's public product documentation, its API and bulk download products focus on verified quarterly and annual financial statements, calculated stock metrics, and long history windows.

Recommended role:

- supply statement-grade fundamentals from income statement, balance sheet, and cash flow data
- provide valuation, profitability, balance-sheet, and cash-flow metrics for screening
- provide longer historical depth for growth and trend calculations
- remain a separate provider store in the database so overlapping or conflicting fields are never silently merged with defeatbeta data

## Provider Separation

FinanceDatabase, defeatbeta-api, and SimFin should not be collapsed into one mixed storage table.

That is especially important for defeatbeta-api and SimFin because:

- they may overlap on fields like market cap, P/E, P/S, revenue, EBITDA, or free cash flow
- they may have different calculation methodologies
- they may update on different schedules
- they may cover different universes of assets
- one may have data where the other has gaps

The database design should preserve provider boundaries first.

The clean model is:

- canonical asset identity tables for shared asset lookup
- provider-specific profile, statement, and metrics tables
- optional derived or canonical screener views built on top of those tables later

That keeps provenance intact and prevents us from accidentally treating unlike values as interchangeable.

## Core Design Principle

The local database should be the system of record for normalized asset metadata.

External providers should be treated as inputs, not as the live UI model.

That gives us:

- fast UI queries
- deterministic filtering
- offline-friendly browsing of previously synced data
- controlled schema evolution
- source provenance and freshness tracking

For fundamentals and ratios, the system should also preserve point-in-time semantics.

That means:

- statement data should be stored with fiscal period context
- ratios should be stored with `as_of_date`
- price-dependent metrics should record the price date/reference used
- computed growth rates should record the source window or lookback

## Recommended Scope

Version 1 of the `Asset Information` tab should focus on:

- asset grouping by category
- searchable and filterable asset catalog
- asset detail panel
- provider/source provenance
- metadata freshness
- handoff into Universe Builder

Version 1 should not yet include:

- a full custom screener builder
- complex ranking models
- portfolio analytics
- live quote streaming
- advanced fundamentals comparison dashboards

Those can follow once the asset master is stable.

## Asset Categories

The UI should group assets at the top level by `asset_class`.

Recommended first-pass categories:

- Equities
- ETFs
- Funds
- Indices
- Forex
- Crypto
- Commodities
- Futures
- Options
- Fixed Income
- Money Markets
- Other

We should also support a secondary `security_type` so the model can distinguish cases such as:

- common stock
- ADR
- preferred
- warrant
- ETF
- mutual fund
- spot crypto
- forex pair
- commodity future
- index

This two-level model is more durable than trying to force everything into one category string.

## Recommended Database Model

The simplest path is to extend the existing SQLite database with asset-reference tables.

That keeps the first implementation small and easy to ship. If the data footprint grows later, we can split reference data into a separate database without changing the UI contract too much.

### `asset_master`

Purpose:

- one canonical record per asset

Suggested fields:

- `asset_id`
- `symbol`
- `display_symbol`
- `name`
- `asset_class`
- `security_type`
- `exchange`
- `mic`
- `country`
- `currency`
- `is_active`
- `is_delisted`
- `is_tradable`
- `first_seen_at`
- `last_seen_at`
- `created_at`
- `updated_at`

Notes:

- `asset_id` should be our internal canonical key
- `symbol` should not be the only identity field because the same symbol can collide across exchanges or asset types

### `asset_identifiers`

Purpose:

- store provider-specific identifiers and aliases

Suggested fields:

- `asset_identifier_id`
- `asset_id`
- `provider`
- `provider_symbol`
- `exchange_symbol`
- `conid`
- `isin`
- `cusip`
- `figi`
- `composite_key`
- `is_primary`
- `created_at`
- `updated_at`

This table is important because our acquisition and brokerage providers may identify the same asset differently.

### `asset_classifications`

Purpose:

- store normalized grouping metadata used for browsing and screening

Suggested fields:

- `asset_id`
- `sector`
- `industry`
- `industry_group`
- `theme`
- `market`
- `region`
- `country`
- `exchange`
- `issuer`
- `brand`
- `tags_json`
- `updated_at`

### `asset_profiles`

Purpose:

- store richer canonical descriptive metadata for the detail pane

Suggested fields:

- `asset_id`
- `description`
- `website`
- `employee_count`
- `ipo_date`
- `market_cap`
- `shares_outstanding`
- `avg_volume`
- `beta`
- `dividend_yield`
- `expense_ratio`
- `fund_family`
- `category_name`
- `raw_profile_json`
- `source`
- `as_of_date`
- `updated_at`

Not every field will apply to every asset class. Nullability is expected.

This table should remain mostly descriptive. It should not try to hold the full provider-specific fundamentals model by itself.

### `provider_credentials`

Purpose:

- store API/provider credentials needed for metadata enrichment

Suggested fields:

- `provider`
- `credential_label`
- `api_key_encrypted`
- `account_email`
- `base_url`
- `is_active`
- `last_validated_at`
- `created_at`
- `updated_at`

Notes:

- the user explicitly wants API keys stored in the database
- keys should be stored encrypted, not plaintext
- the UI should only ever show masked values
- initial use cases include `simfin` and any future providers that require account-based access

### `asset_metric_catalog`

Purpose:

- define the normalized metric names the app understands for screening and display

Suggested fields:

- `metric_id`
- `metric_name`
- `display_name`
- `category`
- `unit_type`
- `value_type`
- `description`
- `preferred_granularity`
- `created_at`
- `updated_at`

Examples of `category`:

- valuation
- profitability
- growth
- cash_flow
- balance_sheet
- market
- price_performance
- analyst

### `provider_dataset_catalog`

Purpose:

- inventory every provider dataset or data-domain we intend to sync so we do not silently omit entire categories of information

Suggested fields:

- `provider`
- `dataset_code`
- `dataset_group`
- `display_name`
- `description`
- `is_enabled`
- `is_structured`
- `supports_history`
- `supports_incremental_refresh`
- `supports_point_in_time`
- `last_documentation_reviewed_at`
- `created_at`
- `updated_at`

Examples of `dataset_group`:

- profile
- metrics
- statements
- price_history
- corporate_actions
- calendar
- filings
- transcripts
- news
- segment_breakdown
- geography_breakdown
- product_breakdown
- peer_benchmarks
- macro_reference

### `provider_raw_payloads`

Purpose:

- store raw provider payloads so new or unmapped fields are never lost during ingestion

Suggested fields:

- `raw_payload_id`
- `provider`
- `dataset_code`
- `asset_id`
- `provider_symbol`
- `provider_record_key`
- `as_of_date`
- `fiscal_year`
- `fiscal_period`
- `payload_json`
- `payload_hash`
- `fetched_at`

This table is one of the most important safeguards against missing fields.

### `provider_field_inventory`

Purpose:

- track every provider field we have observed so schema drift and missed mappings become visible

Suggested fields:

- `provider`
- `dataset_code`
- `field_path`
- `field_name`
- `observed_type`
- `first_seen_at`
- `last_seen_at`
- `mapping_status`
- `mapped_table`
- `mapped_column`
- `notes`

Recommended `mapping_status` values:

- `raw_only`
- `mapped`
- `ignored`
- `needs_review`

### `provider_sync_runs`

Purpose:

- audit provider sync jobs separately from acquisition jobs

Suggested fields:

- `provider_sync_run_id`
- `provider`
- `dataset_code`
- `started_at`
- `finished_at`
- `status`
- `asset_count`
- `record_count`
- `new_field_count`
- `error_summary`

### `defeatbeta_asset_profile_snapshots`

Purpose:

- store defeatbeta descriptive and company-profile snapshots without mixing them into other providers

Suggested fields:

- `snapshot_id`
- `asset_id`
- `provider_symbol`
- `as_of_date`
- `company_name`
- `description`
- `website`
- `sector`
- `industry`
- `exchange`
- `country`
- `currency`
- `market_cap`
- `employee_count`
- `raw_profile_json`
- `fetched_at`

### `defeatbeta_metric_snapshots`

Purpose:

- store defeatbeta-derived ratios, market metrics, and enrichment fields as provider-native facts

Suggested fields:

- `snapshot_id`
- `asset_id`
- `provider_symbol`
- `as_of_date`
- `price_reference_date`
- `market_cap`
- `enterprise_value`
- `ttm_eps`
- `ttm_pe_ratio`
- `forward_pe_ratio`
- `pb_ratio`
- `ps_ratio`
- `peg_ratio`
- `ev_to_ebitda`
- `ev_to_sales`
- `roe`
- `roic`
- `roa`
- `wacc`
- `beta`
- `dividend_yield`
- `payout_ratio`
- `revenue_ttm`
- `ebitda_ttm`
- `operating_income_ttm`
- `net_income_ttm`
- `fcf_ttm`
- `operating_cash_flow_ttm`
- `capex_ttm`
- `gross_margin`
- `operating_margin`
- `net_margin`
- `ebitda_margin`
- `revenue_growth_yoy`
- `eps_growth_yoy`
- `ebitda_growth_yoy`
- `fcf_growth_yoy`
- `price_change_1d`
- `price_change_1w`
- `price_change_1m`
- `price_change_3m`
- `price_change_6m`
- `price_change_1y`
- `avg_volume_30d`
- `raw_metrics_json`
- `fetched_at`

### `defeatbeta_financial_statements`

Purpose:

- store defeatbeta statement snapshots when available, separate from SimFin statement storage

Suggested fields:

- `statement_id`
- `asset_id`
- `provider_symbol`
- `statement_type`
- `period_type`
- `report_date`
- `fiscal_year`
- `fiscal_period`
- `statement_json`
- `fetched_at`

This is useful because defeatbeta documentation explicitly covers statement access in addition to ratios.

### `defeatbeta_filings`

Purpose:

- store SEC filing metadata and references from defeatbeta

Suggested fields:

- `filing_id`
- `asset_id`
- `provider_symbol`
- `filing_type`
- `filing_date`
- `accession_number`
- `document_url`
- `title`
- `filing_json`
- `fetched_at`

### `defeatbeta_officers`

Purpose:

- store company-officer and management metadata from defeatbeta

Suggested fields:

- `officer_id`
- `asset_id`
- `provider_symbol`
- `name`
- `title`
- `age`
- `year_born`
- `total_pay`
- `currency`
- `officer_json`
- `fetched_at`

### `defeatbeta_earnings_calendars`

Purpose:

- store earnings-calendar style dates and events from defeatbeta

Suggested fields:

- `calendar_event_id`
- `asset_id`
- `provider_symbol`
- `event_type`
- `report_date`
- `fiscal_year`
- `fiscal_quarter`
- `calendar_json`
- `fetched_at`

### `defeatbeta_transcripts`

Purpose:

- store earnings call transcripts and related metadata from defeatbeta

Suggested fields:

- `transcript_id`
- `asset_id`
- `provider_symbol`
- `report_date`
- `fiscal_year`
- `fiscal_quarter`
- `speaker_count`
- `transcript_json`
- `fetched_at`

### `defeatbeta_news_articles`

Purpose:

- store financial news linked to an asset from defeatbeta

Suggested fields:

- `news_id`
- `asset_id`
- `provider_symbol`
- `published_at`
- `title`
- `publisher`
- `url`
- `summary`
- `news_json`
- `fetched_at`

### `defeatbeta_corporate_actions`

Purpose:

- store dividends and splits from defeatbeta

Suggested fields:

- `corporate_action_id`
- `asset_id`
- `provider_symbol`
- `action_type`
- `ex_date`
- `record_date`
- `payment_date`
- `amount`
- `ratio`
- `action_json`
- `fetched_at`

### `defeatbeta_revenue_breakdowns`

Purpose:

- store revenue by segment, geography, or product from defeatbeta

Suggested fields:

- `breakdown_id`
- `asset_id`
- `provider_symbol`
- `breakdown_type`
- `report_date`
- `fiscal_year`
- `fiscal_period`
- `breakdown_name`
- `amount`
- `currency`
- `breakdown_json`
- `fetched_at`

### `defeatbeta_industry_metric_snapshots`

Purpose:

- store industry-relative benchmark metrics from defeatbeta

Suggested fields:

- `snapshot_id`
- `asset_id`
- `provider_symbol`
- `as_of_date`
- `industry_name`
- `industry_ttm_pe`
- `industry_ps_ratio`
- `industry_pb_ratio`
- `industry_roe`
- `industry_roa`
- `industry_equity_multiplier`
- `industry_net_margin`
- `industry_asset_turnover`
- `benchmark_json`
- `fetched_at`

### `simfin_income_statements`

Purpose:

- store SimFin income-statement facts as reported or as provided by the API

Suggested fields:

- `statement_id`
- `asset_id`
- `simfin_ticker`
- `simfin_company_id`
- `fiscal_year`
- `fiscal_period`
- `report_date`
- `publish_date`
- `currency`
- `revenue`
- `cost_of_revenue`
- `gross_profit`
- `operating_expense`
- `research_and_development`
- `selling_general_admin`
- `operating_income`
- `pretax_income`
- `income_tax_expense`
- `net_income`
- `diluted_eps`
- `weighted_avg_diluted_shares`
- `raw_statement_json`
- `fetched_at`

### `simfin_balance_sheets`

Purpose:

- store SimFin balance-sheet facts

Suggested fields:

- `statement_id`
- `asset_id`
- `simfin_ticker`
- `simfin_company_id`
- `fiscal_year`
- `fiscal_period`
- `report_date`
- `publish_date`
- `cash_and_equivalents`
- `short_term_investments`
- `inventory`
- `total_current_assets`
- `property_plant_equipment`
- `goodwill`
- `total_assets`
- `short_term_debt`
- `long_term_debt`
- `total_debt`
- `total_current_liabilities`
- `total_liabilities`
- `shareholders_equity`
- `book_value_per_share`
- `shares_outstanding`
- `raw_statement_json`
- `fetched_at`

### `simfin_cash_flow_statements`

Purpose:

- store SimFin cash-flow facts

Suggested fields:

- `statement_id`
- `asset_id`
- `simfin_ticker`
- `simfin_company_id`
- `fiscal_year`
- `fiscal_period`
- `report_date`
- `publish_date`
- `net_cash_from_operating_activities`
- `capital_expenditures`
- `net_cash_from_investing_activities`
- `net_cash_from_financing_activities`
- `free_cash_flow`
- `dividends_paid`
- `share_repurchases`
- `debt_issued`
- `debt_repaid`
- `raw_statement_json`
- `fetched_at`

### `simfin_metric_snapshots`

Purpose:

- store SimFin-calculated ratios and market-linked metrics separately from defeatbeta

Suggested fields:

- `snapshot_id`
- `asset_id`
- `simfin_ticker`
- `simfin_company_id`
- `as_of_date`
- `price_reference_date`
- `market_cap`
- `enterprise_value`
- `pe_ratio`
- `ps_ratio`
- `pb_ratio`
- `peg_ratio`
- `price_to_fcf`
- `ev_to_ebitda`
- `ev_to_sales`
- `fcf_yield`
- `gross_margin`
- `operating_margin`
- `net_margin`
- `ebitda_margin`
- `roe`
- `roa`
- `roic`
- `debt_to_equity`
- `net_debt_to_ebitda`
- `current_ratio`
- `quick_ratio`
- `interest_coverage`
- `revenue_growth_yoy`
- `revenue_growth_3y_cagr`
- `earnings_growth_yoy`
- `earnings_growth_3y_cagr`
- `ebitda_growth_yoy`
- `fcf_growth_yoy`
- `share_count_growth_yoy`
- `price_change_1m`
- `price_change_3m`
- `price_change_6m`
- `price_change_1y`
- `distance_from_52w_high`
- `distance_from_52w_low`
- `raw_metrics_json`
- `fetched_at`

### `simfin_shareprice_snapshots`

Purpose:

- store SimFin share-price history and volume when we use SimFin as a market-data context source for fundamentals and signals

Suggested fields:

- `price_snapshot_id`
- `asset_id`
- `simfin_ticker`
- `market`
- `report_date`
- `open`
- `high`
- `low`
- `close`
- `adjusted_close`
- `volume`
- `fetched_at`

### `simfin_indicator_facts`

Purpose:

- store SimFin indicators and less-common calculated metrics in a narrow format so we do not lose coverage beyond the curated wide tables

Suggested fields:

- `indicator_fact_id`
- `asset_id`
- `simfin_ticker`
- `as_of_date`
- `indicator_name`
- `indicator_category`
- `value_numeric`
- `value_text`
- `unit_type`
- `source_variant`
- `fetched_at`

This matters because SimFin's public materials indicate broad indicator and metric coverage that will exceed any hand-maintained wide table over time.

### `asset_metric_facts`

Purpose:

- provide an optional normalized metrics layer for the eventual screener while preserving source provenance

Suggested fields:

- `asset_metric_fact_id`
- `asset_id`
- `metric_id`
- `provider`
- `as_of_date`
- `fiscal_year`
- `fiscal_period`
- `value_numeric`
- `value_text`
- `unit_type`
- `confidence_rank`
- `source_record_id`
- `created_at`

Notes:

- this table should be derived from provider-specific tables, not written directly as the only source of truth
- it allows the screener to query a unified metrics model later
- provenance remains intact because every fact still points back to a provider-specific record

### `asset_status`

Purpose:

- track whether an asset is usable inside the platform

Suggested fields:

- `asset_id`
- `reference_status`
- `dataset_status`
- `latest_dataset_id`
- `latest_source`
- `latest_download_at`
- `latest_success_at`
- `latest_failure_at`
- `latest_failure_reason`
- `coverage_start`
- `coverage_end`
- `freshness_status`
- `updated_at`

This is what bridges asset metadata with acquisition health.

### `asset_metadata_refresh_runs`

Purpose:

- audit refresh jobs and keep provenance visible

Suggested fields:

- `refresh_run_id`
- `source`
- `started_at`
- `finished_at`
- `status`
- `asset_count`
- `inserted_count`
- `updated_count`
- `failed_count`
- `error_summary`

## Coverage and Field-Completeness Strategy

No hand-written field list will stay complete forever.

The provider docs are broad, and both defeatbeta and SimFin are still evolving. So the design should prevent omissions structurally, not just rely on us remembering every field name.

Recommended rules:

1. Every synced provider response should also be written to `provider_raw_payloads`.
2. Every discovered field path should be tracked in `provider_field_inventory`.
3. New fields should show up in a review queue instead of disappearing silently.
4. We should promote high-value fields into curated tables and UI columns only after they are mapped intentionally.
5. Provider-specific tables remain the source of truth; normalized tables and screener facts are derived layers.

This is the safest way to satisfy the goal of "collect everything possible" without forcing the schema to be rewritten every time a provider adds a field.

## Normalization Rules

To keep the catalog useful, we should normalize around a few simple rules:

- uppercase display symbols for consistency
- store provider-native identifiers separately from canonical identifiers
- preserve raw payloads where helpful, but never make the raw payload the primary query surface
- record source and freshness for each enriched field set
- allow sparse metadata instead of forcing fake defaults
- treat category mapping as versioned logic because source taxonomies will differ
- never merge defeatbeta and SimFin rows into a single providerless record
- support multiple values for the same logical metric when providers disagree
- make any later canonical metric selection explicit and auditable
- archive the raw provider payload before any transformation that could drop fields
- track newly observed provider fields as schema-drift events
- prefer narrow fact tables when a provider exposes too many metrics for a stable wide table

## Metric Families

The asset-information backbone should be designed to support much richer metric coverage than simple profile data.

Recommended first-pass metric families:

### Valuation

- `market_cap`
- `enterprise_value`
- `pe_ratio`
- `ttm_pe_ratio`
- `forward_pe_ratio`
- `pb_ratio`
- `ps_ratio`
- `peg_ratio`
- `price_to_fcf`
- `ev_to_ebitda`
- `ev_to_sales`

### Profitability and Quality

- `gross_margin`
- `operating_margin`
- `net_margin`
- `ebitda_margin`
- `roe`
- `roa`
- `roic`
- `wacc`
- `roic_minus_wacc`
- `interest_coverage`

### Growth

- `revenue_growth_yoy`
- `revenue_growth_qoq`
- `revenue_growth_3y_cagr`
- `eps_growth_yoy`
- `eps_growth_3y_cagr`
- `earnings_growth_yoy`
- `ebitda_growth_yoy`
- `fcf_growth_yoy`
- `book_value_growth_yoy`
- `share_count_growth_yoy`

### Cash Flow and Balance Sheet

- `revenue_ttm`
- `gross_profit_ttm`
- `ebitda_ttm`
- `operating_income_ttm`
- `net_income_ttm`
- `operating_cash_flow_ttm`
- `free_cash_flow_ttm`
- `capital_expenditures_ttm`
- `cash_and_equivalents`
- `total_debt`
- `net_debt`
- `current_ratio`
- `quick_ratio`
- `debt_to_equity`
- `net_debt_to_ebitda`

### Market and Price Performance

- `price`
- `avg_volume_30d`
- `avg_volume_90d`
- `beta`
- `dividend_yield`
- `payout_ratio`
- `price_change_1d`
- `price_change_1w`
- `price_change_1m`
- `price_change_3m`
- `price_change_6m`
- `price_change_1y`
- `distance_from_52w_high`
- `distance_from_52w_low`

### Statements, Events, and Reference Domains

- quarterly income statement line items
- quarterly balance sheet line items
- quarterly cash flow line items
- annual statement line items
- earnings calendar
- dividend history
- split history
- SEC filings
- officers and management data
- earnings call transcripts
- news
- revenue by segment
- revenue by geography
- revenue by product
- industry-relative benchmarks
- macro reference series such as Treasury yields or index return history when used for context

Not every provider will supply every metric.

That is fine.

The schema should be able to represent:

- direct provider values
- missing values
- later derived values we calculate ourselves

## Ingestion Pipeline

### Phase 1: Seed Asset Master

Use FinanceDatabase to populate:

- base symbol
- name
- exchange
- country
- currency
- asset class
- sector and industry where available

This creates broad initial coverage quickly.

### Phase 2: Normalize

Run a normalization pass that:

- maps source categories into our canonical `asset_class` and `security_type`
- deduplicates collisions when symbols overlap across venues
- creates canonical `asset_id` values
- writes provider identifiers into `asset_identifiers`

### Phase 3: Enrich

Use defeatbeta-api to enrich eligible assets with:

- company or asset profile fields
- descriptive metadata
- selected fundamental or profile-style attributes
- optional tags useful for future screening
- financial statements
- earnings calendar data
- dividends and splits
- SEC filings
- officers
- earnings call transcripts
- news
- revenue breakdowns by segment, geography, and product
- industry-relative benchmark ratios

Use SimFin to enrich eligible assets with:

- quarterly and annual statement data
- historical derived metrics
- valuation ratios
- cash-flow and balance-sheet measures
- growth metrics suitable for screening
- price-history context
- broad indicator and signal coverage where available

Do not write defeatbeta and SimFin payloads into the same storage row.

Persist them into separate provider-specific tables and only derive cross-provider views afterward when needed.

### Phase 4: Refresh and Audit

Run periodic metadata refresh jobs that:

- update changed fields
- timestamp freshness
- track failures without deleting older valid metadata
- store refresh history for debugging and trust
- validate API credentials and subscription availability where applicable

## Asset Information Tab UI

The tab should be a browse-first interface.

Recommended layout:

### Left Panel: Category and Filter Controls

- asset class tree
- exchange filter
- country filter
- sector filter
- industry filter
- provider availability filter
- provider source toggle
- data-domain availability filter
- valuation metric filters
- growth metric filters
- profitability metric filters
- event and reference-data filters
- active/delisted toggle
- has dataset toggle
- freshness/status toggle
- search box

This should let the user quickly answer questions like:

- show me all U.S. equities
- show me all crypto assets
- show me semiconductor equities with local datasets
- show me assets that exist in the catalog but have never been downloaded successfully

### Center Panel: Grouped Asset Table

Primary columns:

- Symbol
- Name
- Asset Class
- Security Type
- Exchange
- Country
- Sector
- Industry
- Market Cap
- P/E
- P/S
- EV/EBITDA
- Revenue Growth
- EPS Growth
- FCF
- Next Earnings Date
- Transcript Coverage
- Filing Coverage
- Dataset Status
- Freshness

Recommended behaviors:

- group by asset class by default
- allow regrouping by sector, exchange, or country later
- support fast filtering without re-querying external providers
- support multiselect
- support quick actions such as `Add to Universe`, `Open Datasets`, and later `Screen Similar`

### Right Panel: Asset Detail

Recommended sections:

- Overview
- Classification
- Identifiers
- Provider Data
- Financial Statements
- Valuation
- Growth
- Profitability
- Cash Flow and Balance Sheet
- Corporate Actions
- Filings
- Transcripts
- News
- Segment and Geography Breakdown
- Dataset Coverage
- Acquisition History

Example fields:

- full name
- description
- exchange
- currency
- country
- sector
- industry
- website
- identifiers
- market cap
- enterprise value
- P/E
- P/S
- PEG
- EV/EBITDA
- EBITDA
- revenue
- earnings
- free cash flow
- sales growth
- earnings growth
- FCF growth
- recent percent-change windows
- dividend and split history
- latest filings
- latest transcript availability
- recent news coverage
- segment or geography revenue breakdowns
- provider/source provenance
- last metadata refresh
- last successful dataset acquisition

## Integration With Existing Workflows

The tab should connect to the rest of the platform, not sit off to the side.

Recommended integration points:

- `Universe Builder`
  - add selected assets from the asset catalog directly into a universe
- `Data Collection`
  - show whether an asset has downloadable provider support and existing dataset coverage
- `Acquisition Catalog`
  - jump from an asset to its dataset variants and download history
- future `Research` flows
  - use transcripts, filings, and news as context for candidate review
- `Backtest` and `Optimization`
  - eventually resolve a universe built from screened assets into concrete dataset selections

## Screener Relationship

The future screener should be built on top of this asset master, not in parallel with it.

That means the screener should query normalized tables such as:

- `asset_master`
- `asset_classifications`
- `asset_profiles`
- `asset_status`
- `asset_metric_facts`

Examples of first useful screener rules:

- asset class is `Equity`
- country is `United States`
- exchange in `NASDAQ`, `NYSE`
- sector is `Technology`
- market cap greater than threshold
- P/E less than threshold
- EV/EBITDA less than threshold
- revenue growth greater than threshold
- earnings growth greater than threshold
- free cash flow positive
- transcript available in the last N quarters
- latest filing within expected recency window
- no recent dividend cut if we later derive that signal
- price above 200-day moving average if we later expose technical filters
- dataset status is `Ready`
- freshness status is `Fresh`

This creates a clean path:

1. sync reference data
2. browse and inspect assets
3. filter with screener rules
4. save results into a universe
5. use that universe in acquisition and research workflows

## Recommended Implementation Phases

### Phase 1: Reference Backbone

- add asset-reference tables
- create FinanceDatabase import script
- create normalized category mapping
- add provider credential storage for enrichment sources
- add provider dataset inventory and raw-payload archive tables
- add simple catalog queries

### Phase 2: Asset Information Tab

- add tab shell
- add category filters
- add grouped asset table
- add asset detail pane
- add `Add to Universe` action

### Phase 3: Enrichment

- add defeatbeta-api enrichment pipeline
- add SimFin enrichment pipeline
- keep provider-specific storage separate
- add raw payload archiving and field-inventory discovery
- store provider provenance and refresh timestamps
- expose richer asset detail fields

### Phase 4: Screener

- add saved filter definitions
- add result preview counts
- add `Save as Universe`
- add quick handoff into Data Collection and Backtest workflows

## Risks and Design Notes

- Source taxonomies will not line up perfectly, so category mapping must be explicit and versioned.
- Some asset classes will have sparse metadata. The UI should show missing data cleanly instead of pretending all assets have the same fields.
- Symbol collisions across exchanges are real. Canonical identity cannot rely on ticker alone.
- Commodities and futures may need a more careful contract model later. Version 1 can still categorize them at the asset level without solving every contract-detail problem immediately.
- The first version should bias toward a clean asset catalog and dependable filtering, not maximal metadata density.
- Provider disagreement is expected. The UI should be able to show which provider a metric came from, especially for valuation and growth fields.
- SimFin licensing and retention rules should be reviewed carefully before deciding which downloaded fields are cached long-term and how they may be reused.
- Text-heavy datasets such as transcripts and news can grow quickly, so retention and indexing strategy should be decided up front.
- Not all public documentation pages enumerate every field. Raw-payload capture and field inventory are therefore required if we want confidence that nothing important is being dropped.

## Recommended First Deliverables

If we build this incrementally, the best first concrete milestone is:

- create the asset-reference schema
- build a FinanceDatabase import script
- add provider credential storage with masked UI editing
- add provider dataset inventory, raw payload archive, and field-inventory tracking
- add a read-only `Asset Information` tab with category browsing and a detail panel
- add `Add Selected to Universe`

The next milestone after that should be:

- add defeatbeta profile/metrics ingestion
- add SimFin statement and metric ingestion
- expose provider-separated detail panes before attempting a unified screener layer

That is enough to make the platform noticeably better before the screener even exists.

## References

- FinanceDatabase: https://github.com/jerbouma/FinanceDatabase?tab=readme-ov-file#usage
- defeatbeta-api: https://github.com/defeat-beta/defeatbeta-api/tree/main
- defeatbeta-api README: https://raw.githubusercontent.com/defeat-beta/defeatbeta-api/main/README.md
- defeatbeta-api usage index: https://raw.githubusercontent.com/defeat-beta/defeatbeta-api/main/doc/README.md
- SimFin Data API: https://www.simfin.com/en/fundamental-data-download/
- SimFin Python API docs: https://simfin.readthedocs.io/en/latest/
- SimFin pricing and plan limits: https://www.simfin.com/en/prices/
