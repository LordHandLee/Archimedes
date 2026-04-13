from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest_engine.asset_reference import (
    FINANCEDATABASE_ASSET_SPECS,
    financedatabase_available,
    financedatabase_install_hint,
    import_financedatabase_assets,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import reference assets from FinanceDatabase into the local SQLite catalog."
    )
    parser.add_argument(
        "--catalog",
        default="backtests.sqlite",
        help="Path to the SQLite catalog. Defaults to backtests.sqlite.",
    )
    parser.add_argument(
        "--asset-classes",
        default="all",
        help=(
            "Comma-separated FinanceDatabase asset classes to import. "
            f"Valid values: all,{','.join(spec.code for spec in FINANCEDATABASE_ASSET_SPECS)}"
        ),
    )
    parser.add_argument(
        "--only-primary-listing",
        action="store_true",
        help="Request only primary listings when the FinanceDatabase asset class supports it.",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=0,
        help="Optional row limit per asset class for testing.",
    )
    parser.add_argument(
        "--store-raw-payloads",
        action="store_true",
        help="Also archive raw provider payload rows in provider_raw_payloads.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not financedatabase_available():
        print(f"Missing Python dependency `financedatabase`. {financedatabase_install_hint()}", file=sys.stderr)
        return 1

    asset_classes = None
    if str(args.asset_classes).strip().lower() != "all":
        asset_classes = [item.strip().lower() for item in str(args.asset_classes).split(",") if item.strip()]

    try:
        summary = import_financedatabase_assets(
            catalog_path=Path(args.catalog),
            asset_classes=asset_classes,
            only_primary_listing=bool(args.only_primary_listing),
            limit_per_class=(int(args.limit_per_class) if int(args.limit_per_class) > 0 else None),
            store_raw_payloads=bool(args.store_raw_payloads),
        )
    except Exception as exc:
        print(f"FinanceDatabase import failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
