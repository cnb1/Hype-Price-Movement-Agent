"""
CLI wrapper around hyperliquid_data.sync_csv() for manual or scheduled use.

You normally don't need to run this yourself: the agent (src/hype_agent/
tools.py) calls the same sync logic automatically the first time it needs
data in a given run -- on first use it backfills full history, on later runs
it picks up wherever the CSV left off and appends anything new. This script
exists for cases where you want to pre-warm the data (e.g. before building
the knowledge base), force a full refresh, or run the sync on a cron
schedule independent of the agent.

Usage:
    python src/scripts/fetch_hyperliquid_data.py
    python src/scripts/fetch_hyperliquid_data.py --coin HYPE --interval 4h
    python src/scripts/fetch_hyperliquid_data.py --full-refresh
"""

import argparse
import os
import sys
from pathlib import Path

# This file lives at <project_root>/src/scripts/fetch_hyperliquid_data.py.
# Put <project_root>/src on sys.path so `import hyperliquid_data` resolves
# regardless of the current working directory this script is invoked from.
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hyperliquid_data import RESOURCES_DIR, default_csv_path, sync_csv  # noqa: E402

try:
    from dotenv import load_dotenv

    load_dotenv(RESOURCES_DIR / ".env")
except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser(description="Fetch/update Hyperliquid candlestick history to CSV.")
    parser.add_argument("--coin", default=os.getenv("HL_COIN", "HYPE"), help="Coin/asset symbol, e.g. HYPE")
    parser.add_argument("--interval", default=os.getenv("HL_INTERVAL", "4h"), help="Candle interval, e.g. 4h, 1d")
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: resources/data/<coin>_<interval>_candles.csv)",
    )
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Discard any existing CSV for this coin/interval and re-backfill full history from scratch",
    )
    args = parser.parse_args()

    csv_path = Path(args.out) if args.out else default_csv_path(args.coin, args.interval)

    if args.full_refresh and csv_path.exists():
        print(f"--full-refresh given: removing existing {csv_path} before re-backfilling.")
        csv_path.unlink()

    df = sync_csv(args.coin, args.interval, csv_path)
    print(f"\n{csv_path} now has {len(df)} candles, range {df['open_time'].iloc[0]} -> {df['open_time'].iloc[-1]}")


if __name__ == "__main__":
    main()