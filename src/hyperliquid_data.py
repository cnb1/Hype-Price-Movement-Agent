"""
Shared Hyperliquid candlestick data client: raw fetch, paginated full-history
backfill, and CSV sync (create on first run, append/refresh on later runs).

Used by both:
  - src/scripts/fetch_hyperliquid_data.py  (manual/cron CLI, optional)
  - src/hype_agent/tools.py                (automatic sync the first time the
    agent needs data in a given run)

Hyperliquid Info API reference:
  POST https://api.hyperliquid.xyz/info
  body: {
    "type": "candleSnapshot",
    "req": {"coin": "HYPE", "interval": "4h", "startTime": <ms>, "endTime": <ms>}
  }

Notes:
- No API key is required for this read-only market-data endpoint.
- Each request returns at most the most recent PAGE_CAP candles up to
  `endTime` for that coin/interval. fetch_full_history() pages backward in
  time (moving `endTime` to just before the oldest candle seen so far) to
  assemble as much history as Hyperliquid will actually give up, in case
  that cap is per-request rather than a hard ceiling on total available
  history -- if it turns out to be a hard ceiling, the "no progress" guard
  below stops the loop cleanly instead of spinning.
- Valid intervals: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 8h, 12h, 1d, 3d, 1w, 1M
"""

import time
from pathlib import Path

import pandas as pd
import requests

# This file lives at <project_root>/src/hyperliquid_data.py, so one
# `.parent` up from here is <project_root>.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESOURCES_DIR = PROJECT_ROOT / "resources"
DATA_DIR = RESOURCES_DIR / "data"

HL_INFO_URL = "https://api.hyperliquid.xyz/info"
PAGE_CAP = 5000  # Hyperliquid returns at most this many candles per request

CSV_COLUMNS = [
    "open_time", "close_time", "symbol", "interval",
    "open", "high", "low", "close", "volume", "num_trades",
]


def _now_ms() -> int:
    return int(time.time() * 1000)


def default_csv_path(coin: str, interval: str) -> Path:
    return DATA_DIR / f"{coin.lower()}_{interval}_candles.csv"


def fetch_candles(coin: str, interval: str, start_ms: int, end_ms: int, timeout: int = 30) -> list[dict]:
    """Call Hyperliquid's candleSnapshot info endpoint and return raw candle dicts."""
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": coin,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
        },
    }
    resp = requests.post(HL_INFO_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise ValueError(f"Unexpected response shape from Hyperliquid: {data!r}")
    return data


def candles_to_dataframe(raw_candles: list[dict]) -> pd.DataFrame:
    """Convert raw Hyperliquid candle dicts (t,T,s,i,o,c,h,l,v,n) into a tidy DataFrame, sorted ascending by open_time."""
    if not raw_candles:
        return pd.DataFrame(columns=CSV_COLUMNS)

    df = pd.DataFrame(raw_candles)
    df = df.rename(
        columns={
            "t": "open_time_ms",
            "T": "close_time_ms",
            "s": "symbol",
            "i": "interval",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "n": "num_trades",
        }
    )

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df["num_trades"] = df["num_trades"].astype(int)

    df["open_time"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time_ms"], unit="ms", utc=True)

    df = df[CSV_COLUMNS]
    df = df.sort_values("open_time").drop_duplicates(subset=["open_time"]).reset_index(drop=True)
    return df


def fetch_full_history(
    coin: str,
    interval: str,
    end_ms: int | None = None,
    max_pages: int = 50,
    request_timeout: int = 30,
    sleep_between_pages: float = 0.2,
    log=print,
) -> pd.DataFrame:
    """
    Page backward in time to assemble as much history as Hyperliquid will
    give up for this coin/interval, working around the per-request candle
    cap. Stops when:
      - a page comes back with fewer than PAGE_CAP candles (true start of
        history reached), or
      - a page makes no progress toward older data (guards against the cap
        actually being a hard ceiling on total history rather than
        per-request, which would otherwise return the same window forever), or
      - max_pages is hit (hard backstop).
    """
    cursor_end = end_ms if end_ms is not None else _now_ms()
    pages: list[pd.DataFrame] = []
    previous_earliest_ms = None

    for page_num in range(1, max_pages + 1):
        raw = fetch_candles(coin, interval, start_ms=0, end_ms=cursor_end, timeout=request_timeout)
        if not raw:
            break
        page_df = candles_to_dataframe(raw)
        pages.append(page_df)
        page_earliest_ms = int(page_df["open_time"].iloc[0].timestamp() * 1000)
        log(
            f"  fetched page {page_num}: {len(page_df)} candles "
            f"({page_df['open_time'].iloc[0]} -> {page_df['open_time'].iloc[-1]})"
        )

        if len(raw) < PAGE_CAP:
            break  # fewer than the cap -> this page reached the true start of history
        if previous_earliest_ms is not None and page_earliest_ms >= previous_earliest_ms:
            break  # no progress toward older data -- stop rather than loop forever
        previous_earliest_ms = page_earliest_ms
        cursor_end = page_earliest_ms - 1
        if sleep_between_pages:
            time.sleep(sleep_between_pages)

    if not pages:
        return candles_to_dataframe([])

    combined = pd.concat(pages, ignore_index=True)
    combined = combined.sort_values("open_time").drop_duplicates(subset="open_time", keep="last").reset_index(drop=True)
    return combined


def sync_csv(coin: str, interval: str, csv_path, log=print) -> pd.DataFrame:
    """
    Ensure csv_path holds up-to-date candlestick history for coin/interval:
      - Missing or empty file: backfill full available history from Hyperliquid.
      - Existing file: fetch from the last stored candle's open time forward
        through now (re-fetching that last candle in case it was still
        "in progress" the previous time it was fetched), and merge in
        anything new -- freshly fetched data wins on any overlapping
        timestamp.
    Writes the result back to csv_path and returns it as a DataFrame.
    """
    csv_path = Path(csv_path)
    existing = None
    if csv_path.exists() and csv_path.stat().st_size > 0:
        existing = pd.read_csv(csv_path, parse_dates=["open_time", "close_time"])
        if existing.empty:
            existing = None

    if existing is None:
        log(f"No existing data for {coin} {interval} -- backfilling full history from Hyperliquid...")
        df = fetch_full_history(coin, interval, log=log)
        if df.empty:
            raise RuntimeError(
                f"Hyperliquid returned no candle data for coin={coin!r} interval={interval!r}. "
                "Check that the coin symbol and interval are valid."
            )
        log(f"Backfilled {len(df)} candles ({df['open_time'].iloc[0]} -> {df['open_time'].iloc[-1]}).")
    else:
        existing = existing.sort_values("open_time").reset_index(drop=True)
        last_open_ms = int(pd.Timestamp(existing["open_time"].iloc[-1]).timestamp() * 1000)
        log(
            f"Found {len(existing)} existing candles (latest: {existing['open_time'].iloc[-1]}). "
            "Fetching anything new since then..."
        )
        raw = fetch_candles(coin, interval, start_ms=last_open_ms, end_ms=_now_ms())
        new_df = candles_to_dataframe(raw)
        if new_df.empty:
            log("Already up to date -- no new candles returned.")
            df = existing
        else:
            combined = pd.concat([existing, new_df], ignore_index=True)
            df = combined.sort_values("open_time").drop_duplicates(subset="open_time", keep="last").reset_index(drop=True)
            log(
                f"Fetched {len(new_df)} candles from Hyperliquid; {len(df) - len(existing)} were net-new "
                f"after de-duplication. Now {len(df)} candles total."
            )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df