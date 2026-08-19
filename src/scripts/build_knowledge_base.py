"""
Turn a candlestick CSV (as produced by hyperliquid_data.sync_csv) into a RAG
knowledge store: plain-English summaries at multiple granularities, embedded
with an OpenAI embedding model and persisted to a local Qdrant vector
database.

Four tiers of documents are created so the agent can retrieve at whatever
granularity a question calls for:
  - candle  : one document per raw candle, at whatever interval the data is
              in (e.g. "HYPE candlestick for 2025-03-04 08:00 UTC (4h
              interval)..."). With sub-daily intervals like 4h there are
              several of these per calendar day.
  - daily   : one document per calendar day, rolled up from that day's candles
  - weekly  : one document per ISO week, rolled up from that week's candles
  - monthly : one document per calendar month, rolled up from that month's candles

Each document carries metadata (doc_type, start_date, end_date, OHLCV,
pct_change) so it can be filtered/inspected even outside of semantic search.

This script requires OPENAI_API_KEY (used to embed the documents). If run
without --csv, it also syncs (fetches/appends) fresh data from Hyperliquid
first via the shared hyperliquid_data module -- pass --csv to point at an
already-fetched file instead and skip the network call.

Usage:
    python src/scripts/build_knowledge_base.py                # auto-syncs HL_COIN/HL_INTERVAL data first
    python src/scripts/build_knowledge_base.py --csv resources/data/hype_4h_candles.csv
"""

import argparse
import os
import sys
import uuid
from pathlib import Path

import pandas as pd

# This file lives at <project_root>/src/scripts/build_knowledge_base.py.
# Three `.parent`s up from here is <project_root>; put <project_root>/src on
# sys.path so `import hyperliquid_data` resolves regardless of cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESOURCES_DIR = PROJECT_ROOT / "resources"
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from dotenv import load_dotenv

    load_dotenv(RESOURCES_DIR / ".env")
except ImportError:
    pass

from hyperliquid_data import default_csv_path, sync_csv  # noqa: E402

DEFAULT_PERSIST_DIR = RESOURCES_DIR / "qdrant_store"
DEFAULT_COLLECTION = os.getenv("QDRANT_COLLECTION", "candlestick_knowledge")

NOTABLE_MOVE_STD_MULTIPLIER = 1.5


def load_candles(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["open_time", "close_time"])
    df = df.sort_values("open_time").reset_index(drop=True)
    df["pct_change"] = (df["close"] - df["open"]) / df["open"] * 100.0
    df["range_pct"] = (df["high"] - df["low"]) / df["open"] * 100.0
    df["direction"] = df["pct_change"].apply(
        lambda x: "bullish" if x > 0.05 else ("bearish" if x < -0.05 else "flat/doji")
    )
    move_std = df["pct_change"].std()
    move_mean_abs = df["pct_change"].abs().mean()
    threshold = max(move_mean_abs * NOTABLE_MOVE_STD_MULTIPLIER, move_std)
    df["is_notable_move"] = df["pct_change"].abs() >= threshold
    return df


def _fmt_date(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _fmt_datetime_for_id(ts) -> str:
    """Filesystem/id-safe full timestamp, e.g. '2025-03-04T08-00'. Needed
    because with sub-daily intervals (4h, 1h, ...) there are multiple
    candles per calendar day -- a date-only id would collide and silently
    overwrite earlier candles' documents in the vector store."""
    return pd.Timestamp(ts).strftime("%Y-%m-%dT%H-%M")


def _candle_doc(row, coin: str) -> dict:
    date_str = _fmt_date(row["open_time"])
    datetime_str = pd.Timestamp(row["open_time"]).strftime("%Y-%m-%d %H:%M UTC")
    notable = " This was an unusually large move relative to recent history." if row["is_notable_move"] else ""
    text = (
        f"{coin} candlestick for {datetime_str} ({row['interval']} interval). "
        f"Open: {row['open']:.4f}, High: {row['high']:.4f}, Low: {row['low']:.4f}, "
        f"Close: {row['close']:.4f}, Volume: {row['volume']:.2f}, Trades: {int(row['num_trades'])}. "
        f"Price moved {row['pct_change']:+.2f}% from open to close ({row['direction']}), "
        f"with a range of {row['range_pct']:.2f}% of the open price.{notable}"
    )
    return {
        "id": f"candle-{coin}-{_fmt_datetime_for_id(row['open_time'])}",
        "text": text,
        "metadata": {
            "doc_type": "candle",
            "coin": coin,
            "start_date": date_str,
            "end_date": date_str,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "pct_change": float(row["pct_change"]),
            "direction": row["direction"],
            "is_notable_move": bool(row["is_notable_move"]),
        },
    }


def _rollup_doc(period_df: pd.DataFrame, coin: str, doc_type: str, label: str) -> dict:
    open_ = period_df.iloc[0]["open"]
    close = period_df.iloc[-1]["close"]
    high = period_df["high"].max()
    low = period_df["low"].min()
    volume = period_df["volume"].sum()
    pct_change = (close - open_) / open_ * 100.0
    start_date = _fmt_date(period_df.iloc[0]["open_time"])
    end_date = _fmt_date(period_df.iloc[-1]["open_time"])
    up_periods = int((period_df["pct_change"] > 0.05).sum())
    down_periods = int((period_df["pct_change"] < -0.05).sum())
    notable = period_df[period_df["is_notable_move"]]
    notable_note = ""
    if not notable.empty:
        biggest = notable.loc[notable["pct_change"].abs().idxmax()]
        notable_note = (
            f" The largest single move was {biggest['pct_change']:+.2f}% on "
            f"{_fmt_date(biggest['open_time'])}."
        )
    trend = "up" if pct_change > 0.05 else ("down" if pct_change < -0.05 else "roughly flat")

    text = (
        f"{coin} {doc_type} summary for {label} ({start_date} to {end_date}). "
        f"Opened at {open_:.4f} and closed at {close:.4f}, a change of {pct_change:+.2f}% "
        f"({trend} overall). Period high: {high:.4f}, period low: {low:.4f}. "
        f"Total volume: {volume:.2f} across {len(period_df)} candles "
        f"({up_periods} up, {down_periods} down).{notable_note}"
    )
    return {
        "id": f"{doc_type}-{coin}-{start_date}",
        "text": text,
        "metadata": {
            "doc_type": doc_type,
            "coin": coin,
            "start_date": start_date,
            "end_date": end_date,
            "open": float(open_),
            "high": float(high),
            "low": float(low),
            "close": float(close),
            "volume": float(volume),
            "pct_change": float(pct_change),
            "direction": trend,
            "is_notable_move": bool(not notable.empty),
        },
    }


def build_documents(df: pd.DataFrame, coin: str) -> list[dict]:
    """Pure function: candle DataFrame -> list of {id, text, metadata} docs.

    Kept separate from any embedding/DB calls so it can be unit-tested
    without an OpenAI API key or network access.
    """
    docs = [_candle_doc(row, coin) for _, row in df.iterrows()]

    naive_open_time = df["open_time"].dt.tz_localize(None) if df["open_time"].dt.tz is not None else df["open_time"]

    daily_key = naive_open_time.dt.to_period("D")
    for period, group in df.groupby(daily_key):
        label = _fmt_date(group.iloc[0]["open_time"])
        docs.append(_rollup_doc(group, coin, "daily", label))

    weekly_key = naive_open_time.dt.to_period("W")
    for period, group in df.groupby(weekly_key):
        label = f"week of {_fmt_date(group.iloc[0]['open_time'])}"
        docs.append(_rollup_doc(group, coin, "weekly", label))

    monthly_key = naive_open_time.dt.to_period("M")
    for period, group in df.groupby(monthly_key):
        label = str(period)  # e.g. "2025-03"
        docs.append(_rollup_doc(group, coin, "monthly", label))

    return docs


def _get_qdrant_client(persist_dir: Path):
    """Local on-disk Qdrant by default; set QDRANT_URL to talk to a real
    Qdrant server / Qdrant Cloud instance instead."""
    from qdrant_client import QdrantClient

    url = os.getenv("QDRANT_URL")
    if url:
        return QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY"))
    return QdrantClient(path=str(persist_dir))


def _embed_texts(texts: list[str], embedding_model: str, api_key: str, batch_size: int = 100) -> list[list[float]]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=embedding_model, input=batch)
        vectors.extend([item.embedding for item in resp.data])
    return vectors


def persist_to_qdrant(docs: list[dict], persist_dir: Path, collection_name: str, embedding_model: str, coin: str):
    from qdrant_client.models import Distance, PointStruct, VectorParams

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in your .env file before building the knowledge base.")

    texts = [d["text"] for d in docs]
    vectors = _embed_texts(texts, embedding_model, api_key)
    vector_size = len(vectors[0])

    qdrant = _get_qdrant_client(persist_dir)

    # Reset any previous collection for this coin so re-runs don't duplicate/stale-out docs.
    if qdrant.collection_exists(collection_name):
        qdrant.delete_collection(collection_name)
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    # Qdrant point IDs must be an int or a UUID -- our human-readable doc ids
    # (e.g. "daily-HYPE-2025-03-04") aren't valid IDs, so we derive a stable
    # UUID from each one and keep the original string in the payload as
    # "doc_id" for reference/debugging.
    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, doc["id"])),
            vector=vector,
            payload={**doc["metadata"], "doc_id": doc["id"], "text": doc["text"]},
        )
        for doc, vector in zip(docs, vectors)
    ]

    batch_size = 100
    for i in range(0, len(points), batch_size):
        qdrant.upsert(collection_name=collection_name, points=points[i : i + batch_size])

    return qdrant, collection_name, len(points)


def main():
    parser = argparse.ArgumentParser(description="Build the RAG knowledge base from a candlestick CSV.")
    parser.add_argument(
        "--csv",
        default=None,
        help=(
            "Path to an already-fetched candlestick CSV. If omitted, this script "
            "syncs (backfills or appends) fresh data from Hyperliquid first via "
            "the shared hyperliquid_data module, using --coin/--interval."
        ),
    )
    parser.add_argument("--coin", default=os.getenv("HL_COIN", "HYPE"))
    parser.add_argument("--interval", default=os.getenv("HL_INTERVAL", "4h"), help="Candle interval, e.g. 4h, 1d")
    parser.add_argument("--persist-dir", default=str(DEFAULT_PERSIST_DIR))
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embedding-model", default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    args = parser.parse_args()

    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            raise SystemExit(f"CSV not found: {csv_path}. Omit --csv to auto-fetch from Hyperliquid instead.")
    else:
        csv_path = default_csv_path(args.coin, args.interval)
        sync_csv(args.coin, args.interval, csv_path)

    df = load_candles(csv_path)
    docs = build_documents(df, args.coin)
    print(f"Built {len(docs)} documents from {len(df)} candles "
          f"({sum(1 for d in docs if d['metadata']['doc_type'] == 'candle')} candle, "
          f"{sum(1 for d in docs if d['metadata']['doc_type'] == 'daily')} daily, "
          f"{sum(1 for d in docs if d['metadata']['doc_type'] == 'weekly')} weekly, "
          f"{sum(1 for d in docs if d['metadata']['doc_type'] == 'monthly')} monthly).")

    persist_dir = Path(args.persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    _qdrant, collection_name, num_points = persist_to_qdrant(
        docs, persist_dir, args.collection, args.embedding_model, args.coin
    )
    print(f"Persisted {num_points} documents to Qdrant collection "
          f"'{collection_name}' at {persist_dir}"
          + (f" (server: {os.environ['QDRANT_URL']})" if os.getenv("QDRANT_URL") else ""))


if __name__ == "__main__":
    main()