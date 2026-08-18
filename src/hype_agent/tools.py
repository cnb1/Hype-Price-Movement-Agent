"""
Tools for the candlestick price-analyst agent.

Two complementary paths, on purpose:

1. semantic_search_price_history()  -- classic RAG: embeds the user's query
   and does a nearest-neighbor search over pre-written daily/weekly/monthly
   summaries in the Qdrant vector store. Good for fuzzy, qualitative, or
   exploratory questions where you don't already know a precise condition to
   check ("when did HYPE have a big rally", "was there a sharp drawdown in
   the spring").

2. run_data_analysis()  -- a code-execution tool: the LLM writes pandas code,
   it runs against the real candlestick DataFrame, and the exact computed
   result comes back. This replaces a fixed menu of hand-written lookup
   functions (which can only ever answer the specific question shapes their
   author anticipated) with something that generalizes to *any* numeric or
   analytical question -- streaks, rolling windows, aggregates over an
   arbitrary range, comparisons, whatever -- because the model writes the
   exact computation the question calls for instead of picking from a
   pre-built menu. Anything that needs a guaranteed-correct number should go
   through this, not through semantic search, which only returns
   approximate textual neighbors.

The agent's instructions (see agent.py) tell the LLM when to reach for each.
"""

import builtins
import contextlib
import functools
import io
import os
import signal
from pathlib import Path

import numpy as np
import pandas as pd

# This file lives at <project_root>/src/hype_agent/tools.py, so three
# `.parent`s up from here is <project_root>.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESOURCES_DIR = PROJECT_ROOT / "resources"

try:
    from dotenv import load_dotenv

    load_dotenv(RESOURCES_DIR / ".env")
except ImportError:
    pass

DATA_DIR = RESOURCES_DIR / "data"
QDRANT_DIR = RESOURCES_DIR / "qdrant_store"
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "candlestick_knowledge")

COIN = os.getenv("HL_COIN", "HYPE")
INTERVAL = os.getenv("HL_INTERVAL", "1d")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def _default_csv_path() -> Path:
    return DATA_DIR / f"{COIN.lower()}_{INTERVAL}_candles.csv"


@functools.lru_cache(maxsize=1)
def _load_dataframe() -> pd.DataFrame:
    csv_path = _default_csv_path()
    if not csv_path.exists():
        matches = sorted(DATA_DIR.glob(f"{COIN.lower()}_*_candles.csv"))
        if matches:
            csv_path = matches[0]
        else:
            raise FileNotFoundError(
                f"No candlestick CSV found for {COIN} in {DATA_DIR}. "
                "Run src/scripts/fetch_hyperliquid_data.py first."
            )
    df = pd.read_csv(csv_path, parse_dates=["open_time", "close_time"])
    if df["open_time"].dt.tz is not None:
        df["open_time"] = df["open_time"].dt.tz_localize(None)
    df["date"] = df["open_time"].dt.date.astype(str)
    df["pct_change"] = (df["close"] - df["open"]) / df["open"] * 100.0
    return df.sort_values("open_time").reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def _get_qdrant_client():
    """Local on-disk Qdrant by default; set QDRANT_URL to talk to a real
    Qdrant server / Qdrant Cloud instance instead (must match whatever
    src/scripts/build_knowledge_base.py was run against)."""
    from qdrant_client import QdrantClient

    url = os.getenv("QDRANT_URL")
    if url:
        return QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY"))
    return QdrantClient(path=str(QDRANT_DIR))


@functools.lru_cache(maxsize=1)
def _get_openai_client():
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to resources/.env.")
    return OpenAI(api_key=api_key)


def semantic_search_price_history(query: str, n_results: int = 5) -> dict:
    """Semantically search the RAG knowledge base of price-history summaries.

    Use this for fuzzy, qualitative, or exploratory questions where you don't
    already know the exact date(s) involved -- e.g. "when did the price spike
    hard", "was there a big selloff around some catalyst", "how did the asset
    trend over the quarter". It searches daily, weekly, and monthly summary
    documents by semantic similarity to the query.

    For questions that need an exact number for a KNOWN date or date range,
    prefer run_data_analysis instead -- this search returns approximate
    textual matches, not guaranteed-precise figures.

    Args:
        query (str): Natural-language description of what you're looking for.
        n_results (int): How many matching summaries to return (1-20).

    Returns:
        dict with status, the original query, and a list of results. Each
        result has: summary (text), doc_type (daily/weekly/monthly),
        start_date, end_date, pct_change, direction.
    """
    try:
        qdrant = _get_qdrant_client()
        openai_client = _get_openai_client()
        if not qdrant.collection_exists(COLLECTION_NAME):
            raise RuntimeError(
                f"Qdrant collection '{COLLECTION_NAME}' does not exist yet. "
                "Run src/scripts/build_knowledge_base.py first."
            )
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

    n_results = max(1, min(int(n_results), 20))
    query_vector = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=[query]).data[0].embedding
    response = qdrant.query_points(collection_name=COLLECTION_NAME, query=query_vector, limit=n_results)

    matches = []
    for point in response.points:
        payload = point.payload or {}
        matches.append(
            {
                "summary": payload.get("text"),
                "doc_type": payload.get("doc_type"),
                "start_date": payload.get("start_date"),
                "end_date": payload.get("end_date"),
                "pct_change": payload.get("pct_change"),
                "direction": payload.get("direction"),
                "relevance_score": point.score,
            }
        )

    return {"status": "success", "query": query, "num_results": len(matches), "results": matches}


# Deliberately small: no __import__, no open/eval/exec, no os/sys/subprocess
# access. This is a best-effort sandbox appropriate for a tool you run
# locally under your own account (the same trust model as a Jupyter cell),
# not a hardened boundary for untrusted/multi-tenant input.
_SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
        "len", "list", "map", "filter", "max", "min", "print", "range",
        "round", "set", "sorted", "str", "sum", "tuple", "zip",
        "isinstance", "Exception", "ValueError", "TypeError", "KeyError",
        "IndexError", "StopIteration", "True", "False", "None",
    )
}
_FORBIDDEN_SUBSTRINGS = ("import ", "__", "open(", "exec(", "eval(", "os.", "sys.", "subprocess", "socket")
_EXEC_TIMEOUT_SECONDS = 10


class _CodeTimeoutError(Exception):
    pass


@contextlib.contextmanager
def _time_limit(seconds: int):
    has_alarm = hasattr(signal, "SIGALRM")  # not available on Windows

    def _handler(signum, frame):
        raise _CodeTimeoutError(f"Code execution exceeded {seconds}s timeout")

    if has_alarm:
        previous = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)
    try:
        yield
    finally:
        if has_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)


def _make_json_safe(value):
    """Recursively convert pandas/numpy objects into plain JSON-serializable types."""
    if isinstance(value, pd.DataFrame):
        safe_df = value.astype(object).where(pd.notnull(value), None)
        return safe_df.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return _make_json_safe(value.to_dict())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def run_data_analysis(code: str) -> dict:
    """Run Python/pandas code against the full candlestick dataset and return
    the exact computed result. Use this for ANY question that needs a
    precise or computed answer: a specific date's OHLCV, stats over a date
    range, rolling windows, streaks/consecutive-day patterns, aggregates,
    comparisons -- anything where the answer must be exact rather than a
    semantic match. It's the general-purpose alternative to writing a new
    hand-coded tool for every possible question shape.

    A pandas DataFrame called `df` is already loaded in scope, sorted by
    date ascending, with these columns:
      - open_time, close_time: pandas Timestamps (candle start/end)
      - date: string "YYYY-MM-DD" (handy for filtering/grouping)
      - symbol, interval: strings (e.g. "HYPE", "1d")
      - open, high, low, close: floats
      - volume: float
      - num_trades: int
      - pct_change: float, percent change from open to close for that candle

    `pd` (pandas) and `np` (numpy) are also available. Write code that
    computes the answer and assigns it to a variable named `result` --
    result can be a number, string, list, or dict (keep it to only the
    fields actually needed to answer the question, not a full dataframe
    dump). You can also `print(...)` intermediate values while developing
    the computation; anything printed comes back in `stdout` for you to
    reason about, e.g. if you need to run a quick check before deciding the
    final `result`.

    No imports, file/network access, or dunder attribute access are
    available in this sandbox -- only pandas/numpy operations on `df`.

    Args:
        code (str): Python code that sets a variable named `result`.

    Returns:
        dict with status "success"/"error". On success: `result` (the
        computed answer) and `stdout` (anything printed). On error:
        `error_message` describing what went wrong (e.g. a Python
        exception) so you can fix the code and try again.
    """
    try:
        df = _load_dataframe()
    except FileNotFoundError as e:
        return {"status": "error", "error_message": str(e)}

    lowered = code.lower()
    for token in _FORBIDDEN_SUBSTRINGS:
        if token in lowered:
            return {
                "status": "error",
                "error_message": f"Disallowed construct in code: {token.strip()!r}. "
                "Only pandas/numpy operations on the provided `df` are allowed.",
            }

    # Deliberately a single namespace used as both globals and locals, not
    # two separate dicts. exec(code, globals_dict, locals_dict) with two
    # *different* dicts makes Python treat the code as if it were a class
    # body -- lambdas, comprehensions, and df.apply(...) closures then can't
    # see locals-only names (a well-known exec() gotcha) and raise spurious
    # NameErrors on completely ordinary pandas code. A single namespace
    # behaves like real module-level code and avoids that.
    sandbox_ns = {"__builtins__": _SAFE_BUILTINS, "pd": pd, "np": np, "df": df.copy()}
    stdout = io.StringIO()

    try:
        with _time_limit(_EXEC_TIMEOUT_SECONDS), contextlib.redirect_stdout(stdout):
            exec(code, sandbox_ns)  # noqa: S102 -- sandboxed, see module docstring
    except _CodeTimeoutError as e:
        return {"status": "error", "error_message": str(e), "stdout": stdout.getvalue()}
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"{type(e).__name__}: {e}",
            "stdout": stdout.getvalue(),
        }

    if "result" not in sandbox_ns:
        return {
            "status": "error",
            "error_message": "Code ran successfully but did not set a variable named 'result'.",
            "stdout": stdout.getvalue(),
        }

    return {
        "status": "success",
        "result": _make_json_safe(sandbox_ns["result"]),
        "stdout": stdout.getvalue(),
    }


def get_data_coverage() -> dict:
    """Report which coin/interval/date-range of candlestick data is actually
    loaded, so you know what the knowledge base can and can't answer before
    committing to a specific date or range in other tool calls.

    Returns:
        dict with status, coin, interval, num_candles, start_date, end_date.
    """
    try:
        df = _load_dataframe()
    except FileNotFoundError as e:
        return {"status": "error", "error_message": str(e)}

    return {
        "status": "success",
        "coin": COIN,
        "interval": INTERVAL,
        "num_candles": int(len(df)),
        "start_date": str(df["date"].iloc[0]),
        "end_date": str(df["date"].iloc[-1]),
    }
