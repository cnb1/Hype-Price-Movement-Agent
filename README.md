# HYPE Candlestick RAG Agent

An agentic chatbot, built on Google's [Agent Development Kit (ADK)](https://google.github.io/adk-docs/),
that answers questions about price movements for **HYPE** (Hyperliquid's
native token) by combining:

- a **RAG knowledge store**: daily/weekly/monthly candlestick summaries,
  embedded and searchable by meaning (Qdrant + OpenAI embeddings), for fuzzy
  questions like *"when did it pump hard"* or *"was there a big crash in the
  spring"*, and
- a **code-execution tool**: the LLM writes pandas code that runs against the
  real candlestick DataFrame and gets back an exact computed result, for
  anything numeric — *"what was the closing price on March 4th"*, *"what was
  the % change in March"*, or arbitrarily complex questions like *"when was
  the last time we had 4 consecutive candles that closed down"*.

The chat model itself is an **OpenAI** model (e.g. `gpt-4o-mini`), wired
into ADK via [LiteLLM](https://docs.litellm.ai/docs/tutorials/google_adk) —
you get ADK's agent/tool orchestration without needing Gemini or a GCP
project.

## Why hybrid instead of pure vector RAG?

Candlestick data is numeric and exact by nature. A pure embedding search
over text chunks can tell you the *neighborhood* of a price move but will
happily hallucinate the actual number, and it can only ever answer question
shapes that were anticipated when the text was written — it has no way to
verify something like "4 consecutive down candles" because that fact was
never written into any document.

So this project treats semantic search as a **discovery** tool ("find the
period the user is vaguely describing") and hands anything that needs a
guaranteed-correct number to a **code-execution** tool instead: the model
writes the actual pandas computation for whatever the question is — a single
lookup, a range aggregate, a rolling window, a streak, a comparison — and a
real interpreter runs it. That generalizes to any analytical question
without needing a hand-written function for every question shape in advance.
The agent's system prompt tells it to work this way.

## Project layout

Code lives under `src/`; data, the vector store, and config/secrets live
under `resources/`. `README.md` and `.gitignore` stay at the project root.

```
hype_price_rag/
├── README.md
├── .gitignore
├── src/                               # all code
│   ├── scripts/
│   │   ├── fetch_hyperliquid_data.py    # step 2: pull real candles from Hyperliquid
│   │   └── build_knowledge_base.py      # step 3: CSV -> RAG knowledge store
│   ├── hype_agent/                      # the ADK agent package
│   │   ├── __init__.py
│   │   ├── agent.py                     # root_agent definition + system prompt
│   │   └── tools.py                     # semantic search tool + code-execution tool
│   └── chat.py                          # step 4: terminal chat loop
└── resources/                         # everything else
    ├── requirements.txt
    ├── .env.example                     # copy to resources/.env and fill in
    ├── data/
    │   ├── hype_1d_candles.SAMPLE.csv   # synthetic sample data (see note below)
    │   └── hype_1d_candles.csv          # real data lands here after step 2
    └── qdrant_store/                    # persisted vector DB (created by step 3, local on-disk mode)
```

Every script/module resolves `resources/` relative to its own file location
(walking up to the project root, then into `resources/`), so these commands
work regardless of your current directory — you don't need to `cd` into
`resources/` for paths to resolve. `.env` loading works the same way: each
entry point explicitly loads `resources/.env`, rather than relying on the
current directory.

## Setup

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r resources/requirements.txt
```

### 2. Configure your OpenAI key

```bash
cp resources/.env.example resources/.env
# then edit resources/.env and set OPENAI_API_KEY=sk-...
```

You only need an OpenAI key — Hyperliquid's market-data API is public and
requires no authentication.

### 3. Fetch real HYPE candlestick data from Hyperliquid

```bash
python src/scripts/fetch_hyperliquid_data.py --coin HYPE --interval 1d --days 365
```

This calls Hyperliquid's public `candleSnapshot` info endpoint
(`POST https://api.hyperliquid.xyz/info`) and writes
`resources/data/hype_1d_candles.csv`. Flags:

- `--interval`: `1m,3m,5m,15m,30m,1h,2h,4h,8h,12h,1d,3d,1w,1M`
- `--days`: lookback window (Hyperliquid caps responses at 5000 candles per
  coin/interval, so very long lookbacks at fine intervals get truncated to
  the most recent 5000 bars — daily candles for a year is well within that
  limit)

> **Note on the sample file:** `resources/data/hype_1d_candles.SAMPLE.csv`
> is synthetic placeholder data (randomly generated, clearly labeled)
> included only so the rest of the pipeline can be built/tested without
> live network access. Run this step to replace it with real Hyperliquid
> data before asking the agent real questions — otherwise you'll get
> accurate-sounding answers about fake prices.

### 4. Build the RAG knowledge base

```bash
python src/scripts/build_knowledge_base.py --csv resources/data/hype_1d_candles.csv
```

This reads the CSV, computes per-candle stats (% change, range, bullish/
bearish, notable-move flags), rolls them up into daily/weekly/monthly text
summaries, embeds them with `text-embedding-3-small` (configurable via
`OPENAI_EMBEDDING_MODEL` in `resources/.env`), and persists everything to a
local Qdrant collection in `resources/qdrant_store/`.

By default this runs Qdrant in **embedded, on-disk local mode** — no server
to install or run. If you'd rather point at a real Qdrant server or Qdrant
Cloud instance, set `QDRANT_URL` (and `QDRANT_API_KEY` if needed) in
`resources/.env` and both this script and the agent's tools will use it
instead.

Re-run this any time you refresh the CSV — it replaces the collection
rather than appending, so it never goes stale or duplicates.

### 5. Chat with the agent

Either use the included terminal script:

```bash
cd src && python chat.py
```

or ADK's own CLI/web UI, which needs to be run from `src/` (the parent
directory of the `hype_agent/` package):

```bash
cd src
adk run hype_agent      # terminal chat
adk web                 # browser UI, pick "hype_agent" from the dropdown
```

Example questions:

- "What's the date range of data you have?"
- "What was HYPE's closing price on 2025-03-04?"
- "How did HYPE perform in March 2025 — up or down, and by how much?"
- "When was the biggest single-day drop, and what happened around it?"
- "Was there a strong rally at any point in the data? When, and how big?"
- "When was the last time we had 4 consecutive candles that closed down?"
- "What's the average volume on days where price moved more than 3%?"

## How it works

1. **`src/scripts/fetch_hyperliquid_data.py`** pulls raw OHLCV candles from
   Hyperliquid and normalizes them into a tidy CSV (`open_time, close_time,
   symbol, interval, open, high, low, close, volume, num_trades`).
2. **`src/scripts/build_knowledge_base.py`** derives per-candle metrics (%
   change, intraday range, bullish/bearish/flat label, notable-move flag)
   and writes three tiers of plain-English summary documents — daily,
   weekly, monthly — each carrying structured metadata (dates, OHLCV, %
   change). These are embedded and stored in a local Qdrant vector database.
3. **`src/hype_agent/tools.py`** exposes three tools to the agent:
   - `get_data_coverage` — what date range/coin/interval is actually loaded
   - `run_data_analysis(code)` — runs LLM-written pandas code against the
     full candlestick DataFrame in a restricted sandbox and returns the
     exact computed result (`result` variable the code sets, plus any
     `stdout`). This is the general-purpose replacement for hand-written
     lookup functions — single-day OHLCV, range aggregates, rolling stats,
     streaks, comparisons, anything, all through one tool instead of one
     function per question shape.
   - `semantic_search_price_history(query)` — vector search over the
     daily/weekly/monthly summaries for fuzzy/exploratory questions
4. **`src/hype_agent/agent.py`** wires these tools into an ADK `LlmAgent`
   running an OpenAI model (via `LiteLlm`), with a system prompt that tells
   it when to use semantic search vs. code execution, and never to state a
   number that didn't come from a tool call.
5. **`src/chat.py`** (or `adk run` / `adk web`) drives the conversation loop,
   handling session state and tool-calling via ADK's `Runner`.

Every module in `src/` computes its own `PROJECT_ROOT` from `__file__` (by
walking up the correct number of parent directories to reach the repo root)
and derives `resources/...` paths and the `.env` location from that — so
nothing depends on the current working directory except for Python's own
module resolution (which is why `hype_agent` needs to be imported with
`src/` as the working directory or on `PYTHONPATH`, e.g. via `cd src`
first).

### About the code-execution sandbox

`run_data_analysis` runs model-generated Python via `exec()` with a reduced
builtins set (no `import`, `open`, `eval`, `exec`, dunder attribute access,
`os`/`sys`/`subprocess`/`socket`) and a 10-second wall-clock timeout. This is
a **best-effort** sandbox appropriate for a tool you run locally under your
own OpenAI account — the same trust model as a Jupyter cell you'd run
yourself — not a hardened boundary suitable for untrusted or multi-tenant
input. If you deploy this somewhere multiple people can trigger it, put it
behind a real isolation boundary (subprocess with resource limits, a
container, gVisor, etc.) rather than relying on the in-process checks here.

## Extending this

- **Other assets**: change `HL_COIN` in `resources/.env`, re-run steps 3–4.
  The tools auto-discover the CSV by coin/interval naming convention.
- **Intraday granularity**: set `HL_INTERVAL=1h` (or finer) for more
  granular analysis — just mind the 5000-candle cap per request.
- **Candlestick pattern detection**: `src/scripts/build_knowledge_base.py`'s
  `_daily_doc`/`_rollup_doc` functions are natural places to add real
  pattern recognition (doji, engulfing, hammer, etc.) instead of the
  current simple bullish/bearish/flat heuristic.
- **Swap the LLM**: change `OPENAI_CHAT_MODEL` in `resources/.env` to any
  LiteLLM-supported model string (including Gemini, Anthropic, etc.) if you
  want to compare providers — the rest of the pipeline is provider-agnostic.
