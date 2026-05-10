# Quant Research Agent — LangGraph + Llama 3.2

End-to-end automated research system for rates and FX strategies.  
**LLM proposes → Python validates → LangGraph iterates.**

---

## Architecture

```
Data Layer  →  Feature Engineering  →  Hypothesis Agent (Llama)
                                               ↓
                             Signal Construction + Backtest (pandas)
                                               ↓
                             Evaluation Agent  →  Sharpe / Drawdown
                                               ↓
                                        Critic Agent (Llama)
                                               ↓
                               Refinement Loop (LangGraph, max 5 iters)
```

### Agents

| Agent | File | Role |
|---|---|---|
| **Hypothesis Agent** | `agents/hypothesis_agent.py` | Calls Llama → generates a structured JSON trading hypothesis (signal name, instruments, trade rule, rationale) |
| **Evaluation Agent** | `agents/evaluation_agent.py` | Pure Python — computes Sharpe, max drawdown, annualised return, avg turnover from backtest results |
| **Critic Agent** | `agents/critic_agent.py` | Calls Llama → reads hypothesis + evaluation → returns `accept / reject / refine` with a reason and suggestion |

### LangGraph Nodes & Edges

```
[hypothesis] → [backtest] → [evaluate] → [critic]
                                              │
                   ┌──────── refine ──────────┘
                   │         (up to 5 iterations)
                   │
              [hypothesis]
                   │
              accept / max_iters → END
```

State object: `graph/state.py → ResearchState`

Fields carried through the graph:
- `hypothesis` — LLM-generated JSON
- `data` — loaded DataFrames (lazy, per run)
- `backtest_results` — pandas DataFrame from engine
- `evaluation` — `{sharpe, max_drawdown, annualized_return, avg_turnover}`
- `critic` — `{decision, reason, suggestion}`
- `decision` — `"accept" | "reject" | "refine"`
- `iteration` — loop counter

### Strategy Families Supported

| Theme | Signal | Instruments |
|---|---|---|
| Rates curve | 2s10s steepening z-score | FRED DGS2, DGS10 |
| Rates curve | 5s30s flattening momentum | FRED DGS5, DGS30 |
| FX carry | Rate differential | EURUSD=X, GBPUSD=X, USDJPY=X, AUDUSD=X |
| Positioning | COT net speculative positioning | FX futures (placeholder v0.2) |
| Cross-asset | Carry + trend composite | G10 FX |

---

## Environment Setup

### 1. Python virtual environment

```bash
# From the repo root
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Environment variables

```bash
cp .env.example .env
# Edit .env and set FRED_API_KEY
# Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html
```

`.env` is git-ignored. The app loads it automatically via `python-dotenv`.

### 3. Ollama — local Llama 3.2

Ollama serves Llama locally over HTTP at `http://localhost:11434`.

```bash
# macOS (Homebrew)
brew install ollama

# Or download the installer from https://ollama.com/download
```

**Pull and run Llama 3.2:**

```bash
# Terminal 1 — start the server (leave this running)
ollama serve

# Terminal 2 — pull the model (one-time, ~2 GB)
ollama pull llama3.2

# Optional smoke test in the terminal
ollama run llama3.2 "Return the word OK and nothing else."
```

Ollama exposes a REST API at `http://localhost:11434`.  
LangChain's `ChatOllama` talks to this URL automatically — no API key needed.

**Changing the model:**  
Edit `config/settings.yaml → llm.model` to switch to any model you have pulled
(e.g. `llama3.1`, `mistral`, `qwen2.5`). The code reads this at runtime.

```bash
# List models you have locally
ollama list
```

---

## Running the Pipeline

```bash
# Activate venv first
source venv/bin/activate

# Full research loop (hypothesis → backtest → critic, up to 5 iterations)
python main.py
```

The loop prints each state transition and the final accepted (or best) strategy.

---

## Running Tests

```bash
pytest tests/ -v

# LLM smoke test (requires Ollama running)
pytest tests/test_llm.py -v

# Pure Python tests (no LLM needed)
pytest tests/test_backtest_metrics.py tests/test_data_loaders.py -v
```

---

## Project Structure

```
quant-agents/
│
├── agents/
│   ├── hypothesis_agent.py     # LLM → structured JSON hypothesis
│   ├── critic_agent.py         # LLM → accept / reject / refine
│   └── evaluation_agent.py     # Compute Sharpe, drawdown, turnover
│
├── data/
│   ├── loaders/
│   │   ├── yahoo.py            # yfinance wrapper
│   │   ├── fred.py             # FRED API wrapper (rates)
│   │   ├── ecb.py              # ECB SDW (placeholder v0.2)
│   │   └── cot.py              # CFTC COT (placeholder v0.2)
│   └── processing/
│       ├── yield_curve.py      # 2s10s, 5s30s spreads + momentum
│       ├── fx_carry.py         # Rate differential, FX returns
│       ├── positioning.py      # COT positioning features
│       └── common.py           # Rolling z-score utility
│
├── backtests/
│   ├── engine.py               # Vectorised pandas backtest
│   ├── metrics.py              # Sharpe, max drawdown, annualised return
│   └── validation.py           # Walk-forward validation, sub-period checks
│
├── graph/
│   ├── state.py                # ResearchState TypedDict
│   └── workflow.py             # LangGraph nodes, edges, conditional routing
│
├── models/
│   └── llm.py                  # ChatOllama wrapper + config loader
│
├── config/
│   └── settings.yaml           # LLM model, backtest params, thresholds
│
├── notebooks/
│   ├── 01_data_sanity_check.ipynb
│   └── 02_first_backtest.ipynb
│
├── tests/
│   ├── test_llm.py
│   ├── test_data_loaders.py
│   ├── test_backtest_metrics.py
│   └── test_graph.py
│
├── main.py                     # CLI entry point
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Validation Requirements

All backtests must satisfy:

- **No lookahead bias** — positions use `signal.shift(1)` before computing returns
- **Transaction costs** — estimated at `transaction_cost_bps / 10_000` per unit of turnover
- **Minimum history** — 252 trading days required before accepting a result
- **Sub-period analysis** — evaluation splits history at the midpoint and checks both halves
- **Turnover analysis** — high turnover strategies penalised in critic decision

---

## Acceptance Criteria (v0.1.0)

- [ ] Ollama/Llama 3.2 responds through LangChain
- [ ] LangGraph workflow executes end-to-end
- [ ] Yahoo + FRED data loads successfully
- [ ] At least one yield-curve feature generated
- [ ] At least one FX return series backtested
- [ ] Sharpe and drawdown calculated
- [ ] Critic agent returns accept / reject / refine

---

## Risks & Known Limitations

| Risk | Mitigation |
|---|---|
| Llama produces weak / hallucinated JSON | Retry parser with regex extraction + Pydantic validation |
| Lookahead bias | `signal.shift(1)` enforced in `engine.py`; validation test in `tests/` |
| Overfitting | Walk-forward validation in `backtests/validation.py` |
| COT data lag | Weekly input only; treated as slow-moving filter, not primary signal |
| Llama 3.2 speed | ~5–15 sec per call on M-series Mac; acceptable for research loop |

---

## v0.2.0 Roadmap

- Walk-forward parameter sweeps
- MLflow or SQLite experiment tracking
- Strategy memory (failed hypotheses not repeated)
- ECB SDW + CFTC COT full implementation
- Optional Claude / GPT upgrade for higher-quality hypothesis generation
- Portfolio-level capital allocation and risk scaling