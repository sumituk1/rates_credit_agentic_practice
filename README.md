# Quant Research Agent — LangGraph + Llama 3.2

End-to-end automated research system for rates and FX strategies.  
**LLM reasons → Python validates → LangGraph iterates.**

---

## How the Reasoning Works

This is not prompt-filling. Each LLM call uses **explicit chain-of-thought (CoT)** — the model
is required to reason step-by-step through a structured scratchpad before it is allowed to
output a decision. The reasoning trace is captured and printed at runtime so you can audit it.

### Hypothesis Agent — 5-Step CoT with Live Macro Context

Before proposing a signal, the LLM must answer five questions in sequence:

```
STEP 1 — REGIME ANALYSIS
  What does the current yield curve shape imply?
  Steep / flat / inverted → which macro regime?

STEP 2 — SIGNAL FAMILY SELECTION
  Given the regime, which family has edge?
  Why this family over the others right now?

STEP 3 — INSTRUMENT & PARAMETER SELECTION
  Which instruments? What lookback window?
  Why these over alternatives?

STEP 4 — ECONOMIC MECHANISM
  Carry / mean-reversion / momentum / flow logic?
  Why should this signal predict forward returns?

STEP 5 — FAILURE MODES
  Under what conditions does the signal break?
  How does this attempt address weaknesses from prior iterations?

→ Only after step 5: emit structured JSON hypothesis
```

**Live macro context is injected at prompt time** — the agent fetches the latest FRED yield
curve (2y, 5y, 10y, 30y) and feeds it directly into the prompt. The LLM reasons about
real numbers, not a hypothetical market.

**Iteration memory** — a summary of every previous signal, its Sharpe, drawdown, and the
critic's rejection reason is included in the prompt. The agent is explicitly required to
propose something different and explain why.

### Critic Agent — 4-Step CoT with Full Iteration History

The critic receives the full backtest metrics and the complete history of all prior iterations.
It must reason through four steps before deciding:

```
STEP 1 — STATISTICAL VALIDITY
  Check each threshold explicitly:
  Sharpe >= 0.5? MaxDD >= -0.25? Both sub-period Sharpes > 0?

STEP 2 — ROBUSTNESS
  Is edge consistent across both halves of the sample?
  Genuinely better than previous iterations, or minor variation?

STEP 3 — BIAS & RISK FLAGS
  Lookahead risk? Holding period vs. signal frequency?
  Turnover realistic under transaction costs?

STEP 4 — DECISION + SPECIFIC SUGGESTION
  State decision with metric citations.
  If refining: ONE concrete, surgical change
  (e.g. "increase z-score lookback from 60→120 days given flat curve regime")

→ Only after step 4: emit JSON {decision, reason, suggestion}
```

**Rule-based fallback** — if the LLM fails to produce valid JSON after 3 attempts, the critic
falls back to hard threshold rules so the loop always continues cleanly.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        LangGraph Loop                           │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  [hypothesis]                                            │   │
│  │  • Fetch live FRED yield curve                           │   │
│  │  • Inject history of failed attempts                     │   │
│  │  • 5-step CoT → JSON hypothesis                          │   │
│  └──────────────────┬───────────────────────────────────────┘   │
│                     │                                           │
│  ┌──────────────────▼───────────────────────────────────────┐   │
│  │  [backtest]  (pure Python / pandas)                      │   │
│  │  • Dispatch signal_name → FRED yields or Yahoo FX        │   │
│  │  • Compute z-score signal                                │   │
│  │  • Long-short backtest with transaction costs            │   │
│  └──────────────────┬───────────────────────────────────────┘   │
│                     │                                           │
│  ┌──────────────────▼───────────────────────────────────────┐   │
│  │  [evaluate]  (pure Python)                               │   │
│  │  • Sharpe, max drawdown, ann. return, turnover           │   │
│  │  • Sub-period split check (first half / second half)     │   │
│  └──────────────────┬───────────────────────────────────────┘   │
│                     │                                           │
│  ┌──────────────────▼───────────────────────────────────────┐   │
│  │  [critic]                                                │   │
│  │  • Receives full history of all iterations               │   │
│  │  • 4-step CoT → JSON {decision, reason, suggestion}      │   │
│  │  • Rule-based fallback on parse failure                  │   │
│  └──────────────────┬───────────────────────────────────────┘   │
│                     │                                           │
│  ┌──────────────────▼───────────────────────────────────────┐   │
│  │  [record]  (pure Python)                                 │   │
│  │  • Append iteration to history list                      │   │
│  │  • Store CoT reasoning traces                            │   │
│  └──────────────────┬───────────────────────────────────────┘   │
│                     │                                           │
│            accept / max_iters ──→ END                           │
│            reject / refine ─────→ [hypothesis]  (next iter)    │
└─────────────────────────────────────────────────────────────────┘
```

### State object (`graph/state.py`)

| Field | Type | Purpose |
|---|---|---|
| `hypothesis` | dict | Current JSON hypothesis from Llama |
| `backtest_results` | DataFrame | Raw backtest output |
| `evaluation` | dict | Sharpe, drawdown, sub-period metrics |
| `critic` | dict | `{decision, reason, suggestion}` |
| `decision` | str | `accept / reject / refine` |
| `iteration` | int | Loop counter (max 5) |
| `history` | list | All prior `(hypothesis, evaluation, critic)` triples |
| `reasoning_trace` | str | Hypothesis agent CoT scratchpad (human-readable) |
| `critic_reasoning` | str | Critic agent CoT scratchpad (human-readable) |
| `error` | str | Last backtest error, if any |

### Agents

| Agent | File | LLM? | What it does |
|---|---|---|---|
| **Hypothesis** | `agents/hypothesis_agent.py` | Yes — 5-step CoT | Reasons through macro regime → proposes signal |
| **Evaluation** | `agents/evaluation_agent.py` | No — pure Python | Sharpe, drawdown, sub-period split |
| **Critic** | `agents/critic_agent.py` | Yes — 4-step CoT | Reasons through metrics + history → decision |

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
# Free key: https://fred.stlouisfed.org/docs/api/api_key.html
```

### 3. Ollama — local Llama 3.2

Ollama serves Llama locally over HTTP at `http://localhost:11434`.

```bash
# macOS (Homebrew)
brew install ollama
```

**Start the server and pull the model:**

```bash
# Terminal 1 — leave this running
ollama serve

# Terminal 2 — one-time download (~2 GB)
ollama pull llama3.2

# Quick sanity check
ollama run llama3.2 "Return the word OK and nothing else."
```

**Connecting the code to Ollama:**  
`models/llm.py` uses `langchain_ollama.ChatOllama` which talks to `http://localhost:11434`
automatically. No API key needed. The model name is read from `config/settings.yaml`.

**Switching models:**

```yaml
# config/settings.yaml
llm:
  model: "llama3.2"        # change to any model you have pulled
  temperature: 0.2
```

```bash
# List models available locally
ollama list

# Pull a reasoning-optimised model for stronger CoT
ollama pull deepseek-r1     # DeepSeek R1 — explicit <think> traces
ollama pull qwen2.5         # Qwen 2.5 — strong instruction following
```

> **Tip:** Swap `model: "deepseek-r1"` in `settings.yaml` for significantly stronger
> chain-of-thought. DeepSeek R1 produces explicit `<think>...</think>` reasoning blocks
> before answering — ideal for the hypothesis and critic agents.

---

## Running the Pipeline

```bash
source venv/bin/activate

# Full research loop — runs up to 5 iterations
python main.py
```

**Sample output structure:**

```
══════════════════════════════════════
  ITERATION 1 — us_2s10s_zscore  (REFINE)
══════════════════════════════════════

── HYPOTHESIS REASONING (CoT) ───────
STEP 1 — REGIME ANALYSIS:
The current 2s10s spread of +0.42 bps indicates a marginally steep curve,
consistent with an early-to-mid easing cycle...

STEP 2 — SIGNAL FAMILY SELECTION:
Given the steepening bias, a long-steepener z-score signal should have
mean-reversion edge as curve normalises from inversion...
...

── EVALUATION ───────────────────────
  Sharpe            : 0.412
  Max drawdown      : -0.183
  First-half Sharpe : 0.631
  Second-half Sharpe: 0.188   ← weak second half flagged by critic

── CRITIC REASONING (CoT) ───────────
STEP 1 — STATISTICAL VALIDITY:
Sharpe 0.41 < 0.50 threshold. MaxDD -0.18 passes. Second-half Sharpe
0.19 > 0 but very weak. Does not meet acceptance criteria...

STEP 4 — DECISION:
Refine. The signal degrades in the second half, likely because the 60-day
lookback is too short to span a full rate cycle...

── CRITIC DECISION ──────────────────
  Decision  : REFINE
  Reason    : Sharpe 0.41 below threshold; second-half edge near zero
  Suggestion: Increase z-score lookback from 60 to 120 days
```

---

## Running Tests

```bash
# All tests (pure Python — no LLM, no network)
pytest tests/test_backtest_metrics.py tests/test_data_loaders.py tests/test_graph.py -v

# LLM smoke test (requires Ollama running)
pytest tests/test_llm.py -v -m integration
```

---

## Project Structure

```
quant-agents/
│
├── agents/
│   ├── hypothesis_agent.py   # 5-step CoT + live FRED context + iteration memory
│   ├── critic_agent.py       # 4-step CoT + full history + rule-based fallback
│   └── evaluation_agent.py   # Pure Python — Sharpe, drawdown, sub-period split
│
├── data/
│   ├── loaders/
│   │   ├── fred.py           # FRED API (US yields, arbitrary series)
│   │   ├── yahoo.py          # yfinance FX tickers
│   │   ├── ecb.py            # ECB SDW placeholder (v0.2)
│   │   └── cot.py            # CFTC COT placeholder (v0.2)
│   └── processing/
│       ├── yield_curve.py    # 2s10s, 5s30s spreads, z-scores, approx bond returns
│       ├── fx_carry.py       # Rate differential, FX carry z-score
│       ├── positioning.py    # COT net positioning z-score
│       └── common.py         # Rolling z-score utility
│
├── backtests/
│   ├── engine.py             # Vectorised long-only and long-short pandas backtest
│   ├── metrics.py            # Sharpe, max drawdown, ann. return, Calmar
│   └── validation.py         # Walk-forward, sub-period check, min history gate
│
├── graph/
│   ├── state.py              # ResearchState TypedDict (incl. history + reasoning traces)
│   └── workflow.py           # LangGraph nodes, edges, history recording, routing
│
├── models/
│   └── llm.py                # ChatOllama wrapper — reads model from settings.yaml
│
├── config/
│   └── settings.yaml         # LLM model, backtest params, thresholds
│
├── tests/
│   ├── test_llm.py           # Ollama smoke test (integration, needs Ollama running)
│   ├── test_data_loaders.py  # Feature engineering unit tests (no network)
│   ├── test_backtest_metrics.py  # Backtest engine + metrics unit tests
│   └── test_graph.py         # Graph compilation tests
│
├── main.py                   # CLI — runs the loop, prints full CoT traces
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Validation Requirements

- **No lookahead bias** — positions use `signal.shift(1)` before computing returns
- **Transaction costs** — estimated at `transaction_cost_bps / 10_000` per unit of turnover
- **Minimum history** — 252 trading days required before accepting a result
- **Sub-period analysis** — evaluation splits history at midpoint; both halves checked
- **Iteration memory** — critic explicitly compares against all prior failed attempts

---

## Upgrading to Stronger Reasoning

| Model | How to use | What improves |
|---|---|---|
| `deepseek-r1` | `ollama pull deepseek-r1` | Native `<think>` traces; stronger logical reasoning in critic |
| `qwen2.5` | `ollama pull qwen2.5` | Better instruction following; more reliable JSON output |
| Claude / GPT-4o | Set `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`; swap `llm.py` | Production-quality hypothesis generation |

---

## v0.2.0 Roadmap

- Full ReAct pattern — give hypothesis agent tool access (load data, plot, compute stats mid-reasoning)
- Walk-forward parameter sweeps
- SQLite experiment tracking — persist all iterations across sessions
- Strategy memory — failed hypotheses stored in a vector DB so they survive process restarts
- ECB SDW + CFTC COT full implementation
- Portfolio-level capital allocation and Kelly sizing
