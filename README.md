# CGTRS — Crypto Genetic Trading Research System (paper/backtest only)

Local research system that evolves modular strategy chromosomes with a GA,
validates them through purged rolling walk-forward folds, runs a sequential
bull/bear discussion and Trader proposal per decision, applies an
**authoritative deterministic Risk Control gate** before every simulated order,
audits frozen candidates with a **Judge**, and exposes everything through a
**read-only spectator console**.

> Backtests use simulated fills. Forward testing routes orders **only to the
> Binance USDT-M Futures TESTNET** (mainnet hosts are refused in code). Not
> financial advice. Backtests do not establish future profitability.

Role decomposition (analysts → bull/bear → trader → risk → approval) follows
[TradingAgents](https://github.com/TauricResearch/TradingAgents); the
implementation is independent.

## Quick start (Windows PowerShell, Python 3.11+)

Run from the repository root (`trader`). Root and `backend` launches now share
`backend/data/cgtrs.sqlite`. Relative CLI paths are backend-relative.

```powershell
python -m venv backend/.venv
.\backend\.venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements.txt -r backend/requirements-dev.txt
python -m app.cli llm-test
python -m app.cli run                         # Binance public closed bars + funding
python -m app.cli experiments                 # actual IDs, states, ranking, qualified count
python -m app.cli explain --experiment latest # current local LLM explains existing records
python -m app.cli trade --experiment latest --dry-run --once
python -m app.cli serve                       # http://127.0.0.1:8000, built-in console
```

Do not type `0001`: IDs are printed at the START and end of a run and listed by
`experiments`. Use `latest` or an exact listed ID. `latest` chooses the newest
completed research experiment with a final artifact, not a live run. Aborted
runs are listed with their reason and cannot trade. `--db` must match for every
command when using custom storage. Synthetic demo: `run --config configs/default.json`;
synthetic runs are not eligible for live-market forward testing.

If no strategy qualifies, order execution remains blocked. For an explicitly
unqualified **no-order research preview** only:

```powershell
python -m app.cli trade --experiment latest --dry-run --once --allow-unqualified
```

This can preview the best non-risk-failed candidate when the final archive is
empty; it does not promote it to QUALIFIED. Dry-run never calls account,
position or order endpoints even when testnet keys are present. It uses configured
initial equity and no position, and is not a full paper-account simulator.
Dry-run and actual testnet state default to separate files. For actual testnet
orders a QUALIFIED strategy and testnet keys in the local shell are required:

```powershell
$env:BINANCE_TESTNET_API_KEY = "YOUR_TESTNET_KEY"
$env:BINANCE_TESTNET_API_SECRET = "YOUR_TESTNET_SECRET"
python -m app.cli trade --experiment latest
New-Item backend/KILL -ItemType File           # flatten + stop; remove manually before restarting
```

Mainnet order hosts are refused. Signal data follows the stored experiment
configuration unless `--config` explicitly overrides it. Public-market network
availability is still required. Every real entry is followed by exchange-side
stop/target protection, and failure flattens the position.

## LLM setup (local only)

Put ALL THREE values in `backend/.env` (or the PowerShell environment):

```dotenv
LLM_BASE_URL=https://YOUR_PROVIDER_API_BASE
LLM_API_KEY=YOUR_PRIVATE_KEY
LLM_MODEL=EXACT_MODEL_ID_FROM_YOUR_PROVIDER
```

The base must be an OpenAI-compatible API base, not the portal's website or
chat page. The client appends `/chat/completions`; a full completions URL is
also normalized. Gemini's compatible base is
`https://generativelanguage.googleapis.com/v1beta/openai`; OpenRouter's is
`https://openrouter.ai/api/v1`. For Cortensor use the API base and model ID
actually supplied by your portal; do not guess the portal website as the base.

`backend/.env` has priority over legacy `backend/app/.env`, but a non-empty
shell variable overrides both. In PowerShell clear stale overrides with
`Remove-Item Env:LLM_BASE_URL`, `Remove-Item Env:LLM_MODEL` and
`Remove-Item Env:LLM_API_KEY` (only when present), then restart commands/serve.
Entering only an API key is not sufficient. The spectator page deliberately
has no key-entry field and cannot test or modify secrets. No frontend source
was present upstream; the built-in console replaces the empty frontend checkout.

```powershell
python -m app.cli llm-test                    # real HTTP ping, nonzero on failure
python -m app.cli explain --experiment latest # refresh saved explanations after changing .env
```

`configured` means settings exist, NOT verified connectivity. HTTP 401/403:
check credentials/access; 404: check API base/model; 429/503: provider quota or
capacity. Retryable errors and timeouts get bounded retries, then surface as
errors, never fake LLM output. Do not paste keys into bug reports.

The shipped Binance configuration uses `auto`: any LLM setting attempts LLM
and reports incomplete configuration; no settings uses a labeled offline
template. Explicit `offline` and `narrate: none` remain respected by automated
runs; `explain` explicitly requests the configured LLM. A completed run explains
the first qualified holdout, or the best stored OOS candidate when none qualify,
including no-trade/deferred decisions. Forward-test decisions also receive saved
explanations. Narration never changes ranks, gates or orders. Historical results
are immutable until an explicit `explain` creates new narration artifacts.

## Data and spectator

`run` defaults to live Binance closed bars and funding; snapshot data is immutable,
not a continuously changing backtest. The Market tab streams current data via
WebSocket and refreshes public snapshots. The console is served by Python at `/`;
no Node install or missing frontend submodule is needed. API remains GET/WebSocket
only, SQLite read-only. Local operator commands alone run research or trading.

CSV fixture mode is supported via data config. Gaps/integrity failures abort,
never silently fill. Offline fixture mode generates synthetic bars and is not
real-time market data. Run tests with `python -m pytest backend/tests` from root
or `python -m pytest tests` from backend.

## Pipeline

```text
bars → quality checks → point-in-time features (AMT/SMC/SNR/trend, evidence IDs)
  → per fold: IS evolution (mandatory mutation, lineage) → freeze diverse elite
  → OOS simulation (fees, spread, size/vol slippage, funding, stop gaps)
     each decision: analysts → bull → bear → Trader proposal → Risk gate
  → Judge (trade-level windfall nullification, chromosome-level rejection)
  → sequential gates → retain/reject/defer
→ WFA consistency gate on later, unseen OOS windows → untouched holdout → ranking
```

State machine: `CREATED → DATA_VALIDATION → FEATURE_BUILD → (IS_EVOLUTION →
CANDIDATE_FREEZE → OOS_EVALUATION → JUDGE_REVIEW → FOLD_DONE)×n → WFA_GATE →
HOLDOUT → COMPLETED`, or `ABORTED_INTEGRITY`.

## Layout

```text
backend/app/
  config.py                 typed, versioned experiment config (all thresholds)
  data/                     ingest, point_in_time (LookAheadError), quality
  features/                 amt, smc, snr, trend, pipeline (evidence bundle)
  genetic/                  schema (typed genes), operators, population
  simulator/                costs, ledger (reconciled), engine
  risk/                     gate (authoritative), scorecard (+ gates, fitness)
  agents/                   schemas, discussion, trader, judge, reward, provider
  validation/               walk_forward (purge/embargo), robustness
  orchestration/            state_machine, runner
  persistence/db.py         SQLite, hash-chained append-only events
  api/read_only.py          GET + WebSocket only
backend/configs/default.json  EXAMPLE thresholds, not recommendations
backend/tests/              acceptance-criteria tests
backend/app/api/console.html  Built-in spectator console
```

See `FINDINGS.md` for limitations, risks and the self-review.
