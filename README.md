# quant-lab

A research and validation framework for intraday trading models. It covers data ingestion, structure-based signal detection, a backtest engine that respects confirmation lag, a **multi-stage validation suite**, a prop-firm challenge simulator, live-trading protections and a trade journal.

The included strategy (`strategy/setup.py`, an ICT-style *liquidity sweep → CHoCH → inverse FVG* model) is kept on purpose as a **case study: the validation suite killed it.** Its early backtest looked profitable because of a stop placed on the wrong side of entry. After that bug was fixed, it failed on every instrument it was tested on. My production models are private and are not in this repo.

> Code comments and docstrings are in Turkish. They explain *why* each decision was made, not just what the code does.

## Why this exists

Most retail backtests lie by accident: look-ahead bias, a single lucky parameter, costs set to zero, or one market that happened to trend. This repo turns each of those failure modes into a script that has to pass before a model sees real money.

## Validation suite (`backtest/`)

| Check | Question it answers |
| --- | --- |
| `lookahead_check.py` | Does any signal use a bar it could not have known yet? It recomputes on truncated data and diffs the results. |
| `recursive_check.py` | Do indicators change when more history is prepended? |
| `random_control.py` | Does the model beat random entries with the same stop/target geometry? |
| `robustness.py` | Does it work across a plateau of parameters, or only on one lucky setting? |
| `out_of_sample.py` | Train 2021–23, test 2024–26. Does it hold on unseen data? |
| `windows.py` | Rolling-window stability: what share of windows is profitable, and how bad is the worst one? |
| `layer_ablation.py` | Which layer of a multi-layer signal chain actually adds value? |
| `hour_filter_test.py` | Is a time-of-day filter real, or curve-fitted? |
| `multi_market.py` | Is the edge instrument-specific or general? |
| `cost_sensitivity.py` | At what spread/slippage does the edge disappear? |

## Other components

- **`signals/`**: swing points, BOS/CHoCH market structure, fair value gaps and their inversion lifecycle, liquidity pools and sweeps, session/killzone and macro-news windows. Every event tracks *when it happened* separately from *when it became known*.
- **`backtest/evaluate.py`**: bar-by-bar trade resolution with costs, max holding time and news-window exits. Results are reported in R multiples.
- **`backtest/portfolio.py`**: converts R-results into account equity with position sizing and concurrent-position limits.
- **`backtest/prop_eval.py`**: prop-firm challenge simulator (FTMO / Topstep / Apex-style rules, including daily loss, max/trailing drawdown, profit target and funded-stage scaling). It uses conservative assumptions that always favor the firm.
- **`strategy/protections.py`**: live circuit breakers, such as a pause after N consecutive stops or a daily loss cap, with thresholds tuned on train and verified on test (`protection_scan.py`).
- **`journal/`**: CLI journal for manual trades. Each entry is stored next to the bot's view at that exact moment, so discretionary mistakes become measurable.
- **`data/`**: free 1-minute data sources (Dukascopy ticks, HistData, Binance klines) with resampling and caching.

## Quick start

```bash
pip install -r requirements.txt
pytest                              # 100 tests
python -m backtest.run              # downloads NAS100 1m data, runs the case-study model
python -m backtest.lookahead_check
python -m backtest.random_control
```

## Lessons baked into the code

- A stop on the wrong side of entry turned a losing model into a "winning" one. Setups whose stop lands on the wrong side are now filtered out in `find_setups`, and a gap-based stop mode makes the bug structurally impossible.
- Time windows are defined in **hours**, not bars. A 120-bar window means 10 h on 5m data but 30 h on 15m data, and that silently changed the strategy whenever the timeframe changed.
- When a dashboard showed great multi-market numbers, the cause was a stale data source still pointing at the retired model. Always verify where a number comes from.

## License

MIT. Educational/research use; nothing here is financial advice.
