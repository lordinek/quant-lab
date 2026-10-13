# quant-lab

These are the scripts I run on a trading strategy before I believe its backtest. Most of my ideas die here.

The strategy in `strategy/setup.py` is one of those. It's an ICT-style setup: a liquidity sweep, then a change of character, then entry on an inverted fair value gap. The first backtest looked very good. It turned out the stop was sometimes placed on the wrong side of the entry, and that bug was producing the good results. After that was fixed, it lost money on every market I tried. I kept it in the repo because it's a real example of what these checks are for. The strategies I actually run are not here.

Comments and docstrings are in Turkish, and most of them explain why something is done the way it is.

## The checks

| Script | What it answers |
| --- | --- |
| `lookahead_check.py` | Does any signal use a bar it couldn't have seen yet? Reruns on cut-off data and compares. |
| `recursive_check.py` | Do indicator values change when more history is added in front? |
| `random_control.py` | Does the strategy beat random entries with the same stop and target? |
| `robustness.py` | Does it still work when the parameters move a bit, or only at one exact setting? |
| `out_of_sample.py` | Fit on 2021–2023, then look at 2024–2026. |
| `windows.py` | Rolling windows: how many are profitable, and how bad is the worst? |
| `layer_ablation.py` | Remove one layer of the signal at a time. Which ones actually matter? |
| `hour_filter_test.py` | Is a time-of-day filter real or curve-fitted? |
| `multi_market.py` | Does it work on other instruments too? |
| `cost_sensitivity.py` | How much spread and slippage before the edge is gone? |

## The rest

`signals/` finds swing points, breaks of structure, fair value gaps, liquidity sweeps, trading sessions and news windows. Each event records the bar where it happened and the bar where it could first be known, because mixing those two up is the usual way look-ahead bias gets in.

`backtest/evaluate.py` walks trades bar by bar with costs, a max holding time and optional exits before news. Results are in R (multiples of risk). `portfolio.py` turns R into an account balance.

`backtest/prop_eval.py` simulates prop-firm challenges with FTMO, Topstep and Apex style rules: daily loss limit, max or trailing drawdown, profit target and the funded stage after it. When something is unclear it assumes the rule that is worse for me.

`strategy/protections.py` holds the live stop rules: pause after N stops in a row, after too many stops in 24 hours, or after a drawdown limit. `protection_scan.py` picks the thresholds on training data and checks them on test data.

`journal/` is a small CLI for logging manual trades. Each entry also saves what the bot thought at that moment.

`data/` downloads free 1-minute data from Dukascopy, HistData and Binance.

## Running

```bash
pip install -r requirements.txt
pytest                          # 100 tests
python -m backtest.run          # downloads NAS100 1m data and runs the example strategy
python -m backtest.random_control
```

## Two mistakes that shaped this

- Time windows used to be counted in bars. 120 bars is 10 hours on 5-minute data and 30 hours on 15-minute data, so changing the timeframe quietly changed the strategy. They're in hours now.
- My dashboard once showed great multi-market numbers that were really coming from this dead strategy, through a stale data source. Now I check where a number comes from before I trust it.

MIT license. Not financial advice.
