# Kindred Shape — GTA VI launch-day Twitch CCU forecast (akun 2)

A nearest-neighbour shape forecast for `gta6-twitch-ccu-forecast-01`.

## Idea

No regression. For each hour 2–24, the entry finds the comparable launch
whose hour-1 print is closest to the case's hour-1 print (in log terms) and
copies that launch's shape, scaled to the case's hour 1:

```
center_h = p1 * (nearest_h / nearest_1)
```

The nearest launch is picked per hour, among rolling-midnight comparables
(GTA VI unlocks at midnight local). The spread is the pool's own
disagreement about the hour's shape — the standard deviation of the
log ratios `log(comp_h / comp_1)` around their median, scaled by 0.75
and clamped to [0.20, 1.00].

## Why

Leave-one-out backtest over the 12 rolling comparables: mean pinball loss
0.2842 — worse on average than the fleet's regression ensembles (0.2401),
but 0.12–0.14 on 8 of 12 held-out launches, far better than any regression
on those folds, with two catastrophic folds (1.21, 0.73). A lottery ticket:
if GTA VI's shape resembles its nearest comparable, this entry can beat
every regression in the fleet. The other seven accounts carry the steady
regression bets; this one carries the high-variance shape bet.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Deterministic, standard library only, no network, no clock. Reads every
`<id>.in` in the input directory, writes `<id>.out` (JSON
`{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]}`, 23 rows).

## License

MIT — see LICENSE.
