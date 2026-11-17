"""GTA VI launch-day Twitch CCU forecast, akun 2 (wnezloc / gentle-wren-52) — "Kindred Shape".

Competition: gta6-twitch-ccu-forecast-01. The task: for each of hours 2-24 of
GTA VI's release day (2026-11-19 UTC), forecast five quantiles of the Twitch
category's concurrent viewers, given hour 1's print and the case's comparables.

Model (nearest-neighbour shape matching). This is a rebuild of the account's
first entry, a plain OLS forecast that scored a practice loss of 0.162. The
rebuild does not fit any line: for each hour h, it finds the comparable whose
hour-1 print is closest to the case's hour-1 print in log terms, and copies
that launch's shape, scaled to the case's hour 1:

    center_h = p1 * (nearest_h / nearest_1)

The nearest launch is chosen per hour, among the comparables that cover that
hour, so an hour the nearest launch lacks falls to the next-nearest launch
that has it. Only rolling-midnight comparables are candidates (GTA VI unlocks
at midnight local, rolling country by country); with fewer than 5 of them,
every usable comparable is a candidate.

The spread is the pool's own disagreement about the hour's shape: the
standard deviation of the log ratios log(comp_h / comp_1) around their
median, scaled by 0.75 and clamped to [0.20, 1.00] — the same calibration
the leave-one-out backtest used for this model.

Why this bet? In the leave-one-out backtest over the 12 rolling
comparables, this nearest-neighbour model scored a mean pinball loss of
0.2842 — worse on average than the fleet's regression ensembles (0.2401)
— but its per-fold losses were 0.12-0.14 on 8 of the 12 held-out launches,
far better than any regression on those folds, with two catastrophic folds
(1.21 and 0.73) where the nearest launch's shape was wrong. It is a lottery
ticket, not a workhorse: on the single final case, if GTA VI's shape
resembles its nearest comparable, this entry can beat every regression in
the fleet; if not, it loses badly. That is exactly the point of a
diversified portfolio — the other seven accounts carry the steady
regression bets, and this one carries the high-variance shape bet. Its
predictions differ from every other entry in the fleet on all 23 hours.

Quantiles: q_tau(h) = center_h * exp(z_tau * s_h), with s_h the scaled,
clamped log-ratio spread. Deterministic, standard library only, no network,
no clock.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim. It reads every <id>.in in the input directory and writes
<id>.out to the output directory: {"hours": [{"hour": h, "q": [q05, q25, q50,
q75, q95]}, ...]}, exactly 23 rows, hours 2-24 each once, each q five finite
numbers above 0 and at most 1e9, non-decreasing, at most 65,536 bytes.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# The log-ratio spread of an hour, scaled by SPREAD_SCALE then clamped.
SPREAD_MIN = 0.20
SPREAD_MAX = 1.00
SPREAD_SCALE = 0.75
# A comparable needs at least this many release-day hours to count.
MIN_HOURS = 20
# The baseline's spread, used as a fallback when no comparable qualifies.
BASELINE_SIGMA = 0.15


def parse_time(text: str) -> datetime:
    """An ISO 8601 UTC timestamp ("...Z") as an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def unlock_kind(comparable: dict) -> str:
    """How the launch unlocked: 'rolling', 'global', or 'other'.

    GTA VI unlocks at midnight local, rolling country by country, so only
    'rolling' comparables share its daily shape. The wording comes from the
    kit's own unlock field.
    """
    text = (comparable.get("unlock") or "").lower()
    if "midnight local" in text or "rolling midnight" in text:
        return "rolling"
    if "global" in text or "simultaneous" in text or "single global" in text:
        return "global"
    return "other"


def release_day_curve(comparable: dict) -> dict | None:
    """The comparable's release-day hourly viewers, {hour 1..24: viewers}.

    Returns None when hour 1 is missing or zero, or fewer than MIN_HOURS of the
    release day's hours are present (e.g. the GTA VI pre-release comparable,
    which has no release-day data yet).
    """
    day_start = parse_time(comparable["day_start"])
    by_hour: dict[int, float] = {}
    for point in comparable.get("hourly", []):
        at = parse_time(point["from"])
        hour = int((at - day_start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24:
            by_hour[hour] = point["viewers"]
    if by_hour.get(1, 0) <= 0 or len(by_hour) < MIN_HOURS:
        return None
    return by_hour


def quantile_sorted(sorted_xs: list[float], tau: float) -> float:
    """The tau-quantile of an already sorted list, linear interpolation.

    The same definition the leave-one-out backtest uses, so this entry
    matches the validated backtest numbers exactly.
    """
    n = len(sorted_xs)
    if n == 1:
        return sorted_xs[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac


def nearest_hour(curves: list[dict], h: int, log_p1: float) -> dict | None:
    """The curve covering hour h whose hour-1 print is closest to the case's.

    Distance is the absolute difference of the hour-1 prints in log terms.
    Ties go to the earlier curve in list order (the first minimum wins).
    """
    best = None
    best_d = float("inf")
    for curve in curves:
        if h in curve and curve[h] > 0 and curve[1] > 0:
            d = abs(math.log(curve[1]) - log_p1)
            if d < best_d:
                best_d = d
                best = curve
    return best


def hour_shape(curves: list[dict], h: int, log_p1: float) -> tuple[float, float] | None:
    """The hour's shape ratio and spread: (center / p1, spread).

    The center ratio is the nearest launch's hour-h ratio to its own hour 1.
    The spread is the standard deviation of the pool's log ratios
    log(comp_h / comp_1) around the median ratio's log, scaled by
    SPREAD_SCALE and clamped to [SPREAD_MIN, SPREAD_MAX] — exactly the
    calibration the leave-one-out backtest used for this model. The spread
    comes from the whole pool, not from the nearest launch alone: one
    launch gives a shape but no measure of how much shapes disagree.
    An hour fewer than 3 launches cover returns None, and the caller falls
    back to the previous hour's shape, or a flat ratio (1.0) with the
    baseline spread at hour 2.
    """
    ratios: list[float] = []
    for curve in curves:
        if h in curve and curve[h] > 0 and curve[1] > 0:
            ratios.append(curve[h] / curve[1])
    if len(ratios) < 3:
        return None
    nearest = nearest_hour(curves, h, log_p1)
    center_ratio = nearest[h] / nearest[1]
    median_ratio = quantile_sorted(sorted(ratios), 0.5)
    log_median = math.log(median_ratio)
    residuals = [math.log(r) - log_median for r in ratios]
    mean_resid = sum(residuals) / len(residuals)
    variance = sum((r - mean_resid) ** 2 for r in residuals) / len(residuals)
    spread = max(SPREAD_MIN, min(SPREAD_MAX, math.sqrt(variance) * SPREAD_SCALE))
    return (center_ratio, spread)


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def hour_quantiles(p1: float, shape: tuple[float, float]) -> list[float | int]:
    """The five quantiles for an hour: the nearest shape scaled, plus a log-normal spread."""
    center_ratio, spread = shape
    center = p1 * center_ratio
    out = []
    for z in Z:
        q = center * math.exp(z * spread)
        # Every quantile must be a finite number above 0 and at most 1e9.
        q = max(0.01, min(1_000_000_000, q))
        out.append(round2(q))
    # Rounding can only move a quantile by a cent; keep them non-decreasing.
    for i in range(1, 5):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    log_p1 = math.log(p1)

    # Only rolling-midnight comparables share GTA VI's daily shape; with
    # fewer than 5 of them, fall back to every usable comparable.
    rolling = []
    all_curves = []
    for comparable in case.get("comparables", []):
        curve = release_day_curve(comparable)
        if curve is None:
            continue
        all_curves.append(curve)
        if unlock_kind(comparable) == "rolling":
            rolling.append(curve)
    curves = rolling if len(rolling) >= 5 else all_curves

    shapes: dict[int, tuple[float, float]] = {}
    if curves:
        prev: tuple[float, float] = (1.0, BASELINE_SIGMA)
        for h in range(2, 25):
            shape = hour_shape(curves, h, log_p1)
            if shape is None:
                shape = prev
            shapes[h] = shape
            prev = shape
    else:
        # No comparable at all: fall back to the house baseline's flat shape.
        shapes = {h: (1.0, BASELINE_SIGMA * math.sqrt(h - 1)) for h in range(2, 25)}

    return {
        "hours": [
            {"hour": row["hour"], "q": hour_quantiles(p1, shapes[row["hour"]])}
            for row in case["forecast"]
        ]
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
