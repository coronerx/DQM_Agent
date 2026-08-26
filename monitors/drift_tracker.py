"""
drift_tracker.py — "is this a one-off blip, or has this channel been
trending this way for a while?"
"""

from collections import defaultdict, deque
import numpy as np
from scipy import stats

WINDOW_SIZE = 20
MIN_HISTORY_FOR_TREND = 5

_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))


def record_and_check(channel: str, value: float) -> dict:
    history = _history[channel]
    history.append(value)

    if len(history) < MIN_HISTORY_FOR_TREND:
        return {
            "channel": channel,
            "value": value,
            "window_size": len(history),
            "trend_detectable": False,
            "note": f"only {len(history)} chunks of history so far, need "
                    f"{MIN_HISTORY_FOR_TREND} before trend assessment is meaningful",
        }

    values = np.array(history)
    baseline = values[:-1]
    baseline_mean = baseline.mean()
    baseline_std = baseline.std() if baseline.std() > 0 else 1e-9

    z_latest = (value - baseline_mean) / baseline_std

    # Proper linear regression significance test, NOT a heuristic ratio to
    # the window's own raw standard deviation. The old approach was
    # self-defeating: a real, developing trend inflates the very std used
    # to judge whether the trend is "big enough," making the test hardest
    # to trigger exactly when there's real drift to catch -- confirmed on
    # both synthetic and real CMS data, where the boolean essentially never
    # fired despite an obvious, agent-identified +3.4 GeV drift over 12
    # chunks. There was also a separate gating bug: the old condition
    # additionally required len(history) >= WINDOW_SIZE // 2 == 10 chunks
    # before the boolean could ever be True at all, regardless of the
    # trend's strength. linregress fits the trend line and tests whether
    # the slope is significantly different from zero given the RESIDUAL
    # scatter around that fitted line (noise after removing the trend) --
    # a much better estimate of "normal variation" than the raw variance
    # of a window that already contains the drifting values.
    x = np.arange(len(values))
    regression = stats.linregress(x, values)
    slope = regression.slope
    slope_p_value = regression.pvalue

    return {
        "channel": channel,
        "value": value,
        "window_size": len(history),
        "trend_detectable": True,
        "z_score_vs_recent_baseline": float(z_latest),
        "linear_trend_slope": float(slope),
        "trend_p_value": float(slope_p_value),
        "looks_like_sustained_drift": bool(slope_p_value < 0.05),
    }


def reset(channel: str | None = None) -> None:
    if channel is None:
        _history.clear()
    else:
        _history.pop(channel, None)
