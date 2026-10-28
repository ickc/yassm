"""Datasets used by the examples.

See ``data/README.md`` for their sources.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"

#: Set ``YASSM_EXAMPLES_FAST=1`` to shrink sample sizes and grids so that every
#: example runs in seconds (used by the integration tests). Results are then
#: *not* meaningful.
FAST: bool = bool(os.environ.get("YASSM_EXAMPLES_FAST"))


def size(full: int, fast: int) -> int:
    """Return ``fast`` if `FAST` else ``full``."""
    return fast if FAST else full


def smooth_signal_truth(u: np.ndarray) -> np.ndarray:
    """The smooth function underlying `load_smooth_signal` and `load_smooth_signal_with_ar1`, for ``u`` in [0, 1)."""
    return np.sin(15.0 * u) + np.exp(-0.5 * np.square(u)) + 0.5 * np.square(u - 0.5)


def _load_simulated(name: str) -> pd.DataFrame:
    """Load a simulated series indexed by ``t = 1, ..., n`` with a single column ``y``."""
    df = pd.read_csv(DATA / f"{name}.csv", index_col=0)
    df.index.name = "t"
    return df


def load_nile() -> pd.DataFrame:
    """Annual flow of the river Nile at Aswan, 1871–1970, in 10⁸ m³."""
    df = pd.read_csv(DATA / "nile.dat")
    df.columns = ["Nile"]
    df.index = pd.date_range("1871", "1970", freq="YS")
    return df


def load_nile_gaps() -> pd.DataFrame:
    """The Nile series with two stretches of observations removed (``NaN``)."""
    df = pd.read_csv(DATA / "nile_gaps.csv", usecols=[2])
    df.columns = ["Nile"]
    df.index = pd.date_range("1871", "1970", freq="YS")
    return df


def load_google_trends() -> pd.DataFrame:
    """Weekly worldwide Google Trends search interest in "google", 2004–2016."""
    df = pd.read_csv(DATA / "google_trends.csv", skiprows=range(4), nrows=629)
    # weeks are given as "start - end": index by their start date
    df.index = pd.to_datetime(df.Week.str.split(" - ").map(lambda x: x[0]))
    df.index.name = "week"
    df.google = df.google.astype(np.float64)
    del df["Week"]
    return df


def load_sunspots() -> pd.DataFrame:
    """Yearly mean total sunspot number, 1700–2019."""
    df = pd.read_csv(
        DATA / "sunspots.txt",
        sep=r"\s+",
        names=["decimal year", "SNvalue", "SNerror", "Nb observations"],
    )
    df.index = pd.date_range("1700", "2019", freq="YS")
    return df


def load_ar1_plus_noise() -> pd.DataFrame:
    """Simulated AR(1) state observed with noise (201 points)."""
    return _load_simulated("ar1_plus_noise")


def load_smooth_signal() -> pd.DataFrame:
    """Simulated `smooth_signal_truth` plus noise (500 points)."""
    return _load_simulated("smooth_signal")


def load_smooth_signal_with_ar1() -> pd.DataFrame:
    """Simulated `smooth_signal_truth` plus an AR(1) component and noise (500 points)."""
    return _load_simulated("smooth_signal_with_ar1")


def load_level_shifts() -> pd.DataFrame:
    """Simulated piecewise-constant level with occasional jumps, observed with noise (2048 points)."""
    return _load_simulated("level_shifts")


def load_ma1() -> pd.DataFrame:
    """Simulated MA(1) process (401 points)."""
    return _load_simulated("ma1")


def load_trend_breaks() -> pd.DataFrame:
    """Simulated piecewise-linear trend with occasional breaks, observed with noise (100 points)."""
    return _load_simulated("trend_breaks")


def load_volatility_regimes() -> pd.DataFrame:
    """Simulated zero-mean series whose log-variance is piecewise constant (400 points)."""
    return _load_simulated("volatility_regimes")
