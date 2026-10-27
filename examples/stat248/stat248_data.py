"""Datasets used by the STAT 248 homework examples.

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


def load_nile() -> pd.DataFrame:
    """Annual flow of the river Nile at Aswan, 1871–1970."""
    df = pd.read_csv(DATA / "Nile.dat")
    df.columns = ["Nile"]
    df.index = pd.date_range("1871", "1970", freq="YS")
    return df


def load_Prob2HW3DataStat248() -> pd.DataFrame:
    """Similar to Nile but with missing data."""
    df = pd.read_csv(DATA / "Prob2HW3DataStat248.csv", usecols=[2])
    df.columns = ["Nile"]
    df.index = pd.date_range("1871", "1970", freq="YS")
    return df


def load_googletrends21Jan2016() -> pd.DataFrame:
    """Weekly Google Trends search interest of "google", 2004–2016."""
    df = pd.read_csv(DATA / "googletrends21Jan2016.csv", skiprows=range(4), nrows=629)
    # parse Week to datetime
    df.index = pd.to_datetime(df.Week.str.split(" - ").map(lambda x: x[0]))
    # cast for inference
    df.google = df.google.astype(np.float64)
    del df["Week"]
    return df


def load_SN_y_tot_V2() -> pd.DataFrame:
    """Yearly mean total sunspot number, 1700–2019."""
    df = pd.read_csv(
        DATA / "SN_y_tot_V2.0.txt",
        sep=r"\s+",
        names=["decimal year", "SNvalue", "SNerror", "Nb observations"],
    )
    df.index = pd.date_range("1700", "2019", freq="YS")
    return df


def _load_indexed(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv", index_col=0)


def load_Prob3HW3DataStat248() -> pd.DataFrame:
    return _load_indexed("Prob3HW3DataStat248")


def load_Prob4HW3DataStat248() -> pd.DataFrame:
    return _load_indexed("Prob4HW3DataStat248")


def load_Prob5HW3DataStat248() -> pd.DataFrame:
    return _load_indexed("Prob5HW3DataStat248")


def load_Prob6HW3DataStat248() -> pd.DataFrame:
    return _load_indexed("Prob6HW3DataStat248")


def load_Prob3HW4DataStat248() -> pd.DataFrame:
    return _load_indexed("Prob3HW4DataStat248")


def load_Prob5HW4DataStat248() -> pd.DataFrame:
    return _load_indexed("Prob5HW4DataStat248")


def load_Prob6HW4DataStat248() -> pd.DataFrame:
    return _load_indexed("Prob6HW4DataStat248")


def load_Prob5HW5DataStat248() -> pd.DataFrame:
    return _load_indexed("Prob5HW5DataStat248")


def load_Prob6HW5DataStat248() -> pd.DataFrame:
    return _load_indexed("Prob6HW5DataStat248")


def load_Prob7HW5DataStat248() -> pd.DataFrame:
    return _load_indexed("Prob7HW5DataStat248")
