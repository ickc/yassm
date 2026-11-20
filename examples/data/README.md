# Datasets

| File | Description | Source |
| --- | --- | --- |
| `nile.dat` | Annual flow of the river Nile at Aswan, 1871–1970 | Durbin & Koopman, *Time Series Analysis by State Space Methods*; also R's `datasets::Nile` |
| `nile_gaps.csv` | The Nile series with 15 observations removed | Derived from `nile.dat` |
| `google_trends.csv` | Weekly worldwide Google Trends interest in "google", 2004–2016 | [Google Trends](https://trends.google.com/) |
| `sunspots.txt` | Yearly mean total sunspot number, 1700–2019 | WDC-SILSO, Royal Observatory of Belgium, Brussels, [CC BY-NC 4.0](https://www.sidc.be/SILSO/datafiles) |
| `ar1_plus_noise.csv` | AR(1) state observed with noise | Simulated[^course] |
| `smooth_signal.csv` | A smooth function plus noise (see `example_data.smooth_signal_truth`) | Simulated[^course] |
| `smooth_signal_with_ar1.csv` | The same smooth function plus an AR(1) component and noise | Simulated[^course] |
| `level_shifts.csv` | Piecewise-constant level with occasional jumps, observed with noise | Simulated[^course] |
| `ma1.csv` | MA(1) process | Simulated[^course] |
| `trend_breaks.csv` | Piecewise-linear trend observed with noise | Simulated[^course] |
| `volatility_regimes.csv` | Zero-mean series with a piecewise-constant log-variance | Simulated[^course] |

[^course]: Simulated series provided with the coursework of STAT 248 *Analysis of Time Series*, UC Berkeley, spring 2022.
