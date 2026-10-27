# STAT 248 homework

yassm was written for STAT 248 *Analysis of Time Series* at UC Berkeley (spring 2022).
The course was taught in R; every model here was implemented from scratch in Python and Numba instead.
These are the homework solutions that use yassm (homework 3–5),
as [jupytext](https://jupytext.readthedocs.io/) percent-format scripts:
open them as notebooks in Jupyter, or run them as plain Python scripts.

| Topic | Scripts |
| --- | --- |
| Kalman filtering & smoothing, MLE (local level / local linear / local ratio models), comparison with `statsmodels` | `hw3_q*.py`, `hw4_q1.py`, `hw4_q2.py` |
| Grid-based (non-Gaussian) filtering | `hw3_q6.py` |
| Gibbs sampling / forward-filtering backward-sampling | `hw4_q3.py`, `hw4_q5.py` |
| AR(2) direct sampling, MA(1) posterior | `hw4_q4.py`, `hw4_q6.py` |
| Bootstrap and optimal particle filters, MLE by grid search and `scipy.optimize` | `hw5_q*.py` |

Run them all (with reduced sample sizes) as integration tests:

```sh
pixi run examples
```

or a single one at full scale (some take a long time):

```sh
pixi run python examples/stat248/hw5_q5.py
```

`YASSM_EXAMPLES_FAST=1` shrinks sample sizes and grids so that each script runs in seconds;
the results are then not meaningful.
The datasets and their sources are listed in [`data/README.md`](data/README.md).
