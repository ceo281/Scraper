"""Monte Carlo race simulation with coefficient uncertainty.

Each run:
  1. draws a plausible set of factor weights (beta) from the model's
     uncertainty: a normal approximation around the fitted values, with spread
     taken from the curvature of the likelihood (Laplace approximation);
  2. runs the race once: every horse gets score + random luck (Gumbel noise),
     and the finishing order is the sort of those totals. Sorting
     score + Gumbel noise is mathematically the same as drawing from the
     model's win/place probabilities (Plackett-Luce).

Repeating this N times gives win %, top-3 %, and how stable each horse's
chance is when the model's factor weights are uncertain.

Sensitivity: rerun with one factor switched off (its weight forced to 0) and
see how much each horse's win % moves. That shows which factor is driving
each horse's rating.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model import FEATURES, CupModel, _design


def coefficient_covariance(model: CupModel, train: pd.DataFrame) -> np.ndarray:
    """Inverse of the negative Hessian of the penalised ranked log-likelihood."""
    train = train[train["weight"].notna()]
    H = model.l2 * np.eye(len(FEATURES))
    for _, race in train.groupby("year"):
        if race["place"].notna().sum() < 1:
            continue
        X = _design(race, model.mu, model.sd)
        places = race["place"].to_numpy()
        order = [i for i in np.argsort(places) if places[i] <= model.depth]
        alive = np.ones(len(X), bool)
        for i in order:
            Xa = X[alive]
            u = Xa @ model.beta
            p = np.exp(u - u.max()); p /= p.sum()
            m = p @ Xa
            H += (Xa * p[:, None]).T @ Xa - np.outer(m, m)
            alive[i] = False
    return np.linalg.inv(H)


def simulate(model: CupModel, train: pd.DataFrame, race: pd.DataFrame,
             runs: int = 10_000, seed: int = 0, uncertain: bool = True,
             zero: str | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = _design(race, model.mu, model.sd)
    beta = model.beta.copy()
    if uncertain:
        cov = coefficient_covariance(model, train)
        betas = rng.multivariate_normal(beta, cov, size=runs)
    else:
        betas = np.tile(beta, (runs, 1))
    if zero is not None:
        betas[:, FEATURES.index(zero)] = 0.0
    scores = betas @ X.T                               # runs x horses
    luck = rng.gumbel(size=scores.shape)
    order = np.argsort(-(scores + luck), axis=1)       # finishing order per run
    n = len(race)
    wins = np.bincount(order[:, 0], minlength=n)
    top3 = np.zeros(n)
    for k in range(min(3, n)):
        top3 += np.bincount(order[:, k], minlength=n)
    p = np.exp(scores - scores.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    lo, hi = np.percentile(p, [10, 90], axis=0)
    return pd.DataFrame({
        "horse": race["horse"].to_numpy(),
        "weight": race["weight"].to_numpy(),
        "barrier": race["barrier"].to_numpy(),
        "wins": wins,
        "win_pct": wins / runs,
        "top3_pct": top3 / runs,
        "chance_p10": lo,
        "chance_p90": hi,
    })


def sensitivity(model: CupModel, train: pd.DataFrame, race: pd.DataFrame,
                runs: int = 10_000, seed: int = 0) -> pd.DataFrame:
    """Win % with every factor on, then with each factor switched off.
    Uses the fitted weights (no coefficient noise) so differences come only
    from the factor removed."""
    base = simulate(model, train, race, runs, seed, uncertain=False)
    out = pd.DataFrame({"horse": base["horse"], "all_factors": base["win_pct"]})
    for f in FEATURES:
        out[f"no_{f}"] = simulate(model, train, race, runs, seed, uncertain=False, zero=f)["win_pct"]
    return out
