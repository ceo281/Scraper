"""Walk-forward features and a ranked conditional-logit (Plackett-Luce) model.

Why this model
--------------
A race is a choice among the runners in *that* race, so the natural model is
a conditional logit: each horse gets a strength exp(x . beta) and its chance
of winning is its strength divided by the field's total. Fitting on only the
winner throws away most of the information, so the likelihood uses the top
three finishers in order (Plackett-Luce): P(1st) * P(2nd | 1st gone) * ...

Leakage rule
------------
Every history feature for year Y (trainer/jockey/horse Cup record) is built
from years strictly before Y. The backtest also trains only on years before
the year it predicts, so past winners are genuinely out-of-sample.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURES = [
    "weight_rel",      # kg above/below the race's mean weight (handicap = rating)
    "barrier_frac",    # barrier / field size (0 = rail, 1 = widest)
    "bred_north",      # northern-hemisphere bred (GB/IRE/FR/GER/JPN/USA...)
    "trainer_wins",    # log1p(prior Cup wins by any listed trainer)
    "trainer_top3",    # log1p(prior Cup top-3 finishes by any listed trainer)
    "jockey_wins",     # log1p(prior Cup wins by the jockey)
    "jockey_top3",     # log1p(prior Cup top-3 finishes by the jockey)
    "horse_prev_top5", # horse finished top 5 in an earlier Cup
    "horse_prev_ran",  # horse has run in an earlier Cup
]


def _people(s: str) -> list[str]:
    if not isinstance(s, str):
        return []
    out = []
    for part in re.split(r"&|/|\band\b", s):
        p = re.sub(r"^(sir|dr)\s+", "", part.strip().lower())
        p = re.sub(r"[^a-z' ]", "", p).strip()
        if p:
            out.append(p)
    return out


def _horse_key(s: str) -> str:
    return re.sub(r"[^a-z]", "", str(s).lower())


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["year", "place"], na_position="last").copy()
    df["horse_key"] = df["horse"].map(_horse_key)
    out = []
    for year, race in df.groupby("year", sort=True):
        prior = df[df["year"] < year]
        tw, tt, jw, jt = {}, {}, {}, {}
        for _, r in prior.iterrows():
            win, top3 = r["place"] == 1, r["place"] <= 3
            for t in _people(r["trainer"]):
                tw[t] = tw.get(t, 0) + win
                tt[t] = tt.get(t, 0) + top3
            for j in _people(r["jockey"]):
                jw[j] = jw.get(j, 0) + win
                jt[j] = jt.get(j, 0) + top3
        best_prev = prior.groupby("horse_key")["place"].min()
        race = race.copy()
        race["trainer_wins"] = [math.log1p(max([tw.get(t, 0) for t in _people(s)] or [0])) for s in race["trainer"]]
        race["trainer_top3"] = [math.log1p(max([tt.get(t, 0) for t in _people(s)] or [0])) for s in race["trainer"]]
        race["jockey_wins"] = [math.log1p(max([jw.get(j, 0) for j in _people(s)] or [0])) for s in race["jockey"]]
        race["jockey_top3"] = [math.log1p(max([jt.get(j, 0) for j in _people(s)] or [0])) for s in race["jockey"]]
        race["horse_prev_ran"] = race["horse_key"].isin(best_prev.index).astype(float)
        race["horse_prev_top5"] = race["horse_key"].map(best_prev).le(5).astype(float)
        race["weight_rel"] = race["weight"] - race["weight"].mean()
        race["barrier_frac"] = race["barrier"] / len(race)
        out.append(race)
    return pd.concat(out, ignore_index=True)


def _design(race: pd.DataFrame, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Standardised feature matrix. A feature missing for the whole race is set
    to 0 for every runner: constant within a race, so it cancels out of the
    conditional logit and neither helps nor hurts. Partial gaps get the race
    median."""
    X = race[FEATURES].astype(float).copy()
    for c in FEATURES:
        if X[c].isna().all():
            X[c] = mu[FEATURES.index(c)]
        else:
            X[c] = X[c].fillna(X[c].median())
    return (X.to_numpy() - mu) / sd


@dataclass
class CupModel:
    l2: float = 1.0
    depth: int = 3          # how many placings feed the ranked likelihood
    beta: np.ndarray | None = None
    mu: np.ndarray | None = None
    sd: np.ndarray | None = None

    def fit(self, train: pd.DataFrame, iters: int = 3000, lr: float = 0.05) -> "CupModel":
        train = train[train["weight"].notna()]  # weight is the core signal
        F = train[FEATURES].astype(float)
        self.mu = F.mean().fillna(0).to_numpy()
        self.sd = F.std().replace(0, 1).fillna(1).to_numpy()
        races = []
        for _, race in train.groupby("year"):
            if race["place"].notna().sum() < 1:
                continue
            X = _design(race, self.mu, self.sd)
            order = [i for i in np.argsort(race["place"].to_numpy()) if race["place"].iloc[i] <= self.depth]
            races.append((X, order))
        n_obs = sum(len(o) for _, o in races)
        beta = np.zeros(len(FEATURES))
        # Plain gradient ascent; the problem is convex and tiny.
        for _ in range(iters):
            grad = -self.l2 * beta
            for X, order in races:
                alive = np.ones(len(X), bool)
                for i in order:
                    u = X[alive] @ beta
                    p = np.exp(u - u.max()); p /= p.sum()
                    grad += X[i] - p @ X[alive]
                    alive[i] = False
            beta += lr * grad / max(n_obs, 1)
        self.beta = beta
        return self

    def predict(self, race: pd.DataFrame) -> np.ndarray:
        u = _design(race, self.mu, self.sd) @ self.beta
        p = np.exp(u - u.max())
        return p / p.sum()

    def coefficients(self) -> pd.Series:
        return pd.Series(self.beta, index=FEATURES).sort_values(key=abs, ascending=False)


def backtest(df: pd.DataFrame, first_test: int = 2006, min_train_races: int = 8, l2: float = 1.0) -> pd.DataFrame:
    """Predict every past Cup using only earlier Cups, and score it against the
    real winner. This is the 'old winners as the error measure' step."""
    rows = []
    has_result = df.groupby("year")["place"].apply(lambda s: (s == 1).any())
    for year in sorted(df["year"].unique()):
        if year < first_test or not has_result.get(year, False):
            continue
        train = df[df["year"] < year]
        if train[train["weight"].notna()]["year"].nunique() < min_train_races:
            continue
        race = df[df["year"] == year].reset_index(drop=True)
        if race["weight"].isna().all():
            continue
        p = CupModel(l2=l2).fit(train).predict(race)
        w = int(np.flatnonzero(race["place"].to_numpy() == 1)[0])
        rank = int((p > p[w]).sum()) + 1
        n = len(race)
        top_weight_rank = int((race["weight"] > race["weight"].iloc[w]).sum()) + 1
        rows.append({
            "year": year,
            "winner": race["horse"].iloc[w],
            "model_pick": race["horse"].iloc[int(p.argmax())],
            "winner_rank": rank,
            "winner_prob": p[w],
            "field": n,
            "logloss": -math.log(p[w]),
            "uniform_logloss": math.log(n),
            "top_weight_rank": top_weight_rank,
        })
    return pd.DataFrame(rows)


def summarise(bt: pd.DataFrame) -> dict:
    n = len(bt)
    return {
        "races_tested": n,
        "top1_hits": int((bt.winner_rank == 1).sum()),
        "top3_hits": int((bt.winner_rank <= 3).sum()),
        "top5_hits": int((bt.winner_rank <= 5).sum()),
        "random_top1_expected": float((1 / bt.field).sum()),
        "random_top5_expected": float((5 / bt.field).sum()),
        "mean_winner_rank": float(bt.winner_rank.mean()),
        "random_mean_rank": float(((bt.field + 1) / 2).mean()),
        "mean_logloss": float(bt.logloss.mean()),
        "uniform_logloss": float(bt.uniform_logloss.mean()),
    }
