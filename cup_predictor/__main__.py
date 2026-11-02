"""CLI: python -m cup_predictor [build|backtest|predict] [--year 2026]

build     fetch Wikipedia pages (cached) and write dataset/runners.csv
backtest  predict every past Cup using only earlier Cups; score vs real winner
predict   train on all completed Cups and rank the target year's field
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .fetch import fetch_range
from .model import CupModel, add_features, backtest, summarise
from .parse import build_dataset

DATASET = Path(__file__).resolve().parent / "dataset" / "runners.csv"


def build(first: int, last: int) -> pd.DataFrame:
    df = build_dataset(fetch_range(first, last))
    DATASET.parent.mkdir(exist_ok=True)
    df.to_csv(DATASET, index=False)
    print(f"wrote {len(df)} runners from {df.year.nunique()} Cups -> {DATASET}")
    return df


def load() -> pd.DataFrame:
    if not DATASET.exists():
        raise SystemExit("no dataset yet: run `python -m cup_predictor build` first")
    return add_features(pd.read_csv(DATASET))


def main() -> None:
    ap = argparse.ArgumentParser(prog="cup_predictor")
    ap.add_argument("command", choices=["build", "backtest", "predict"])
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--first", type=int, default=1990)
    ap.add_argument("--l2", type=float, default=100.0)  # heavy shrinkage: least-bad in backtest
    a = ap.parse_args()
    pd.set_option("display.width", 200)

    if a.command == "build":
        build(a.first, a.year)
        return

    df = load()
    if a.command == "backtest":
        bt = backtest(df[df.year < a.year], l2=a.l2)
        show = bt.assign(winner_prob=bt.winner_prob.map("{:.1%}".format))
        print(show[["year", "winner", "winner_rank", "winner_prob", "field", "model_pick", "top_weight_rank"]].to_string(index=False))
        s = summarise(bt)
        print()
        print(f"Races tested           : {s['races_tested']}")
        print(f"Winner was model's #1  : {s['top1_hits']}  (random would get ~{s['random_top1_expected']:.1f})")
        print(f"Winner in model's top 3: {s['top3_hits']}")
        print(f"Winner in model's top 5: {s['top5_hits']}  (random would get ~{s['random_top5_expected']:.1f})")
        print(f"Mean rank of winner    : {s['mean_winner_rank']:.1f}  (random {s['random_mean_rank']:.1f})")
        print(f"Log-loss on winner     : {s['mean_logloss']:.3f}  (uniform guess {s['uniform_logloss']:.3f}; lower is better)")
        return

    train = df[(df.year < a.year) & df.place.notna()]
    race = df[df.year == a.year].reset_index(drop=True)
    if race.empty:
        raise SystemExit(f"no field for {a.year}: run build with --year {a.year}")
    m = CupModel(l2=a.l2).fit(train)
    race["win_prob"] = m.predict(race)
    race["fair_odds"] = 1 / race["win_prob"]
    out = race.sort_values("win_prob", ascending=False)
    print(f"Model trained on {train[train.weight.notna()].year.nunique()} Cups with weights recorded.\n")
    print("Coefficients (standardised; + helps, - hurts):")
    print(m.coefficients().round(3).to_string(), "\n")
    print(out[["horse", "trainer", "jockey", "weight", "barrier", "win_prob", "fair_odds"]]
          .assign(win_prob=out.win_prob.map("{:.1%}".format), fair_odds=out.fair_odds.map("${:.1f}".format))
          .to_string(index=False))


if __name__ == "__main__":
    main()
