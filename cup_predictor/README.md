# Melbourne Cup predictor

```
python -m cup_predictor build      # fetch Wikipedia fields 1990-2026 (cached), write dataset/runners.csv
python -m cup_predictor backtest   # predict each past Cup using only earlier Cups, score vs real winner
python -m cup_predictor predict --year 2026
```

**Model:** ranked conditional logit (Plackett-Luce over the top 3 placings) on
weight relative to the field, barrier, northern-hemisphere breeding, and the
prior Cup records of trainer, jockey and horse. All history features use only
earlier years, so there is no look-ahead.

**Backtest (2009-2025, 16 Cups, walk-forward):** the model has *no demonstrated
edge*. Its log-loss is about the same as a uniform guess (3.20 vs 3.18), it
never ranked the winner first, and the winner's mean rank was 11.8 of 24
(random is 12.5). Treat the output as a field ranked by small structural
tilts, not as a tip.

**Why:** public field data does not include the information that predicts
the Cup: form, ratings, lead-up runs, track conditions and the betting market.
No historical starting prices were available, and SP is the single strongest
predictor. Add those before expecting any skill. Data gaps: Wikipedia lacks
weights before 2003 in most years, barriers in some years, and breeding origin
in others. A feature that is blank for a whole race cancels out of the model.
