"""Melbourne Cup winner predictor.

Pipeline: fetch past fields from Wikipedia -> normalise -> walk-forward
conditional-logit model -> backtest against every past winner -> predict.
"""
