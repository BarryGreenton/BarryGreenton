# China equity size exposures

How much of a stock’s size exposure remains after accounting for its industry?

The CNE6 replication notebook aligns capitalization and industry by stock and trading date, clips extremes using median absolute deviations and estimates industry-neutral size. Its MIDCAP variant fits size to its absolute cube in a weighted cross-sectional regression, with weights proportional to the square root of circulating market value.

The public code preserves the notebook’s regression direction. A small cross-section exercises the residual calculation. The size notebook contains the implemented study. The archived volatility notebooks contain no comparable implementation.

## Run the example

From this project directory, using Python 3.12 in a virtual environment:

```sh
python -m pip install -r requirements.txt
python example.py
```

[preprocessing.py](preprocessing.py) contains the original offline notebook functions for MAD clipping by trading day, industry-median imputation, standardization and regression on industry dummies. The industry-neutral residual is the input to the example’s MIDCAP regression. These helpers additionally require scikit-learn.

[Return to the portfolio](../../README.md)
