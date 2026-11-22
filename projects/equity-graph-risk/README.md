# Learning correlations between stocks

How can a portfolio model anticipate which stocks will move together?

A temporal Transformer embeds 36 features per stock. A second attention layer uses historical correlations as an edge bias. Pairwise cosine similarities of the final embeddings produce a symmetric correlation matrix. The broader team study combined this risk estimate with a return forecast and a mean–variance optimizer.

The research draft reports a 19.78% cumulative return and 15.70% annualized volatility for its evaluated stock portfolio. The public example runs the original risk encoder and decoder on synthetic tensors and checks the shape, diagonal and symmetry of their output.

## Run the example

From this project directory, using Python 3.12 in a virtual environment:

```sh
python -m pip install -r requirements.txt
python example.py
```

The archived return-side network is available in [return_forecaster.py](return_forecaster.py). It combines an LSTM, positional encoding and a Transformer. The runnable example above exercises the separate correlation model.

[Return to the portfolio](../../README.md)
