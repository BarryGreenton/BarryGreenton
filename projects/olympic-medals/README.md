# Olympic medal forecasting

Predicting a first medal and predicting a medal count are different statistical tasks.

For 2025 MCM Problem C the team used 11 indicators, a 100-tree random forest classifier for medal incidence, and a 200-tree regressor for positive counts. Historical Games supplied the backtests. Bootstrap intervals expressed uncertainty around the forecasts for Los Angeles 2028.

Entity preparation came first. Historical delegations appeared under multiple names, and the cleaning script restricted its analysis to editions from 1948 onward. The public utility preserves its delegation dictionary and exports the standardized table from a supplied competition CSV.

## Run the example

From this project directory, using Python 3.12 in a virtual environment:

```sh
python -m pip install -r requirements.txt
python example.py
```

For the historical input, run `python example.py summerOly_medal_counts.csv --output standardized.csv`. The input requires the columns `NOC` and `Year`. Unmapped names are labelled `Unknown` for review.

**Historical replay.** Applying the original name dictionary and date cutoff to 1,435 medal-table rows retained 1,211 rows. 0 retained rows were labelled unknown. This check covers the input-mapping step of the forecasting study.

[Return to the portfolio](../../README.md)
