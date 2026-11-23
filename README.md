# Linsong Ge

I study computer science and finance at South China University of Technology and spent a semester at UC Berkeley. I work on quantitative portfolio research. Two questions have shaped my projects so far: how closely can a small, tradable basket follow an index, and how much can the history of two stocks tell us about the risk of holding them together?

## Portfolio construction and trading signals

**[Tracking the CSI 300 with mixed-integer optimization](https://github.com/BarryGreenton/Stock-Index-Fitting-MILP)**  
During my asset-management internship, I built a Gurobi model that turns index weights into an affordable basket of whole trading lots. I tested the holdings against minute-by-minute index moves and compared ways of penalizing tracking deviations. A colleague’s [companion implementation](https://github.com/Wang-Bingjie-nus/Stock-Index-Fitting) explores a different construction rule: start with a feasible greedy basket, then trade off allocation error against correlation-based tracking risk. It includes the team’s comparative charts.

**[Learning which stocks move together](https://github.com/BarryGreenton/equity-graph-risk)**  
I worked on the risk branch of a student portfolio research project. A temporal Transformer encodes each stock’s history; graph attention uses past relationships to estimate future pairwise correlations. In the [full team implementation](https://github.com/Xyy1001/ETF_ML_SCUT), a separate LSTM–Transformer branch forecasts prices. The backtest combines expected returns and a covariance matrix to calculate portfolio weights.

**[Price-action factor research at Minsy AI](https://github.com/asech109/pa-lib)** · private team repository  
During a month-long remote internship with a Berkeley classmate, I researched price and volume signals for a trading research assistant. I organized the factor workspace, translated trading ideas into testable rules, and built a backtest to compare signals and simple weighted combinations against baseline strategies.

| Research project | What I investigated |
| --- | --- |
| [ETF allocation](https://github.com/BarryGreenton/etf-allocation) | How a rolling covariance estimate changes the risk input to a ten-ETF allocation model. I worked on the covariance calculation and backtesting. |
| [ETF rotation](https://github.com/BarryGreenton/etf-rotation) | How a price-change rule and a trend score rank the same four funds, with positions lagged one session in the backtest. |
| [China equity style factors](https://github.com/BarryGreenton/equity-style) | How industry adjustment and nonlinear size exposure change the information carried by market capitalization. |
| [Chinese equity market risk](https://github.com/BarryGreenton/market-risk) | How turnover, reversal and other daily indicators can inform exposure to equities, bonds and cash. |

## Statistical modeling and computer vision

**[The halo effect in speed-dating decisions](https://github.com/DaDa-Chan/speed-dating-halo)** · UC Berkeley, Linear Models  
Our team compared the traits people said they valued with their decisions after meeting a partner. We used Elastic Net with cross-validation grouped by participant, so one person’s dates stayed in the same fold. Participant fixed effects let us compare choices made by the same person. The Frisch–Waugh–Lovell theorem gives the interpretation: after accounting for the other ratings, how much of the remaining variation in decisions is associated with attractiveness?

- [Olympic medal forecasting](https://github.com/BarryGreenton/olympic-medals) · For 2025 MCM Problem C, our team separated the probability of winning a medal from the expected count for medal-winning delegations.
- [Underwater image enhancement](https://github.com/BarryGreenton/underwater-imaging) · Our team used colour, brightness and edge statistics to select corrections for different kinds of image degradation.

