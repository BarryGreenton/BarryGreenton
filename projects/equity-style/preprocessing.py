import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def mad_based_outlier_clip(df, col, group_col='trade_date', n=5):

    def clip_group(x):
        x_median = np.median(x)
        x_mad = np.median(np.abs(x - x_median))
        upper = x_median + n * x_mad
        lower = x_median - n * x_mad
        return np.clip(x, lower, upper)
    df[col + '_clip'] = df.groupby(group_col)[col].transform(clip_group)
    return df


def fillna_by_industry_median(group, col):
    median = group[col].median()
    group[col] = group[col].fillna(median)
    return group


def zscore(x):
    mean = x.mean()
    std = x.std()
    return (x - mean) / std if std > 0 else 0


def industry_neutralize(group, col, industry_col):
    dummies = pd.get_dummies(group[industry_col], prefix='ind')
    X = dummies.values
    y = group[col].values
    if X.shape[1] == 0 or len(y) == 0:
        return pd.Series([np.nan] * len(y), index=group.index)
    reg = LinearRegression(fit_intercept=True)
    reg.fit(X, y)
    y_pred = reg.predict(X)
    residual = y - y_pred
    return pd.Series(residual, index=group.index)
