import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import WLS
from statsmodels.tools import add_constant

def calc_midcap(group):
    X = add_constant(group['Size_abs_cube'])
    y = group['Size']
    weights = np.sqrt(group['circ_mv'])
    model = WLS(y, X, weights=weights)
    result = model.fit()
    group['MIDCAP'] = result.resid
    return group


def mad_clip(x, n=5):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    upper = med + n * mad
    lower = med - n * mad
    return np.clip(x, lower, upper)

if __name__ == "__main__":
    group = pd.DataFrame({"Size": [-1.4, -.7, -.1, .2, .9, 1.5], "circ_mv": [12, 17, 23, 29, 34, 56]})
    group["Size_abs_cube"] = abs(group.Size) ** 3
    result = calc_midcap(group)
    assert np.isclose(np.average(result.MIDCAP, weights=np.sqrt(group.circ_mv)), 0)
    print(result)
