# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from pyg_base import eq, drange, nona, near
from pyg_timeseries import rolling_quantile, rolling_quantile_


def ck(res, df):
    assert len(nona(df)) == len(nona(res))
    assert near(res, df)

def test_short_dataframe():
    a = pd.DataFrame(np.random.normal(0,1,(1000,3)), columns = ['a', 'b', 'c'], index = drange(-999))
    n = 500; quantile = 0.5
    axis = 0; min_periods = None; data = None; state = instate = None
    for ts in [a, a.iloc[:100]]:
        for min_periods in [1,50,None]:
            for interpolation in ['linear', 'lower', 'higher', 'midpoint']:
                res = rolling_quantile_(ts, 500, 0.5, min_periods = min_periods, interpolation = interpolation)['data']
                df = ts.rolling(500, min_periods=min_periods).quantile(0.5, interpolation = interpolation)
                ck(res, df)
