# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from pyg_base import eq, drange, nona
from pyg_timeseries import rolling_quantile, rolling_quantile_


def ck(res, df):
    assert len(nona(df)) == len(nona(res))
    assert eq(df, res) or abs(df - res).max().max()<1e-10

def test_short_dataframe():
    a = pd.DataFrame(np.random.normal(0,1,(1000,3)), columns = ['a', 'b', 'c'], index = drange(-999))
    n = 500; quantile = 0.5
    axis = 0; min_periods = None; data = None; state = instate = None
    res = rolling_quantile_(a, 500, 0.5)['data']
    res50 = rolling_quantile_(a, 500, 0.5, min_periods = 50)['data']
    res1 = rolling_quantile_(a, 500, 0.5, min_periods = 1)['data']
    df = a.rolling(500).quantile(0.5)
    df50 = a.rolling(500, min_periods = 50).quantile(0.5)
    df1 = a.rolling(500, min_periods = 1).quantile(0.5)

    
    ck(res, df)
    ck(res1, df1)
    ck(res50, df50)

    a = a.iloc[:100]
    res = rolling_quantile_(a, 500, 0.5)['data']
    res50 = rolling_quantile_(a, 500, 0.5, min_periods = 50)['data']
    res1 = rolling_quantile_(a, 500, 0.5, min_periods = 1)['data']
    df = a.rolling(500).quantile(0.5)
    df50 = a.rolling(500, min_periods = 50).quantile(0.5)
    df1 = a.rolling(500, min_periods = 1).quantile(0.5)

    ck(res, df)
    ck(res1, df1)
    ck(res50, df50)
