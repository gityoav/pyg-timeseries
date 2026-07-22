# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from pyg_base import eq, drange, nona
from pyg_timeseries import rolling_quantile, rolling_quantile_

def test_short_dataframe():
    a = pd.DataFrame(np.random.normal(0,1,(1000,3)), columns = ['a', 'b', 'c'], index = drange(-999))
    n = 500; quantile = 0.5
    axis = 0; min_periods = None; data = None; state = instate = None
    res = rolling_quantile_(a, 500, 0.5)
    res50 = rolling_quantile_(a, 500, 0.5, min_periods = 50)
    res1 = rolling_quantile_(a, 500, 0.5, min_periods = 1)
    assert len(nona(res['data'])) == 501
    assert len(nona(res50['data'])) == 951
    assert len(nona(res1['data'])) == 1000

    a = a.iloc[:100]
    res = rolling_quantile_(a, 500, 0.5)
    res50 = rolling_quantile_(a, 500, 0.5, min_periods = 50)
    res1 = rolling_quantile_(a, 500, 0.5, min_periods = 1)
    assert len(nona(res['data'])) == 0
    assert len(nona(res50['data'])) == 51
    assert len(nona(res1['data'])) == 100
