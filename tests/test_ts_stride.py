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

def test_rolling_quantile_state_is_safe_and_consistent():
    a = pd.Series(np.random.normal(0, 1, 800), drange(-799))
    whole = arq(a, 500, 0.05, min_periods=100)
    for k in [600, 799]:                       # split anywhere == whole
        old = arq_(a.iloc[:k], 500, 0.05, min_periods=100)
        new = arq(a.iloc[k:], 500, 0.05, state=old['state'])
        assert abs(pd.concat([old['data'], new]) - whole).max() < 1e-12
    # 1-row continuation (the run_scenario pattern) - no crash, correct last value
    base = arq_(a.iloc[:-1], 500, 0.05, min_periods=100)
    last = arq_(a.iloc[-1:], 500, 0.05, instate=base['state'])['data']
    assert abs(float(np.asarray(last)[-1]) - float(whole.iloc[-1])) < 1e-12
    # short series and short multi-column frame (len < window) must not crash
    assert arq(a.iloc[:60], 500, 0.05, min_periods=100).isna().all()
    df = pd.DataFrame(np.random.normal(0, 1, (60, 3)), drange(-59), columns=['a', 'b', 'c'])
    r = arq(df, 500, 0.05, min_periods=100)
    assert isinstance(r, pd.DataFrame) and r.shape == (60, 3)




def test_rolling_quantile_with_states():
    a = np.random.normal(0, 1, 100)
    b = np.random.normal(0, 1, 200)
    c = np.concatenate([a,b])
    for min_periods in [None, 100, 200]:
        resa = rolling_quantile_(a, 50, 0.05, min_periods=min_periods)
        resb = rolling_quantile_(b, 50, 0.05, min_periods=min_periods, instate = resa['state'])
        resc = rolling_quantile_(c, 50, 0.05, min_periods=min_periods)
        data_c = resc['data']
        data_ab = np.concatenate([resa['data'], resb['data']])
        assert near(data_ab, data_c)
        assert len(nona(data_c)) == len(nona(data_ab))


def test_rolling_quantile_is_nan_skipping():
    a = pd.Series(np.abs(np.random.normal(1, 0.2, 2000)), drange(-1999))
    gaps = np.zeros(2000, bool); gaps[100::9] = True
    a[gaps] = np.nan
    mine = arq(a, 50, 0.05, min_periods=10)
    drop = nona(a).rolling(50, min_periods=10).quantile(0.05).reindex(a.index)
    idx = mine.dropna().index.intersection(drop.dropna().index)
    assert len(idx) > 1000
    assert abs(mine.reindex(idx) - drop.reindex(idx)).max() < 1e-13

def test_rolling_quantile_matches_pandas_rampup():
    """rolling_quantile must equal pandas a.rolling(n, min_periods=mp).quantile(q) throughout the
    warm-up ramp on dense data - across every n/min_periods (incl. 1, n-1, n)/q - and start emitting
    at exactly the same point (this is the ramp off-by-one fix)."""
    a = pd.Series(np.random.normal(0, 1, 400), drange(-399))
    for n in [10, 50, 100]:
        for mp in [1, 2, n // 2, n - 1, n]:
            for q in [0.05, 0.25, 0.5, 0.9]:
                mine = arq(a, n, q, min_periods=mp)
                ref = a.rolling(n, min_periods=mp).quantile(q)
                assert abs(mine - ref).max() < 1e-12
                assert int((mine.isna() ^ ref.isna()).sum()) == 0   # ramp starts at the same point
    # multi-quantile keeps a column per quantile; multi-column keeps its own columns
    assert list(arq(a, 100, [0.1, 0.5, 0.9]).columns) == [0.1, 0.5, 0.9]
    df = pd.DataFrame(np.random.normal(0, 1, (400, 3)), a.index, columns=['x', 'y', 'z'])
    assert list(arq(df, 100, 0.05, min_periods=10).columns) == ['x', 'y', 'z']
