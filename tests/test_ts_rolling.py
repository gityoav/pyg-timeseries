from pyg_timeseries import diff, ratio
from pyg_base import eq, drange
import pandas as pd
import numpy as np

def test_diff_with_time():
    a = 1. * np.arange(10)**2
    time = np.array([0,0,1,1,1,2,2,2,2,3])
    dff = np.array([np.nan, np.nan, 4-1, 9-1, 16-1, 25-16, 36-16, 49-16, 64-16, 81-64])
    assert eq(dff, diff(a, n=1, time = time))    
    rto = np.array([np.nan, np.nan, 4/1, 9/1, 16/1, 25/16, 36/16, 49/16, 64/16, 81/64])
    assert eq(rto, ratio(a, n=1, time = time))    
    dff2 = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 25-1, 36-1, 49-1, 64-1, 81-16])
    assert eq(dff2, diff(a, n=2, time = time))    
    rto2 = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 25/1, 36/1, 49/1, 64/1, 81/16])
    assert eq(rto2, ratio(a, n=2, time = time))    


def test_diff():
    a = 1. * np.arange(10)**2
    state = 0.
    assert diff(a, state = state)[0] == 0

    b = pd.Series(a, drange(9))
    assert eq(diff(b), b.diff())
    assert diff(b, state = state).iloc[0] == 0

    d = pd.DataFrame(dict(a = a), drange(9))
    assert eq(diff(d), d.diff())

