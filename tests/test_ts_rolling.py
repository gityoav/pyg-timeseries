from pyg_timeseries import diff, ratio
from pyg_timeseries import rolling_max, rolling_mean, rolling_median, rolling_quantile, rolling_rank, rolling_rms, rolling_skew, rolling_std, rolling_sum, ffill, staleness
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


def test_rolling_empty():
    for a in (np.array([]), pd.Series([],[], dtype = float), pd.DataFrame([],[])):
        for f in [rolling_max, rolling_mean, rolling_median, rolling_quantile, rolling_rank, rolling_rms, rolling_skew, rolling_std, rolling_sum]:
            res = f(a, 20)
            assert eq(res, a)


def test_not_enough_data_goes_to_nan():
    inputs = [np.full(100, np.nan), np.concatenate([np.full(100, np.nan), [0]*3])]
    for a in inputs:
        b = rolling_max(a, 10)
        assert np.min(np.isnan(b))        

def test_staleness():
    a = np.array([np.nan, 1, np.nan, np.nan, 2, np.nan, np.nan, np.nan])
    assert eq(staleness(a), np.array([1,0,1,2,0,1,2,3]))
    assert eq(staleness(a, state = 3), np.array([4,0,1,2,0,1,2,3]))
    assert eq(staleness(np.array([]), state = 3), np.array([]))
    a = np.array([[np.nan, 1, np.nan, np.nan, 2, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]).T
    assert eq(staleness(a), np.array([[1,0,1,2,0,1,2,3],[1,2,3,4,5,6,7,8]]).T)

def test_ffill():
    n=0; axis = 0; start_decay = None; end_decay = None; data = None; state = None
    a = np.array([np.nan, 1, np.nan, np.nan, 2, np.nan, np.nan, np.nan])
    assert eq(ffill(a), np.array([np.nan, 1,1,1,2,2,2,2]))    
    assert eq(ffill(a, state = dict(prev = 3, i = 2)), np.array([3, 1,1,1,2,2,2,2]))    
    assert eq(ffill(a, n = 2, state = dict(prev = 3, i = 2)), np.array([np.nan, 1,1,1,2,2,2,np.nan]))    
    assert eq(ffill(a, n = 2, end_decay = 2, state = dict(prev = 3, i = 2)), np.array([np.nan, 1,0.5,0,2,1,0,np.nan]))    

def test_ffill_2d():
    n=0; axis = 0; start_decay = None; end_decay = None; data = None; state = None
    a = np.array([np.nan, 1, np.nan, np.nan, 2, np.nan, np.nan, np.nan])
    t2 = lambda a: np.array([a,a]).T
    a = t2(a)
    assert eq(ffill(a), t2(np.array([np.nan, 1,1,1,2,2,2,2])))    
    assert eq(ffill(a, n = [1,2]), np.array([[np.nan, 1,1,np.nan,2,2,np.nan,np.nan], [np.nan, 1,1,1,2,2,2,np.nan]]).T)    

