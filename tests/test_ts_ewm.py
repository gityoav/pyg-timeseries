from pyg_timeseries import ewma, ewmstd, ewmrms, ewmskew, ewmLR
from pyg_timeseries import ewmcor, ewmxcor, ewmcorr, ts_max, ewmcorrelation, ewmcovar, ewmcovariance
from pyg_base import eq, dt, calendar, drange, Dict, is_pd, is_df, is_series
import pandas as pd; import numpy as np

t = dt(2021,3,1)
cal = calendar('US')



def test_ewm_with_weights():
    a = pd.Series(np.arange(1,1001), drange(-999))
    wgt = 1/abs(a)
    res = ewma(a, 10)
    res1 = ewma(a, 10, wgt = 1/abs(a))
    assert np.all(res1.iloc[1:] < res.iloc[1:])
    res2 = ewma(a, 10, wgt = 1/abs(a)**2)
    assert np.all(res2.iloc[1:] < res1.iloc[1:])
    for func in (ewma, ewmstd, ewmrms, ewmskew):
        base = func(a, 10)        
        res = func(a, 10, wgt = 1/abs(a))
        assert abs((res-base)/(res+base).iloc[-500:]).max() < 0.1
        

    

def test_ewm_monthly():    
    months = drange('-500m', t, '1m')
    a = pd.Series(np.random.normal(0,1,501), months)
    days = cal.drange('-500m', t, 1)
    for f in ewma, ewmstd, ewmskew, ewmrms:
        assert eq(f(a, 3).reindex(days), f(a.reindex(days), 3))
        assert eq(f(a.reindex(days).ffill(), 3, time = 'm'), f(a.reindex(days),3).ffill())


def test_ewm_weekly():    
    months = drange('-500w', t, '1w')
    a = pd.Series(np.random.normal(0,1,501), months)
    days = cal.drange('-500w', t, '1b')
    for f in ewma, ewmstd, ewmskew, ewmrms:
        assert eq(f(a, 3).reindex(days), f(a.reindex(days), 3))
        assert eq(f(a.reindex(days).ffill(), 3, time = 'w'), f(a.reindex(days),3).ffill())

def test_ewm_yearly():    
    t = dt(2021,1,1)
    yearly = drange('-100y', t, '1y')
    a = pd.Series(np.random.normal(0,1,101), yearly)
    days = cal.drange('-100y', t, 1)
    for f in ewma, ewmstd, ewmskew, ewmrms:
        assert eq(f(a, 3).reindex(days), f(a.reindex(days), 3))
        assert eq(f(a.reindex(days).ffill(), 3, time = 'y'), f(a.reindex(days),3).ffill())

def test_ewm_LR():
    a0 = pd.Series(np.random.normal(0,1,10000), drange(-9999))
    a1 = pd.Series(np.random.normal(0,1,10000), drange(-9999))
    b = (a0 - a1) + pd.Series(np.random.normal(0,1,10000), drange(-9999))
    a = pd.concat([a0,a1], axis=1)
    LR = ewmLR(a,b,50)
    assert abs(LR.m.mean()[0]-1)<0.5
    assert abs(LR.m.mean()[1]+1)<0.5
    a = Dict(a0 = a0, a1 = a1)
    LR2 = ewmLR(a,b,50)
    assert eq(LR2.a0.m, LR.m[0])
    assert eq(LR2.a0.c, LR.c[0])
    assert 'state' not in LR2.a0
    a = [a0,a1]
    LR3 = ewmLR(a,b,50)
    assert eq(LR2.a0, LR3[0])
    assert eq(LR2.a1, LR3[1])

def test_ewmcor():
    """ the two functions are not identical but should be pretty close """
    np.random.seed(0)
    a = np.random.normal(0,1,(1000,10)) 
    a = a + np.random.normal(0, 1, (1000, 1))
    res = ewmcorrelation(a, 20)
    res2 = ewmxcor(a,a,20)
    assert np.max(abs(res2[-1]-res[-1])) < 0.02
    res = ewmcorr(a, 20)
    res2 = ewmcor(a,a,20)
    assert np.max(abs(res2[-1]-res[-1])) < 0.1
    assert np.max(abs(res[-1])) <= 1.00001
    assert np.max(abs(res2[-1])) <= 1.00001

    for overlapping in range(1,5):
        for bias in [True, False]:
            res = ewmcorr(a, 20, overlapping = overlapping, bias = bias)
            res2 = ewmcor(a,a,20, overlapping = overlapping, bias = bias)
            assert np.max(abs(res2[-1]-res[-1])) < 0.2 * overlapping
            assert np.max(abs(res[-1])) <= 1.00001
            assert np.max(abs(res2[-1])) <= 1.00001
            res = ewmcorrelation(a, 20, overlapping = overlapping,bias = bias)
            res2 = ewmxcor(a,a,20, overlapping = overlapping, bias = bias)
            assert np.max(abs(res2[-1]-res[-1])) < 0.2 * overlapping
            assert np.max(abs(res[-1])) <= 1.00001
            assert np.max(abs(res2[-1])) <= 1.00001
        
    ### now type checking
    res = ewmxcor(a, b = pd.Series(a[:,0], drange(999)), n = 20)
    assert isinstance(res, np.ndarray)
    res = ewmxcor(pd.DataFrame(a, columns = list('abcdefghij'), index = drange(999)), b = pd.Series(a[:,0], drange(999)), n = 20)
    assert is_df(res) and list(res.columns) == list('abcdefghij')
    res = ewmxcor(pd.DataFrame(a, columns = list('abcdefghij'), index = drange(999)), a[:,0], n = 20)
    assert is_df(res) and list(res.columns) == list('abcdefghij')
    res = ewmxcor(a[:,1], b = pd.Series(a[:,0], drange(999)), n = 20)
    assert is_series(res)
    res = ewmxcor(b = a[:,1], a = pd.Series(a[:,0], drange(999)), n = 20)
    assert is_series(res)


def test_ewmxcor_agrees_with_ewm_correlation():
    a = np.random.normal(0,1,(2000,2))
    for overlapping in range(1,5):
        for n in range(5,30,5):
            for bias in (True, False):
                x = ewmxcor(a,a,n, bias = bias, overlapping = overlapping)[:,0,1]
                y = ewmcorrelation(a, n, bias = bias, overlapping = overlapping)[:,0,1]
                assert np.max(abs(x-y)[100:]) < 1e-6
        
    
def test_ewm_empty():
    for a in (np.array([]), pd.Series([],[], dtype = float), pd.DataFrame([],[])):
        for f in [ewma, ewmstd, ewmrms, ewmskew]:
            res = f(a, 20)
            assert eq(res, a)
