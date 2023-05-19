from pyg_timeseries import ewma, ewmstd, ewmrms, ewmskew, ewmLR, ewmcor, ewmcorr, ts_max
from pyg_base import eq, dt, calendar, drange, Dict
import pandas as pd; import numpy as np

t = dt(2021,3,1)
cal = calendar('US')

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
    res = ewmcorr(a, 20)
    for i in range(10):
        for j in range(i):        
            x = ewmcor(a[:,i], a[:, j], 20)
            y = res[:, i, j]
            assert ts_max(abs(x - y)[20:])<0.05

    
def test_ewm_empty():
    for a in (np.array([]), pd.Series([],[], dtype = float), pd.DataFrame([],[])):
        for f in [ewma, ewmstd, ewmrms, ewmskew]:
            res = f(a, 20)
            assert eq(res, a)
