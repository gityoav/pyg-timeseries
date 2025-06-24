
import numpy as np

def generate_autocor_process(autocor, n, seed = None):
    """
    >>> n = 10000
    >>> autocor = np.array([0.01] * 100)
    """
    if seed is not None:
        np.random.seed(seed)
    res = np.random.normal(0,1,n)
    a = len(autocor)
    c = autocor[::-1]
    for i in range(1, n):
        before = res[max(i-a,0):i]
        b = len(before)        
        res[i] += np.sum(c[-b:] * before)
    return res


import pandas as pd
from pyg_base import dt, drange
from pyg_timeseries import shift, ewmxcor, cumsum
ts = pd.Series(generate_autocor_process(autocor, n), drange(1-n))
ewmxcor(cumsum(ts), cumsum(shift(ts)), 100)