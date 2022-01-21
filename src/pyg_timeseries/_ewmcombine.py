from pyg_base import df_reindex, df_concat, is_nums, Dict, mul_, loop, dictable, try_back, is_df, pd2np
from pyg_timeseries._rolling import na2v, ffill, v2na, fnna
from pyg_timeseries._ewm import ewma, ewmcorr, ewmstd
from pyg_timeseries._pandas import fnna_like
import numpy as np
import pandas as pd


@pd2np
def _ewmcombine(a, w, n = 1024, vol_days = None, full = False):
    """
    We assume all the joining together etc. is done
    
    Parameters
    ----------
    a : np.array
        2-dimensional array of signals
    w : np.ndarray
        2-dimensional array of weights

    Returns
    -------
    

    :Math:
    ------
    
    We have a collection of signals that we assume are RMS-1, i.e. N(0,1)-ish.
    Our relative weights w on each of the signals changes over time, perhaps because 
    - we allocate differently or 
    - because some signals only live at certain times of the year.


    We would like to create a variable X:
    X = \sum_i w_i * a_i

    But we also want to scale it so that X is RMS-1
    Since X is not necessarily RMS-1, we would want to divide it by its own vol estimate, which we can do with ewmrms(X, n)
    
    However, this does not take into account the fact that weights can change over time so the vol estimate yesterday may be irrelevant as w have changed.


    Single factor model
    -------------------        
    So let us assume a single factor correlation rho(t)

    E(X) = 0
    E(X^2) = X2 = \sum w_i^2 + \sum_{i<>j} w_i w_j rho(t)
        
    W1 = \sum w_i
    W2 = \sum w_i^2
    WIJ =  \sum_{i<>j} w_i w_j = \sum(i,j) w_i w_j - W2 = W1**2 - W2
    
    We reverse the equation and express rho as a function of E(X^2) 
    rho(t) = (X2 - W2)/WIJ
    
    This is a point estimate for rho. Once we have rho, we can estimate the unknown variance of X at each point of time.

    Full factor model
    -----------------
    Here we use ewmcorr to estimate a running correlation between all the signals. If we don't have a value (e.g. this is a recent timeseries so no data as yet)
    then we default to rho(t) from the single factor model above.
    
    Once we have the full correlation matrix estimation, we can estimate variance as...
    E(X^2) = w^T * C * w

    """
    fs = fnna(a)
    a_ = na2v(ffill(a))
    w_ = na2v(ffill(w))
    for i, j in enumerate(fs):
        w_[:j, i] = 0
    if vol_days:
        vols = v2na(ffill(ewmstd(a, vol_days)))
        a_ = a_ / vols
    else:
        vols = 1
    x = np.sum(a_ * w_, axis = 1)
    w1 = np.sum(w_, axis = 1)
    x2 = x**2
    w2 = np.sum(w_ ** 2, axis = 1)
    wij = w1 ** 2 - w2
    wij[wij == 0] = np.nan
    rho = (x2 - w2)/wij
    erho = ewma(rho, n) # this gives us our estimate for rho at time t
    if full:
        c = ewmcorr(a, n)
        c_ = np.zeros(c.shape)
        variance = np.empty(c.shape[0])
        cip = np.zeros(c.shape[1:])
        for i in range(c.shape[0]):
            ci = c[i]
            wi = w_[i]
            ci[np.isnan(ci)] = cip[np.isnan(ci)]
            ci[np.isnan(ci)] = erho[i]
            variance[i] = np.matmul(wi, np.matmul(ci,wi))
            c_[i] = ci
    else:
        variance = w2 + wij * erho # and hence an estimate for variance
    variance[variance<=0] = np.nan
    vol = ffill(np.sqrt(variance))
    mult = 1/vol
    data = x/vol
    normalized_mult = mult * w1
    return dict(data = data, vol = vol, rho = erho, mult = mult, normalized_mult = normalized_mult, cor = c_ if full else None, vols = vols if vol_days else 1)


@loop(list)
def _col(value, i):
    if isinstance(value, np.ndarray) and len(value.shape) > 1:
        return value[:, i] if value.shape[1]>1 else value[:, 0]
    elif isinstance(value, pd.DataFrame):
        return value.iloc[:, i] if value.shape[1]>1 else value.iloc[:,0]
    else:
        return value        
    

def ewmcombine(tss, wgts, n = 1024, vol_days = None, full = False, join = 'oj', method = None):
    """
    We expect tss and wgts to be provided as lists at the moment

    Parameters
    ----------
    tss : list of timeseries
        the data we want to combine into a single signal
    wgts : list of weights, either floats or timeseries
        
    n : TYPE, optional
        DESCRIPTION. The default is 1024.
    full : TYPE, optional
        DESCRIPTION. The default is False.
    join : TYPE, optional
        DESCRIPTION. The default is 'oj'.
    method : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    res : TYPE
        DESCRIPTION.

    """
    if is_df(tss):
        wgts = fnna_like(tss, wgts).ffill()
        return _ewmcombine(tss, wgts, n = n, vol_days = vol_days, full = full)      
    if len(tss) == 1:
        return Dict(data = tss[0], vol = None, rho = None, mult = 1/wgts[0], normalized_mult = 1, cor = None, vols = 1)            
    cols = [tuple(ts.columns) for ts in tss if is_df(ts) and ts.shape[1] > 1]
    if len(cols):
        assert len(set(cols)) == 1
        cols = list(cols[0])
        res = Dict(dictable([ewmcombine(_col(tss, i), _col(wgts, i), n = n, full = full, join = join, method = method) for i in range(len(cols))]))
        res = res.do(try_back(lambda v: df_concat(v, cols)))
        return res        
    if is_nums(wgts):
        tss, wgts = zip(*[(ts if w>0 else -ts, abs(w)) for ts, w in zip(tss, wgts) if w!=0]) ## we drop any zero weights and invert any negative weights
        a = df_concat(list(tss))
        w = fnna_like(a, wgts, 0.).values.astype(float)
    else:
        tss, wgts = zip(*[(mul_(ts,np.sign(df_reindex(w, ts, 'ffill'))), abs(w)) for ts, w in zip(tss, wgts)]) ## if we is a timeseries, we multiply the two after reindexing w to ts
        a = df_concat(list(tss))
        w = df_concat([df_reindex(wgt, a.index, method = 'ffill') for wgt in wgts]).values.astype(float)
    if len(tss) == 1:
        return Dict(data = tss[0], vol = None, rho = None, mult = 1/wgts[0], normalized_mult = 1, cor = None, vols = 1)            
    idx = a.index
    res = _ewmcombine(a.values, w, n = n, vol_days = vol_days, full = full)
    res = Dict({k : (pd.Series(v, idx) if len(v.shape) == 1 else [pd.Series(v[:,i], idx) for i in range(v.shape[1])]) if isinstance(v, np.ndarray) and v.shape[0] == len(idx) else v for k, v in res.items()})
    return res

ewmcombine.output = ['data', 'vol', 'rho', 'mult', 'normalized_mult',  'cor', 'vols']

def ewmcombined(tss, wgts, n = 1024, vol_days = None, full = False, join = 'oj', method = None):
    """
    We expect tss and wgts to be provided as lists at the moment

    Parameters
    ----------
    tss : list of timeseries
        the data we want to combine into a single signal
    wgts : list of weights, either floats or timeseries
        
    n : TYPE, optional
        DESCRIPTION. The default is 1024.
    full : TYPE, optional
        DESCRIPTION. The default is False.
    join : TYPE, optional
        DESCRIPTION. The default is 'oj'.
    method : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    res : TYPE
        DESCRIPTION.

    """
    return ewmcombine(tss, wgts, n = n, full = full, join = join, method = method)['data']

    