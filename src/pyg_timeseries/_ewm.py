import numpy as np; import pandas as pd
from pyg_timeseries._math import stdev_calculation_ewm, skew_calculation, cor_calculation_ewm, covariance_calculation, corr_calculation_ewm, LR_calculation_ewm, variance_calculation_ewm, _w
from pyg_timeseries._decorators import compiled, first_, _data_state
from pyg_timeseries._expanding import cumsum
from pyg_base import pd2np, clock, loop_all, loop, is_pd, is_df, presync, df_concat

__all__ = ['ewma', 'ewmstd', 'ewmvar', 'ewmskew', 'ewmrms',  'ewmcor',  'ewmcorr', 'ewmLR', 'ewmGLM',
           'ewma_', 'ewmstd_', 'ewmskew_', 'ewmrms_', 'ewmcor_', 'ewmvar_','ewmLR_', 'ewmGLM_',]


############################################
##
## compiled functions, unfortunately, both these methods are much slower
##
###########################################

# import numba
# from numba import int32, float32    # import the types
# from numba.experimental import jitclass

# spec = [
#     ('t0',  float32),               # a simple scalar field
#     ('t1',  float32),               # a simple scalar field
#     ('t',   float32),               # a simple scalar field
#     ('a',   float32),               # a simple scalar field
#     ('w',   float32),               # a simple scalar field
# ]

# @jitclass(spec)
# class c_ewma(object):
#     def __init__(self, a, t, t0, t1, w):
#         self.a = a
#         self.t = t
#         self.t0 = t0
#         self.t1 = t1
#         self.w = w

#     def push(self, a, t):
#         if np.isnan(a):
#             return np.nan
#         if t == self.t:
#             self.t1 = self.t1 + (1-self.w) * (a - self.a)
#             return self.t1/self.t0
#         else:
#             p = self.w if np.isnan(t) else self.w**(t-self.t)
#             self.t0 = self.t0 * p + (1-self.w)
#             self.t1 = self.t1 * p + (1-self.w) * a
#             return self.t1/self.t0


# @compiled
# def _ewma_(ai, ti, ai0, ti0, t0, t1, w):
#     """
#     we receive 
    
#     - current values, (ai, ti)
#     - previous values (ai0, ti0)
#     - current state of the moments t0, t1
#     - parameters, w
    
#     We return:
#     result, current values, updated moments
#     res, a, t, t0, t1
#     """
#     if np.isnan(ai):
#         res = np.nan
#         return res, ai0, ti0, t0, t1
#     else:
#         if ti == ti0:
#             t1 = t1 + (1-w) * (ai - ai0)
#             res = t1/t0
#             return res, ai, ti, t0, t1 
#         else:
#             p = w**(ti-ti0)
#             t0 = t0 * p + (1-w)
#             t1 = t1 * p + (1-w) * ai
#             res = t1/t0
#             return res, ai, ti, t0, t1

@pd2np
@compiled
def _ewma(a, n, time, t = np.nan, t0 = 0, t1 = 0):
    if n == 1:
        return a, t, t0, t1
    w = _w(n)
    res = np.empty_like(a)
    i0 = 0
    for i in range(a.shape[0]):
        if np.isnan(a[i]):
            res[i] = np.nan
        else:
            if time[i] == t:
                t1 = t1 + (1-w) * (a[i] - a[i0])
            else:
                p = w if np.isnan(time[i]) else w**(time[i]-t)
                t0 = t0 * p + (1-w)
                t1 = t1 * p + (1-w) * a[i]
                t = time[i]
            i0 = i
            res[i] = np.nan if t0 == 0 else t1/t0
    return res, t, t0, t1


@pd2np
@compiled
def _ewmrms(a, n, time, t = np.nan, t0 = 0., t2 = 0., exc_zero = False, max_move = 0):
    if n == 1:
        return a, t, t0, t2
    w = _w(n)
    res = np.empty_like(a)
    apply_max = max_move>0
    i0 = 0; ai0 = a[i0]
    for i in range(a.shape[0]):
        if np.isnan(a[i]):
            res[i] = np.nan
        else:
            if apply_max and not np.isnan(res[i0]):
                ai = max(min(res[i0] * max_move, a[i]), -res[i0] * max_move)
            else:
                ai = a[i]                
            if exc_zero and ai == 0:
                res[i] = res[i0]
            elif time[i] == t:
                t2 = t2 + (1-w) * (ai**2 - ai0**2)
            else:
                p = w if np.isnan(time[i]) else w**(time[i]-t)
                v = ai
                t0 = t0 * p + (1-w)
                t2 = t2 * p + (1-w) * v**2
                t = time[i]
            i0 = i; ai0 = ai
            res[i] = np.nan if t0 == 0 else np.sqrt(t2/t0)
    return res, t, t0, t2



@pd2np
@compiled
def _ewmstd(a, n, time, t = np.nan, t0 = 0, t1 = 0, t2 = 0, w2 = 0, min_sample = 0.25, bias = False, exc_zero = False, max_move = 0, calculator = stdev_calculation_ewm):
    if n == 1:
        return np.full_like(a, 0.0), t, t0, t1, t2, w2
    w = _w(n)
    res = np.empty_like(a)
    apply_max = max_move>0
    i0 = 0; ai0 = a[i0]
    for i in range(a.shape[0]):
        if np.isnan(a[i]):
            res[i] = np.nan
        else:
            if apply_max and not np.isnan(res[i0]):
                ai = max(min(res[i0] * max_move, a[i]), -res[i0] * max_move)
            else:
                ai = a[i]                
            if exc_zero and ai == 0:
                res[i] = res[i0]
            elif time[i] == t:
                t1 = t1 + (1-w) * (ai - ai0)
                t2 = t2 + (1-w) * (ai**2 - ai0**2)
            else:
                p = w if np.isnan(time[i]-t) else w**(time[i]-t)
                v = ai
                t0 = t0 * p + (1-w)
                w2 = w2 * p**2 + (1-w)**2
                t1 = t1 * p + (1-w) * v
                t2 = t2 * p + (1-w) * v**2
                t = time[i]
            i0 = i; ai0 = ai
            res[i] = calculator(t0, t1, t2, w2 = w2, min_sample = min_sample, bias = bias)
    return res, t, t0, t1, t2, w2


@pd2np
@compiled
def _ewmcor(a, b, ba, n, time, t = np.nan, t0 = 0, a1 = 0, a2 = 0, b1 = 0, b2 = 0, ab = 0, w2 = 0, min_sample = 0.25, bias = False):
    """
    _ewmcor(a, b, ba, n, time, t)
    n = 50
    t = np.nan; t0 = 0; a1 = 0; a2 = 0; b1 = 0; b2 = 0; ab = 0; w2 = 0; min_sample = 0.25; bias = False    
    data, t, t0, a1, a2, b1, b2, ab, w2 = _ewmcor(a, b, ba, 200, time, t = np.nan, t0 = 0, a1 = 0, a2 = 0, b1 = 0, b2 = 0, ab = 0, w2 = 0, min_sample = 0.25, bias = False)
    pd.Series(data, drange(-9999)).plot()

    """
    if n == 1:
        return np.full_like(a, np.nan), t, t0, a1, a2, b1, b2, ab, w2
    w = _w(n)
    res = np.empty_like(a)
    i0 = 0
    for i in range(a.shape[0]):
        if np.isnan(a[i]) or np.isnan(b[i]):
            res[i] = np.nan
        else:
            if time[i] == t:
                a1 = a1 + (1-w) * (a[i] - a[i0])
                a2 = a2 + (1-w) * (a[i]**2 - a[i0]**2)
                b1 = b1 + (1-w) * (b[i] - b[i0])
                b2 = b2 + (1-w) * (b[i]**2 - b[i0]**2)
                ab = ab + (1-w) * (ba[i] - ba[i0])
            else:
                p = w if np.isnan(time[i]) else w**(time[i]-t)
                t0 = t0 * p + (1-w)
                w2 = w2 * p**2 + (1-w)**2
                a1 = a1 * p + (1-w) * a[i]
                a2 = a2 * p + (1-w) * a[i]**2
                b1 = b1 * p + (1-w) * b[i]
                b2 = b2 * p + (1-w) * b[i]**2
                ab = ab * p + (1-w) * ba[i]
                t = time[i]
            i0 = i
            res[i0] = cor_calculation_ewm(t0 = t0, a1 = a1, a2 = a2, b1 = b1, b2 = b2, ab = ab, w2 = w2, min_sample = min_sample, bias = bias)
    return res, t, t0, a1, a2, b1, b2, ab, w2


@compiled
def _ewmcorrelation(a, n, a0 = None, a1 = None, a2 = None, aa = None, prev = None, w2 = None, min_sample = 0.25, bias = False, overlapping = 1):
    """
    synchronized correlation calculation
    a is assumed to be total returns as opposed to deltas
    
    For each pair j>k we will store
    prev[j,k] : prev value of j when j,k were non-nan
    prev[k,j] : prev value of k when j,k were non-nan

    a0[j,k] : the total 0th moment of j on the pairing (j,k)
    a1[j,k] : the total 1st moment of j on the pairing (j,k)
    a2[j,k] : the total 2nd moment of j on the pairing (j,k)

    a0[j,k] : the total 0th moment of k on the pairing (j,k)
    a1[j,k] : the total 1st moment of k on the pairing (j,k)
    a2[j,k] : the total 2nd moment of k on the pairing (j,k)
    
    """
    m = a.shape[1]
    if n == 1:
        return np.full((a.shape[0], m, m), np.nan), a0, a1, a2, aa, prev, w2
    p = w = _w(n)
    v = 1 - w
    res = np.full((a.shape[0], m, m), np.nan)
    prev = np.full((m,m,overlapping), np.nan) if prev is None else prev
    a0 = np.zeros((m,m)) if a0 is None else a0
    a1 = np.zeros((m,m)) if a1 is None else a1
    a2 = np.zeros((m,m)) if a2 is None else a2
    aa = np.zeros((m,m)) if aa is None else aa
    w2 = np.zeros((m,m)) if w2 is None else w2
    for i in range(a.shape[0]):
        for j in range(m):
            res[i, j, j] = 1.
        for j in range(m):
            if ~np.isnan(a[i,j]):
                for k in range(j):
                    if ~np.isnan(a[i,k]):
                        if ~np.isnan(prev[j,k,-1]) and ~np.isnan(prev[k,j,-1]):
                            dx = a[i,j] - prev[j,k,-1]
                            dy = a[i,k] - prev[k,j,-1]
                            w2[j,k] = w2[k,j] = w2[j,k] * p**2 + v**2       
                            a0[j,k] = a0[k,j] = a0[j,k] * p + v
                            a1[j,k] = a1[j,k] * p + v * dx
                            a1[k,j] = a1[k,j] * p + v * dy
                            a2[j,k] = a2[j,k] * p + v * dx ** 2        
                            a2[k,j] = a2[k,j] * p + v * dy ** 2                                
                            aa[j,k] = aa[k,j] = aa[j,k] * p + v * dx * dy
                            res[i, k, j] = res[i, j, k] = corr_calculation_ewm(a0 = a0[j,k], a1 = a1[j,k], a2 = a2[j,k], aw2 = w2[j,k], 
                                                                               b0 = a0[k,j], b1 = a1[k,j], b2 = a2[k,j], bw2 = w2[k,j],
                                                                               ab = aa[j,k], ab0 = a0[j,k],
                                                                               min_sample = min_sample, bias = bias)
                        if overlapping > 1:
                            prev[j,k,1:] = prev[j,k,:-1]
                            prev[k,j,1:] = prev[k,j,:-1]
                        prev[j,k,0] = a[i,j]
                        prev[k,j,0] = a[i,k]
                            
    return res, a0, a1, a2, aa, prev, w2




@compiled
def _ewmcovariance(a, n, a0 = None, a1 = None, aa = None, prev = None, min_sample = 0.25, bias = False, overlapping = 1):
    """

    """
    m = a.shape[1]
    if n == 1:
        return np.full((a.shape[0], m, m), np.nan), a0, a1, aa, prev
    p = w = _w(n)
    v = 1 - w
    res = np.full((a.shape[0], m, m), np.nan)
    prev = np.full((m,m,overlapping), np.nan) if prev is None else prev
    a0 = np.zeros((m,m)) if a0 is None else a0
    a1 = np.zeros((m,m)) if a1 is None else a1
    aa = np.zeros((m,m)) if aa is None else aa
    for i in range(a.shape[0]):
        for j in range(m):
            if ~np.isnan(a[i,j]):
                for k in range(j+1):
                    if ~np.isnan(a[i,k]):
                        if ~np.isnan(prev[j,k,-1]) and ~np.isnan(prev[k,j,-1]):
                            dx = a[i,j] - prev[j,k,-1]
                            dy = a[i,k] - prev[k,j,-1]
                            a0[j,k] = a0[k,j] = a0[j,k] * p + v
                            a1[j,k] = a1[j,k] * p + v * dx
                            a1[k,j] = a1[k,j] * p + v * dy
                            aa[j,k] = aa[k,j] = aa[j,k] * p + v * dx * dy
                            res[i, k, j] = res[i, j, k] = covariance_calculation(a0 = a0[j,k], a1 = a1[j,k], 
                                                                           b0 = a0[k,j], b1 = a1[k,j],
                                                                           ab = aa[j,k], ab0 = a0[j,k],
                                                                           min_sample = min_sample, bias = bias)                    
                        if overlapping > 1:
                            prev[j,k,1:] = prev[j,k,:-1]
                            prev[k,j,1:] = prev[k,j,:-1]
                        prev[j,k,0] = a[i,j]
                        prev[k,j,0] = a[i,k]
    return res, a0, a1, aa, prev



def ewmcovariance_(a, n, min_sample = 0.25, bias = False, overlapping = 1, instate = None, join = 'outer', method = None):
    """
    This calculates a full correlation matrix as a timeseries. Also returns the recent state of the calculations.
    
    :Returns:
    ---------
        a dict with:
            - data: t x n x n covariance matrix
            - index: timeseries index
            - columns: columns of original data

    See ewmcorr for full details.
    
    """
    state = {} if instate is None else instate
    arr = df_concat(a, join = join, method = method)
    if isinstance(arr, np.ndarray):
        res, a0, a1, aa, prev = _ewmcovariance(arr, n, min_sample = min_sample, bias = bias, overlapping = overlapping, **state)
        # if res.shape[1] == 2:
        #     res = res[:, 0, 1]
        return dict(data = res, columns = None, index = None, state = dict(a0=a0, a1=a1, aa=aa, prev = prev))
    elif is_df(arr):
        index = arr.index
        columns = list(arr.columns)
        res, a0, a1, aa, prev = _ewmcovariance(arr.values, n, min_sample = min_sample, overlapping = overlapping, bias = bias, **state)
        state = dict(a0=a0, a1=a1, aa=aa, prev = prev)
        return dict(data = res, columns = columns, index = index, state = state)
    else:
        raise ValueError('unsure how to calculate correlation matrix for a %s'%a)

ewmcovariance_.output = ['data', 'columns', 'index', 'state']


def ewmcovariance(a, n, min_sample = 0.25, bias = False, overlapping = 1, instate = None, join = 'outer', method = None):
    """
    This calculates a full covariance matrix as a timeseries. 

    :Parameters:
    ----------
    a : np.array or a pd.DataFrame
        multi-variable timeseries to calculate correlation for
    n : int
        days for which rolling correlation is calculated.
    min_sample : float, optional
        Minimum observations needed before we calculate correlation. The default is 0.25.
    bias : bool, optional
        input to stdev calculations, the default is False.
    instate : dict, optional
        historical calculations so far.

    :Returns:
    -------
    covariance (as t x n x n np.array)
        
        
    :Example: a pair of ts
    ---------
    >>> a = pd.DataFrame(np.random.normal(0,1,(10000,10)), drange(-9999))
    >>> cov1_ = ewmcovariance_(cumsum(a), 250)
    >>> cov2_ = ewmcovar_(a, 250)
    >>> cov1 = cov1_['data']
    >>> cov2 = cov2_['data']
    >>> vol1 = pd.DataFrame([[row[j,j] for j in range(10)] for row in cov1], a.index) ** 0.5
    >>> vol2 = pd.DataFrame([[row[j,j] for j in range(10)] for row in cov2], a.index) ** 0.5
    >>> vol1.plot()
    >>> vol2.plot()

    
    >>> # We first check that diagonal is indeed the (biased) variance of the variables: 
    >>> ratio = pd.Series([cov1[-1][i,i] for i in range(10)]) / ewmvar(a, 250, bias=True).iloc[-1]
    >>> assert ratio.max() < 1.0001 and ratio.min() > 0.9999
    
    To access individually, here we calculate the correlation between 0th and 1st timeseries. That correlation is close to 0 (Fisher distribution) so...
    
    >>> cor = pd.Series(cov1[:,0,1] / np.sqrt(cov1[:,0,0] * cov1[:,1,1]), a.index)
    >>> cor.plot()
    >>> assert cor.max() < 0.3 and cor.min() > -0.3
    """
    return ewmcovariance_(a, n, min_sample = min_sample, bias = bias, overlapping = overlapping, instate = instate , join = join, method = method).get('data')



def ewmcovar_(a, n, min_sample = 0.25, bias = False, overlapping = 1, instate = None, join = 'outer', method = None):
    """
    This calculates a full covariance matrix as a timeseries. Also returns the recent state of the calculations.
    
    :Returns:
    ---------
        a dict with:
            - data: t x n x n covariance matrix
            - index: timeseries index
            - columns: columns of original data

    See ewmcovar for full details.
    
    """
    state = {} if instate is None else instate
    arr = df_concat(a, join = join, method = method)
    if overlapping == 1:
        state['prev'] = state.pop('prev', np.zeros((arr.shape[1], arr.shape[1], 1)))
    return ewmcovariance_(cumsum(arr), n = n, min_sample=min_sample, overlapping = overlapping, bias = bias, instate = state, join = join, method = method)   

ewmcovar_.output = ['data', 'columns', 'index', 'state']


def ewmcovar(a, n, min_sample = 0.25, bias = False, overlapping = 1, instate = None, join = 'outer', method = None):
    """
    This calculates a full covariance matrix as a timeseries. 

    :Parameters:
    ----------
    a : np.array or a pd.DataFrame
        multi-variable timeseries to calculate correlation for
    n : int
        days for which rolling correlation is calculated.
    min_sample : float, optional
        Minimum observations needed before we calculate correlation. The default is 0.25.
    bias : bool, optional
        input to stdev calculations, the default is False.
    instate : dict, optional
        historical calculations so far.

    :Returns:
    -------
    covariance (as t x n x n np.array)
        
        
    :Example: a pair of ts
    ---------
    >>> a = pd.DataFrame(np.random.normal(0,1,(10000,10)), drange(-9999))
    >>> res = ewmcovar(a, 250)
    
    >>> # We first check that diagonal is indeed the (biased) variance of the variables: 
    >>> ratio = pd.Series([res[-1][i,i] for i in range(10)]) / ewmvar(a, 250, bias=True).iloc[-1]
    >>> assert ratio.max() < 1.02 and ratio.min() > 0.98
    
    To access individually, here we calculate the correlation between 0th and 1st timeseries. That correlation is close to 0 (Fisher distribution) so...
    
    >>> cov = pd.Series(res[:,0,1] / np.sqrt(res[:,0,0] * res[:,1,1]), a.index)
    >>> cov.plot()
    >>> assert cov.max() < 0.3 and cor.min() > -0.3
    """
    return ewmcovar_(a, n, min_sample = min_sample, bias = bias, instate = instate , join = join, method = method).get('data')


def ewmcorr_(a, n, min_sample = 0.25, bias = False, overlapping = 1, instate = None, join = 'outer', method = None):
    """
    This calculates a full correlation matrix as a timeseries. Also returns the recent state of the calculations.
    See ewmcorr for full details.
    
    """
    state = {} if instate is None else instate
    arr = df_concat(a, join = join, method = method)
    if overlapping == 1:
        state['prev'] = state.pop('prev', np.zeros((arr.shape[1], arr.shape[1], 1)))
    return ewmcorrelation_(cumsum(arr), n = n, min_sample=min_sample, bias = bias, overlapping = overlapping, instate = state, join = join, method = method)   

ewmcorr_.output = ['data', 'state', 'index', 'columns']


def ewmcorr(a, n, min_sample = 0.25, bias = False, overlapping = 1, instate = None, join = 'outer', method = None):
    """
    This calculates a full correlation matrix as a timeseries. 

    :Parameters:
    ----------
    a : np.array or a pd.DataFrame
        multi-variable timeseries to calculate correlation for
    n : int
        days for which rolling correlation is calculated.
    min_sample : float, optional
        Minimum observations needed before we calculate correlation. The default is 0.25.
    bias : bool, optional
        input to stdev calculations, the default is False.
    instate : dict, optional
        historical calculations so far.

    :Returns:
    -------
    correlation dataset
        an xarray.Dataset unless there are only two timeseries in a, in which case, a single column correlation timeseries is returned
        
        
    :Example: a pair of ts
    ---------
    >>> rtn = np.random.normal(0,1,10000)
    >>> adj = cumsum(rtn)
    >>> x0 = ewmacd(adj, 10, 20, 30)[50:]
    >>> x1 = ewmacd(adj, 20, 40, 30)[50:]
    >>> a = pd.DataFrame(np.array([x0,x1]).T, drange(-9949))
    >>> res = ewmcorr(a, n)
    >>> ts = pd.Series(res[:,0,1], a.index)

    >>> ts    
    >>>     Out[130]: 
    >>>     1994-06-07         NaN
    >>>     1994-06-08         NaN
    >>>     1994-06-09         NaN
    >>>     1994-06-10         NaN
    >>>     1994-06-11         NaN
      
    >>>     2021-08-29    0.890766
    >>>     2021-08-30    0.886926
    >>>     2021-08-31    0.883054
    >>>     2021-09-01    0.879577
    >>>     2021-09-02    0.875766
    >>>     Length: 9950, dtype: float64    

    :Example: multi column ts
    ---------
    >>> rtn = np.random.normal(0,1,10000)
    >>> adj = cumsum(rtn)
    >>> x0 = ewmacd(adj, 10, 20, 30)[50:]
    >>> x1 = ewmacd(adj, 20, 40, 30)[50:]
    >>> x2 = ewmacd(adj, 40, 80, 30)[50:]
    >>> a = pd.DataFrame(np.array([x0,x1,x2]).T, drange(-9949), ['a','b','c'])
    >>> ds = ewmcorr_(a, n)
    >>> ds.keys()
    >>> dict_keys(['data', 'columns', 'index', 'state'])
    
    >>> ds['data'].shape
    >>> (9950, 3, 3)

    To access individual correlations:
    
    >>> a_vs_b = pd.Series(ds['data'][:, 1, 2], ds['index'])
    
    To access all correlations to a:    
    
    >>> a_vs_all = pd.DataFrame(ds['data'][:,:,0], ds['index'], ds['columns'])
    >>> a_vs_all.plot()
    >>> a_vs_all
    
                  a         b         c
    1997-12-13  1.0       NaN       NaN
    1997-12-14  1.0       NaN       NaN
    1997-12-15  1.0       NaN       NaN
    1997-12-16  1.0       NaN       NaN
    1997-12-17  1.0       NaN       NaN
            ...       ...       ...
    2025-03-06  1.0  0.895512  0.778449
    2025-03-07  1.0  0.884339  0.753600
    2025-03-08  1.0  0.871431  0.726200
    2025-03-09  1.0  0.858903  0.699254
    2025-03-10  1.0  0.846466  0.673574
    
    """
    return ewmcorr_(a, n, min_sample = min_sample, bias = bias, overlapping = overlapping, instate = instate, join = join, method = method).get('data')

def ewmcorrelation_(a, n, min_sample = 0.25, bias = False, overlapping = 1, instate = None, join = 'outer', method = None):
    """
    This calculates a full correlation matrix as a timeseries. Also returns the recent state of the calculations.
    See ewmcorrelation for full details.
    
    """
    state = {} if instate is None else instate
    arr = df_concat(a, join = join, method = method)
    if isinstance(arr, np.ndarray):
        res, a0, a1, a2, aa, prev, w2 = _ewmcorrelation(arr, n, min_sample = min_sample, bias = bias, overlapping = overlapping, **state)
        return dict(data = res, index = None, columns = None, state = dict(a0=a0, a1=a1, a2=a2, aa=aa, prev = prev, w2 = w2))
    elif is_df(arr):
        index = arr.index
        columns = list(arr.columns)
        res, a0, a1, a2, aa, prev, w2 = _ewmcorrelation(arr.values, n, min_sample = min_sample, bias = bias, overlapping = overlapping, **state)
        state = dict(a0=a0, a1=a1, a2=a2, aa=aa, prev=prev, w2 = w2)
        return dict(data = res, columns = columns, index = index, state = state)
    else:
        raise ValueError('unsure how to calculate correlation matrix for a %s'%a)

ewmcorrelation_.output = ['data', 'state', 'index', 'columns']


def ewmcorrelation(a, n, min_sample = 0.25, bias = False, overlapping = 1, instate = None, join = 'outer', method = None):
    """
    This calculates a full correlation matrix as a timeseries. 

    :Parameters:
    ----------
    a : np.array or a pd.DataFrame
        multi-variable timeseries of aggregate returns to calculate correlation for
    n : int
        days for which rolling correlation is calculated.
    min_sample : float, optional
        Minimum observations needed before we calculate correlation. The default is 0.25.
    bias : bool, optional
        input to stdev calculations, the default is False.
    instate : dict, optional
        historical calculations so far.

    :Returns:
    -------
    correlation dataset
        an xarray.Dataset unless there are only two timeseries in a, in which case, a single column correlation timeseries is returned
        
        
    :Example: multi column ts
    ---------
    >>> join = 'outer'; method = None
    >>> rtn = np.random.normal(0,1,10000)
    >>> adj = cumsum(rtn)
    >>> x0 = ewmacd(adj, 10, 20, vol = 18)
    >>> x1 = ewmacd(adj, 20, 40, vol = 18)
    >>> x2 = ewmacd(adj, 40, 80, vol = 18)
    >>> x0[np.random.normal(0,1,10000)>1] = np.nan
    >>> x1[np.random.normal(0,1,10000)>1] = np.nan
    >>> x2[np.random.normal(0,1,10000)>1] = np.nan
    >>> a = pd.DataFrame(np.array([x0,x1,x2]).T, drange(-9999), ['a','b','c'])
    >>> ds = ewmcorrelation(a, 30)
    >>> ds2 = ewmcorrelation(a, 30, overlapping = 5)
    >>> c = ewmcorr(diff(a), 30)
    >>> ds.shape
    >>> (10000, 3, 3)

    To access individual correlations:
    
    >>> a_vs_b = pd.Series(ds[:, 1, 2], a.index)
    >>> a_vs_b5 = pd.Series(ds2[:, 1, 2], a.index)
    >>> df_concat([a_vs_b, a_vs_b5], ['1 day', '5 day overlapping']).plot() 
    
    
    To access all correlations to a:
    >>> a_vs_all = pd.DataFrame(ds[:,:,0], a.index, a.columns)
    >>> a_vs_all.ffill().plot()
        
    """
    return ewmcorrelation_(a, n, min_sample = min_sample, bias = bias, overlapping = overlapping, instate = instate , join = join, method = method).get('data')


@pd2np
@compiled
def _ewmLR(a, b, ba, n, time, t = np.nan, t0 = 0, a1 = 0, a2 = 0, b1 = 0, b2 = 0, ab = 0, w2 = 0, min_sample = 0.25, bias = False):
    """
    We have a and b for which we want to fit:
        b_i = c + m a_i 
    
    minimize LSE under weights w_i. We let:
        w = \sum_i w_i
    
    LSE(c,m) = \sum w_i (c + m a_i - b_i)^2
    dLSE/dc  = 0  <==> \sum w_i  (c + m a_i - b_i) = 0    [1]
    dLSE/dm  = 0 <==> \sum w_i  a_i (c + m a_i - b_i) = 0 [2]

    c     + mE(a)    = E(b)     [1]
    cE(a) + mE(a^2)  = E(ab)    [2]
    
    cE(a) + mE(a)^2  = E(a)E(n) [1] * E(a) 
    m(E(a^2) - E(a)^2) = E(ab) - E(a)E(b)
    
    m = covar(a,b)/var(a)
    c = E(b) - mE(a)
    

    :Example:
    -----------
    a = np.random.normal(0,1,10000)
    b = 0.5 * a + np.random.normal(0,1,10000)
    _ewmLRt(a,b,100)
    """
    w = _w(n)
    m = np.empty_like(a)
    c = np.empty_like(a)
    i0 = 0
    for i in range(a.shape[0]):
        if np.isnan(a[i]) or np.isnan(b[i]):
            c[i] = m[i] = np.nan
        else:
            if time[i] == t:
                a1 = a1 + (1-w) * (a[i] - a[i0])
                a2 = a2 + (1-w) * (a[i]**2 - a[i0]**2)
                b1 = b1 + (1-w) * (b[i] - b[i0])
                b2 = b2 + (1-w) * (b[i]**2 - b[i0]**2)
                ab = ab + (1-w) * (ba[i] - ba[i0])
            else:
                p = w if np.isnan(time[i]) else w**(time[i]-t)
                t0 = t0 * p + (1-w)
                w2 = w2 * p**2 + (1-w)**2
                a1 = a1 * p + (1-w) * a[i]
                a2 = a2 * p + (1-w) * a[i]**2
                b1 = b1 * p + (1-w) * b[i]
                b2 = b2 * p + (1-w) * b[i]**2
                ab = ab * p + (1-w) * ba[i]
                t = time[i]
            i0 = i
            c[i0], m[i0] = LR_calculation_ewm(t0 = t0, a1 = a1, a2 = a2, b1 = b1, b2 = b2, ab = ab, w2 = w2, min_sample = min_sample, bias = bias)
    return c, m, t, t0, a1, a2, b1, b2, ab, w2


@compiled
def xTx(x):
    res = np.empty((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        for j in range(i):
            res[i][j] = res[j][i] = x[i] * x[j]
        res[i][i] = x[i]**2
    return res


@compiled
def _ewmGLM(a, b, n, time, t, t0, a2, ab, min_weight = 0.25):
    """
    We assume b is single column while a is multicolumn. We are fitting
    
    b[i] =\sum_j m_j a_j[i]
    
    LSE(m) = \sum_i w_i (b[i] - \sum_j m_j * a_j[i])^2
    dLSE/dm_k = 0  
    <==>  \sum_i w_i (b[i] - \sum_j m_j * a_j[i]) a_k[i] = 0
    <==>  E(b*a_k) = m_k E(a_k^2) + sum_{j<>k} m_k E(a_j a_k) 
    
    E is expectation under w. 
    
    We set
    a2[i,j] = E(a_i a_j)
    ab[j] = E(a_j * b)
    
    And invert
    A*m = B
    
    Example
    --------
    >>> from pyg import *
    >>> a = np.random.normal(0,1,(5000,10))
    >>> m = np.random.normal(1,1,10)
    >>> a[a>2] = np.nan
    >>> b = np.sum(a * m,axis=1) + np.random.normal(0,1,5000)
    >>> n = 30
    >>> t = np.nan; t0 = 0; time = np.full_like(b, np.nan); a2 = np.zeros((a.shape[1],a.shape[1])); ab = np.zeros(a.shape[1])
    >>> res = _ewmGLM(a, b, n, time, t, t0, a2, ab)
    >>> pd.DataFrame(res[0], drange(-4999)).plot()
    >>> pd.Series(m)
    """
    w = _w(n)
    res = np.empty_like(a)
    nana = np.sum(np.isnan(a), axis = 1)
    i0 = 0
    for i in range(a.shape[0]):
        if nana[i]>0 or np.isnan(b[i]):
            res[i] = np.nan
        else:
            if time[i] == t:
                ab = ab + (1-w) * (b[i]*a[i] - b[i0]*a[i0])
                a2 = a2 + (1-w) * xTx(a[i]) - (1-w) * xTx(a[i0])
            else:
                p = w if np.isnan(time[i]) else w**(time[i]-t)
                t0 = t0 * p + (1-w)
                ab = ab * p + (1-w) * b[i]*a[i]
                a2 = a2 * p + (1-w) * xTx(a[i])
                t = time[i]                
            i0 = i
            if t0>min_weight:
                a2i = np.linalg.inv(a2/t0)
                res[i] = a2i.dot(ab/t0)
            else:
                res[i] = np.nan
    return res, t, t0, a2, ab
    

@pd2np
@compiled
def _ewmskew(a, n, time, bias = 0, t = np.nan, t0 = 0, t1 = 0, t2 = 0, t3 = 0, min_sample = 0.25):
    w = _w(n)
    days = n if n>1 else w/(1-w)
    d = 1 + days
    res = np.empty_like(a)
    i0 = 0
    for i in range(a.shape[0]):
        if np.isnan(a[i]):
            res[i] = np.nan
        else:
            if time[i] == t:
                t1 = t1 + (1-w) * (a[i] - a[i0])
                t2 = t2 + (1-w) * (a[i]**2 - a[i0]**2)
                t3 = t3 + (1-w) * (a[i]**3 - a[i0]**3)
            else:
                p = w if np.isnan(time[i]) else w**(time[i]-t)
                v = a[i]
                t0 = t0 * p + (1-w)
                t1 = t1 * p + (1-w) * v
                t2 = t2 * p + (1-w) * v**2
                t3 = t3 * p + (1-w) * v**3
                t = time[i]
            i0 = i
            res[i] = skew_calculation(t0 * d, t1 * d, t2 * d, t3 * d, bias, min_sample)
    return res, t, t0, t1, t2, t3

##### clock management functions  ############################################

@loop_all
def _ewmat(a, n, time = None, t = None, t0 = 0, t1 = 0):
    time = clock(a, time, t)
    t = 0 if t is None or np.isnan(t) else t
    return _ewma(a = a, n = n, time = time, t = t, t0 = t0, t1 = t1)

@loop_all
def _ewmrmst(a, n, time = None, t = None, t0 = 0, t2 = 0, exc_zero = False, max_move = 0):
    time = clock(a, time, t)
    t = 0 if t is None or np.isnan(t) else t
    return _ewmrms(a, n, time = time, t = t, t0 = t0, t2 = t2, exc_zero = exc_zero, max_move = max_move)

@loop_all
def _ewmstdt(a, n, time = None, t = None, t0 = 0, t1 = 0, t2 = 0, w2 = 0, min_sample = 0.25, bias = False, exc_zero = False, max_move = 0, calculator = stdev_calculation_ewm):
    time = clock(a, time, t)
    t = 0 if t is None or np.isnan(t) else t
    return _ewmstd(a, n, time = time, t = t, t0 = t0, t1 = t1, t2 = t2, w2 = w2, min_sample=min_sample, bias = bias, calculator = calculator, exc_zero = exc_zero, max_move = max_move)



@loop_all
@presync
def _ewmcort(a, b, n, time = None, t = None, t0 = 0, a1 = 0, a2 = 0, b1 = 0, b2 = 0, ab = 0, w2 = 0, min_sample = 0.25, bias = False):
    ba = b * a
    time = clock(ba, time, t)
    t = 0 if t is None or np.isnan(t) else t
    return _ewmcor(a = a, b = b, ba = ba, n = n, time = time, t = t, t0 = t0, a1 = a1, a2 = a2, b1 = b1, b2 = b2, ab = ab, w2 = w2, min_sample=min_sample, bias = bias)

@loop_all
@presync
def _ewmLRt(a, b, n, time = None, t = None, t0 = 0, a1 = 0, a2 = 0, b1 = 0, b2 = 0, ab = 0, w2 = 0, min_sample = 0.25, bias = False):
    ba = b * a
    time = clock(ba, time, t)
    t = 0 if t is None or np.isnan(t) else t
    return _ewmLR(a = a, b = b, ba = ba, n = n, time = time, t = t, t0 = t0, a1 = a1, a2 = a2, b1 = b1, b2 = b2, ab = ab, w2 = w2, min_sample=min_sample, bias = bias)

@loop(list, dict)
@presync(columns = False)
def _ewmGLMt(a, b, n, time = None, t = None, t0 = 0, a2 = None, ab = None, min_sample = 0.25):
    """
    >>> from pyg import *
    >>> a = np.random.normal(0,1,(5000,10))
    >>> m = np.random.normal(1,1,10)
    >>> b = np.sum(a * m,axis=1) + np.random.normal(0,1,5000)
    >>> a[a>2] = np.nan
    >>> n = 30
    >>> time = None; t = None; t0 = 0; a2 = None; ab = None; min_sample = 0.25
    >>> res = ewmGLM_(a, b, n)
    
    """
    time = clock(b, time, t)
    t = 0 if t is None or np.isnan(t) else t
    a2 = np.zeros((a.shape[1], a.shape[1])) if a2 is None else a2
    ab = np.zeros(a.shape[1]) if ab is None else ab
    a_ = a.values if is_pd(a) else a  #np.array(list(a.values)) 
    b_ = b.values if is_pd(b) else b
    res = _ewmGLM(a = a_, b = b_, n = n, time = time, t = t, t0 = t0, a2 = a2, ab = ab, min_weight = min_sample)
    if is_pd(a):
         res = (pd.DataFrame(res[0], a.index, columns = a.columns),) + res[1:]
    return res


@loop_all
def _ewmskewt(a, n, time = None, t = None, bias = False, t0 = 0, t1 = 0, t2 = 0, t3 = 0, min_sample = 0.25):
    time = clock(a, time, t)
    t = 0 if t is None or np.isnan(t) else t
    return _ewmskew(a, n, time = time, bias = int(bias), t = t, t0 = t0, t1 = t1, t2 = t2, t3 = t3, min_sample=min_sample)    


#####   exposed API functions ############################################

def ewma_(a, n, time = None, data = None, instate = None):
    """
    Equivalent to ewma but returns a state parameter for instantiation of later calculations.
    See ewma documentation for more details
    """
    state = instate or {}
    return _data_state(['data', 't', 't0', 't1'], _ewmat(a = a, n = n, time = time, **state))

ewma_.output = ['data', 'state']


def ewma(a, n, time = None, axis = 0, data = None, state = None):
    """
    ewma is equivalent to a.ewm(n).mean() but with...
    - supports np.ndarrays as well as timeseries
    - handles nan by skipping them
    - allows state-management
    - ability to supply a 'clock' to the calculation
    
    :Parameters:
    ------------
    a : array/timeseries
    n : int/fraction
        The number or days (or a ratio) to scale the history
    time : Calendar, 'b/d/y/m' or a timeseries of time (use clock(a) to see output)
        If time parameter is provided, we allow multiple observations per unit of time. i.e., converging to the last observation in time unit. 
            - if we have intraday data, and set time = 'd', then 
            - the ewm calculation on last observations per day is what is retained. 
            - the ewm calculation on each intraday observation is same as an ewm(past EOD + current intraday observation)
    
    data: None.
        unused at the moment. Allow code such as func(live, **func_(history)) to work
    state: dict, optional
        state parameters used to instantiate the internal calculations, based on history prior to 'a' provided. 
    
    :Example: matching pandas
    -------------------------
    >>> import pandas as pd; import numpy as np; from pyg import *
    >>> a = pd.Series(np.random.normal(0,1,10000), drange(-9999))
    >>> ts = ewma(a,10); df = a.ewm(10).mean()
    >>> assert abs(ts-df).max()<1e-10

    :Example: numpy arrays support
    ------------------------------
    >>> assert eq(ewma(a.values, 10), ewma(a,10).values)

    :Example: nan handling
    ----------------------
    >>> a[a.values<0.1] = np.nan
    >>> ts = ewma(a,10, time = 'i'); df = a.ewm(10).mean() # note: pandas assumes, 'time' pass per index entry, even if value is nan
    >>> assert abs(ts-df).max()<1e-10
    
    >>> pd.concat([ts,df], axis=1)
    >>>                        0         1
    >>> 1993-09-24  0.263875  0.263875
    >>> 1993-09-25       NaN  0.263875
    >>> 1993-09-26       NaN  0.263875
    >>> 1993-09-27       NaN  0.263875
    >>> 1993-09-28       NaN  0.263875
    >>>                  ...       ...
    >>> 2021-02-04       NaN  0.786506
    >>> 2021-02-05  0.928817  0.928817
    >>> 2021-02-06       NaN  0.928817
    >>> 2021-02-07  0.839168  0.839168
    >>> 2021-02-08  0.831109  0.831109

    :Example: state management
    --------------------------
    >>> old = a.iloc[:5000]
    >>> new = a.iloc[5000:]
    >>> old_ts = ewma_(old, 10)
    >>> new_ts = ewma(new, 10, **old_ts) # instantiation with previous ewma
    >>> ts = ewma(a,10)
    >>> assert eq(new_ts, ts.iloc[5000:])

    :Example: Support for time & clock
    ------------------------------------
    >>> daily = a
    >>> monthly = daily.resample('M').last()
    >>> m_ts = ewma(monthly, 3) ## 3-month ewma run on monthly data
    >>> d_ts = ewma(daily, 3, 'm') ## 3-month ewma run on daily data
    >>> daily_resampled_to_month = d_ts.resample('M').last()
    >>> assert abs(daily_resampled_to_month - m_ts).max() < 1e-10

    So you can run a 3-monthly ewma on daily, where within month, most recent value is used with the EOM history.
    
    :Example: Support for dict/list of arrays
    --------------------------------------
    >>> x = pd.Series(np.random.normal(0,1,1000), drange(-999)); y = pd.Series(np.random.normal(0,1,1000), drange(-999))
    >>> a = dict(x = x, y = y)    
    >>> assert eq(ewma(dict(x=x, y=y),10), dict(x=ewma(x,10), y=ewma(y,10)))
    >>> assert eq(ewma([x,y],10), [ewma(x,10), ewma(y,10)])
    
             
    :Returns:
    -------
    an array/timeseries of ewma
    
    """
    state = state or {}
    return first_(_ewmat(a=a, n=n, time = time, axis=axis, **state))


def ewmrms_(a, n, time = None, axis = 0, data = None, exc_zero = False, max_move = 0, instate = None):
    """
    Equivalent to ewmrms but returns a state parameter for instantiation of later calculations.
    See ewmrms documentation for more details
    """
    state = instate or {}
    return _data_state(['data', 't', 't0', 't2'],_ewmrmst(a, n, time = time, axis=axis, exc_zero = exc_zero, max_move = max_move,**state))

ewmrms_.output = ['data', 'state']

def ewmrms(a, n, time=None, axis=0, data = None, exc_zero = False, max_move = 0, state = None):
    """
    ewmrms is equivalent to (a**2).ewm(n).mean()**0.5  but with...
    - supports np.ndarrays as well as timeseries
    - handles nan by skipping them
    - allows state-management
    - ability to supply a 'clock' to the calculation
    
    :Parameters:
    ------------
    a : array/timeseries
    n : int/fraction
        The number or days (or a ratio) to scale the history
    time : Calendar, 'b/d/y/m' or a timeseries of time (use clock(a) to see output)
        If time parameter is provided, we allow multiple observations per unit of time. i.e., converging to the last observation in time unit. 
            - if we have intraday data, and set time = 'd', then 
            - the ewm calculation on last observations per day is what is retained. 
            - the ewm calculation on each intraday observation is same as an ewm(past EOD + current intraday observation)
    
    exc_zero: bool
        if True, will skip zeros inside a and will assume these are a result of forward filling
    max_move: float
        if positive, will "clip" a[i] at current stdev estimation * max_move
    data: None.
        unused at the moment. Allow code such as func(live, **func_(history)) to work
    state: dict, optional
        state parameters used to instantiate the internal calculations, based on history prior to 'a' provided. 
    
    :Example: matching pandas
    -------------------------
    >>> import pandas as pd; import numpy as np; from pyg import *
    >>> a = pd.Series(np.random.normal(0,1,10000), drange(-9999))
    >>> ts = ewmrms(a,10); df = (a**2).ewm(10).mean()**0.5
    >>> assert abs(ts-df).max()<1e-10

    :Example: numpy arrays support
    ------------------------------
    >>> assert eq(ewmrms(a.values, 10), ewmrms(a,10).values)

    :Example: nan handling
    ----------------------
    >>> a[a.values<0.1] = np.nan
    >>> ts = ewmrms(a,10, time = 'i'); df = (a**2).ewm(10).mean()**0.5 # note: pandas assumes, 'time' pass per index entry, even if value is nan
    >>> assert abs(ts-df).max()<1e-10
    
    >>> pd.concat([ts,df], axis=1)
    >>>                        0         1
    >>> 1993-09-24  0.263875  0.263875
    >>> 1993-09-25       NaN  0.263875
    >>> 1993-09-26       NaN  0.263875
    >>> 1993-09-27       NaN  0.263875
    >>> 1993-09-28       NaN  0.263875
    >>>                  ...       ...
    >>> 2021-02-04       NaN  0.786506
    >>> 2021-02-05  0.928817  0.928817
    >>> 2021-02-06       NaN  0.928817
    >>> 2021-02-07  0.839168  0.839168
    >>> 2021-02-08  0.831109  0.831109
        
    :Example: exc_zero
    ------------------
    >>> import pandas as pd; import numpy as np; from pyg import *
    >>> a = pd.Series(np.random.normal(0,1,10000), drange(-9999))
    >>> a[dt(-10):] = 0
    >>> exclude = ewmrms(a, 10, exc_zero = True)
    >>> assert exclude .iloc[-1] == exclude.iloc[-10]
    >>> no_exclude = ewmrms(a, 10)
    >>> assert no_exclude.iloc[-1] < no_exclude.iloc[-10]

    :Example: max_move
    ------------------
    >>> import pandas as pd; import numpy as np; from pyg import *
    >>> a = pd.Series(np.random.normal(0,1,10000), drange(-9999)) ## a series with stdev of 1 approximately
    >>> a.iloc[-1] = 100. ## put a bad data point with a huge spike...
    >>> max_move = ewmrms(a, 10, max_move = 4)
    >>> assert max_move.iloc[-1] < 2
    >>> no_max_move = ewmrms(a, 10)
    >>> assert no_max_move.iloc[-1] > 20

    :Example: state management
    --------------------------
    >>> old = a.iloc[:5000]
    >>> new = a.iloc[5000:]
    >>> old_ts = ewmrms_(old, 10)
    >>> new_ts = ewmrms(new, 10, **old_ts) # instantiation with previous ewma
    >>> ts = ewmrms(a,10)
    >>> assert eq(new_ts, ts.iloc[5000:])

    :Example: Support for time & clock
    ------------------------------------
    >>> daily = a
    >>> monthly = daily.resample('M').last()
    >>> m_ts = ewmrms(monthly, 3) ## 3-month ewma run on monthly data
    >>> d_ts = ewmrms(daily, 3, 'm') ## 3-month ewma run on daily data
    >>> daily_resampled_to_month = d_ts.resample('M').last()
    >>> assert abs(daily_resampled_to_month - m_ts).max() < 1e-10

    So you can run a 3-monthly ewma on daily, where within month, most recent value is used with the EOM history.
    
    :Example: Support for dict/list of arrays
    --------------------------------------
    >>> x = pd.Series(np.random.normal(0,1,1000), drange(-999)); y = pd.Series(np.random.normal(0,1,1000), drange(-999))
    >>> a = dict(x = x, y = y)    
    >>> assert eq(ewmrms(dict(x=x, y=y),10), dict(x=ewmrms(x,10), y=ewmrms(y,10)))
    >>> assert eq(ewmrms([x,y],10), [ewmrms(x,10), ewmrms(y,10)])
    
             
    :Returns:
    -------
    an array/timeseries of ewma
    
    """
    state = state or {}
    return first_(_ewmrmst(a, n, time = time, axis=axis, exc_zero = exc_zero, max_move = max_move, **state))

def ewmstd_(a, n, time = None, min_sample=0.25, bias = False, axis=0, data = None, exc_zero = False, max_move = 0, instate = None):
    """
    Equivalent to ewmstd but returns a state parameter for instantiation of later calculations.
    See ewmstd documentation for more details
    """
    state = instate or {}
    return _data_state(['data', 't', 't0', 't1', 't2', 'w2'],_ewmstdt(a, n, time = time, min_sample=min_sample, axis=axis, exc_zero = exc_zero, max_move = max_move, calculator = stdev_calculation_ewm, **state))

ewmstd_.output = ['data', 'state']

def ewmstd(a, n, time = None, min_sample=0.25, bias = False, axis=0, data = None, exc_zero = False, max_move = 0, state = None):
    """
    ewmstd is equivalent to a.ewm(n).std() but with...
    - supports np.ndarrays as well as timeseries
    - handles nan by skipping them
    - allows state-management
    - ability to supply a 'clock' to the calculation
    
    :Parameters:
    ------------
    a : array/timeseries
    n : int/fraction
        The number or days (or a ratio) to scale the history
    time : Calendar, 'b/d/y/m' or a timeseries of time (use clock(a) to see output)
        If time parameter is provided, we allow multiple observations per unit of time. i.e., converging to the last observation in time unit. 
            - if we have intraday data, and set time = 'd', then 
            - the ewm calculation on last observations per day is what is retained. 
            - the ewm calculation on each intraday observation is same as an ewm(past EOD + current intraday observation)
    
    exc_zero: bool
        if True, will skip zeros inside a and will assume these are a result of forward filling
    max_move: float
        if positive, will "clip" a[i] at current stdev estimation * max_move
    data: None.
        unused at the moment. Allow code such as func(live, **func_(history)) to work
    state: dict, optional
        state parameters used to instantiate the internal calculations, based on history prior to 'a' provided. 
    
    :Example: matching pandas
    -------------------------
    >>> import pandas as pd; import numpy as np; from pyg import *
    >>> a = pd.Series(np.random.normal(0,1,10000), drange(-9999))
    >>> ts = ewmstd(a,10); df = a.ewm(10).std()
    >>> assert abs(ts-df).max()<1e-10
    >>> ts = ewmstd(a,10, bias = True); df = a.ewm(10).std(bias = True)
    >>> assert abs(ts-df).max()<1e-10


    :Example: exc_zero
    ------------------
    >>> import pandas as pd; import numpy as np; from pyg import *
    >>> a = pd.Series(np.random.normal(0,1,10000), drange(-9999))
    >>> a[dt(-10):] = 0
    >>> exclude = ewmstd(a, 10, exc_zero = True)
    >>> assert exclude .iloc[-1] == exclude.iloc[-10]
    >>> no_exclude = ewmstd(a, 10)
    >>> assert no_exclude.iloc[-1] < no_exclude.iloc[-10]

    :Example: max_move
    ------------------
    >>> import pandas as pd; import numpy as np; from pyg import *
    >>> a = pd.Series(np.random.normal(0,1,10000), drange(-9999)) ## a series with stdev of 1 approximately
    >>> a.iloc[-1] = 100. ## put a bad data point with a huge spike...
    >>> max_move = ewmstd(a, 10, max_move = 4)
    >>> assert max_move.iloc[-1] < 2
    >>> no_max_move = ewmstd(a, 10)
    >>> assert no_max_move.iloc[-1] > 20

    :Example: numpy arrays support
    ------------------------------
    >>> assert eq(ewmstd(a.values, 10), ewmstd(a,10).values)

    :Example: nan handling
    ----------------------
    >>> a[a.values<-0.1] = np.nan
    >>> ts = ewmstd(a,10, time = 'i'); df = a.ewm(10).std() # note: pandas assumes, 'time' pass per index entry, even if value is nan
    >>> assert abs(ts-df).max()<1e-10
    >>> ts = ewmstd(a,10, time = 'i', bias = True); df = a.ewm(10).std(bias = True) # note: pandas assumes, 'time' pass per index entry, even if value is nan
    >>> assert abs(ts-df).max()<1e-10
    
        
    :Example: state management
    --------------------------
    >>> old = a.iloc[:5000]
    >>> new = a.iloc[5000:]
    >>> old_ts = ewmstd_(old, 10)
    >>> new_ts = ewmstd(new, 10, **old_ts) # instantiation with previous ewma
    >>> ts = ewmstd(a,10)
    >>> assert eq(new_ts, ts.iloc[5000:])

    :Example: Support for time & clock
    ------------------------------------
    >>> daily = a
    >>> monthly = daily.resample('M').last()
    >>> m_ts = ewmstd(monthly, 3) ## 3-month ewma run on monthly data
    >>> d_ts = ewmstd(daily, 3, 'm') ## 3-month ewma run on daily data
    >>> daily_resampled_to_month = d_ts.resample('M').last()
    >>> assert abs(daily_resampled_to_month - m_ts).max() < 1e-10

    So you can run a 3-monthly ewma on daily, where within month, most recent value is used with the EOM history.
    
    :Example: Support for dict/list of arrays
    --------------------------------------
    >>> x = pd.Series(np.random.normal(0,1,1000), drange(-999)); y = pd.Series(np.random.normal(0,1,1000), drange(-999))
    >>> a = dict(x = x, y = y)    
    >>> assert eq(ewmstd(dict(x=x, y=y),10), dict(x=ewmstd(x,10), y=ewmstd(y,10)))
    >>> assert eq(ewmstd([x,y],10), [ewmstd(x,10), ewmstd(y,10)])
    
             
    :Returns:
    -------
    an array/timeseries of ewma
    
    """
    state = state or {}
    return first_(_ewmstdt(a, n, time = time, min_sample=min_sample, bias = bias, axis=axis, exc_zero = exc_zero, max_move = max_move,  calculator = stdev_calculation_ewm, **state))


def ewmvar_(a, n, time = None, min_sample=0.25, bias = False, axis=0, exc_zero = False, data = None, instate = None):
    """
    Equivalent to ewmvar but returns a state parameter for instantiation of later calculations.
    See ewmvar documentation for more details
    """
    state = instate or {}
    max_move = 0
    return _data_state(['data', 't', 't0', 't1', 't2', 'w2'],_ewmstdt(a, n, time = time, min_sample=min_sample, axis=axis, exc_zero = exc_zero, max_move = max_move, calculator = variance_calculation_ewm, **state))

def ewmvar(a, n, time = None, min_sample=0.25, bias = False, axis=0, exc_zero = False, data = None, state = None):
    """
    ewmstd is equivalent to a.ewm(n).var() but with...
    - supports np.ndarrays as well as timeseries
    - handles nan by skipping them
    - allows state-management
    - ability to supply a 'clock' to the calculation
    
    :Parameters:
    ------------
    a : array/timeseries
    n : int/fraction
        The number or days (or a ratio) to scale the history
    time : Calendar, 'b/d/y/m' or a timeseries of time (use clock(a) to see output)
        If time parameter is provided, we allow multiple observations per unit of time. i.e., converging to the last observation in time unit. 
            - if we have intraday data, and set time = 'd', then 
            - the ewm calculation on last observations per day is what is retained. 
            - the ewm calculation on each intraday observation is same as an ewm(past EOD + current intraday observation)
    
    data: None.
        unused at the moment. Allow code such as func(live, **func_(history)) to work
    exc_zero: bool
        if True, will skip zeros inside a and will assume these are a result of forward filling
    state: dict, optional
        state parameters used to instantiate the internal calculations, based on history prior to 'a' provided. 
    
    :Example: matching pandas
    -------------------------
    >>> import pandas as pd; import numpy as np; from pyg import *
    >>> a = pd.Series(np.random.normal(0,1,10000), drange(-9999))
    >>> ts = ewmvar(a,10); df = a.ewm(10).var()
    >>> assert abs(ts-df).max()<1e-10
    >>> ts = ewmvar(a,10, bias = True); df = a.ewm(10).var(bias = True)
    >>> assert abs(ts-df).max()<1e-10


    :Example: numpy arrays support
    ------------------------------
    >>> assert eq(ewmvar(a.values, 10), ewmvar(a,10).values)

    :Example: nan handling
    ----------------------
    >>> a[a.values<-0.1] = np.nan
    >>> ts = ewmvar(a,10, time = 'i'); df = a.ewm(10).var() # note: pandas assumes, 'time' pass per index entry, even if value is nan
    >>> assert abs(ts-df).max()<1e-10
    >>> ts = ewmvar(a,10, time = 'i', bias = True); df = a.ewm(10).var(bias = True) # note: pandas assumes, 'time' pass per index entry, even if value is nan
    >>> assert abs(ts-df).max()<1e-10
    
        
    :Example: state management
    --------------------------
    >>> old = a.iloc[:5000]
    >>> new = a.iloc[5000:]
    >>> old_ts = ewmvar_(old, 10)
    >>> new_ts = ewmvar(new, 10, **old_ts) # instantiation with previous ewma
    >>> ts = ewmvar(a,10)
    >>> assert eq(new_ts, ts.iloc[5000:])

    :Example: Support for time & clock
    ------------------------------------
    >>> daily = a
    >>> monthly = daily.resample('M').last()
    >>> m_ts = ewmvar(monthly, 3) ## 3-month ewma run on monthly data
    >>> d_ts = ewmvar(daily, 3, 'm') ## 3-month ewma run on daily data
    >>> daily_resampled_to_month = d_ts.resample('M').last()
    >>> assert abs(daily_resampled_to_month - m_ts).max() < 1e-10

    So you can run a 3-monthly ewma on daily, where within month, most recent value is used with the EOM history.
    
    :Example: Support for dict/list of arrays
    --------------------------------------
    >>> x = pd.Series(np.random.normal(0,1,1000), drange(-999)); y = pd.Series(np.random.normal(0,1,1000), drange(-999))
    >>> a = dict(x = x, y = y)    
    >>> assert eq(ewmvar(dict(x=x, y=y),10), dict(x=ewmvar(x,10), y=ewmvar(y,10)))
    >>> assert eq(ewmvar([x,y],10), [ewmvar(x,10), ewmvar(y,10)])
    
             
    :Returns:
    -------
    an array/timeseries of ewma
    
    """
    state = state or {}
    max_move = 0
    return first_(_ewmstdt(a, n, time = time, min_sample=min_sample, bias = bias, axis=axis, exc_zero = exc_zero, max_move = max_move, calculator = variance_calculation_ewm, **state))


ewmstd_.output = ['data', 'state']


def ewmcor_(a, b, n, time = None, min_sample = 0.25, bias = True, axis = 0, data = None, instate = None):
    """
    Equivalent to ewmcor but returns a state parameter for instantiation of later calculations.
    See ewmcor documentation for more details
    """
    state = instate or {}    
    return _data_state(['data', 't', 't0', 'a1', 'a2', 'b1', 'b2', 'ab', 'w2'], _ewmcort(a = a, b = b, n = n, time = time, min_sample=min_sample, bias = bias, axis=axis, **state))

ewmcor_.output = ['data', 'state']


def ewmcor(a, b, n, time = None, min_sample = 0.25, bias = True, axis = 0, data = None, state = None):
    """
    calculates pair-wise correlation between a and b.
    
    :Parameters:
    ----------
    a : array/timeseries
    b : array/timeseries
    n : int/fraction
        The number or days (or a ratio) to scale the history
    time : Calendar, 'b/d/y/m' or a timeseries of time (use clock(a) to see output)
        If time parameter is provided, we allow multiple observations per unit of time. i.e., converging to the last observation in time unit. 
            - if we have intraday data, and set time = 'd', then 
            - the ewm calculation on last observations per day is what is retained. 
            - the ewm calculation on each intraday observation is same as an ewm(past EOD + current intraday observation)
    min_sample : floar, optional
        minimum weight of observations before we return a reading. The default is 0.25. This ensures that we don't get silly numbers due to small population.
    bias : book, optional
        vol estimation for a and b should really by unbiased. Nevertheless, we track pandas and set bias = True as a default.
    axis : int, optional
        axis of calculation. The default is 0.
    data : place holder, ignore, optional
        ignore. The default is None.
    state : dict, optional
        Output from a previous run of ewmcor_. The default is None.
        
    :Example: matching pandas
    -------------------------
    >>> import pandas as pd; import numpy as np; from pyg import *
    >>> a = pd.Series(np.random.normal(0,1,10000), drange(-9999))
    >>> b = pd.Series(np.random.normal(0,1,9000), drange(-8999))
    >>> ts = ewmcor(a, b, n = 10); df = a.ewm(10).corr(b)
    >>> assert abs(ts-df).max()<1e-10

    :Example: numpy arrays support
    ------------------------------
    >>> assert eq(ewmcor(a.values, b.values, 10), ewmcor(a, b, 10).values)

    :Example: nan handling
    ----------------------
    >>> a[a.values<-0.1] = np.nan
    >>> ts = ewmcor(a, b, 10, time = 'i'); df = a.ewm(10).corr(b) # note: pandas assumes, 'time' pass per index entry, even if value is nan
    >>> assert abs(ts-df).max()<1e-10

    :Example: state management
    --------------------------
    >>> a = pd.Series(np.random.normal(0,1,10000), drange(-9999))
    >>> b = pd.Series(np.random.normal(0,1,10000), drange(-9999))
    >>> old_a = a.iloc[:5000]; old_b = b.iloc[:5000]
    >>> new_a = a.iloc[5000:]; new_b = b.iloc[5000:]
    >>> old_ts = ewmcor_(old_a, old_b, 10)
    >>> new_ts = ewmcor(new_a, new_b, 10, **old_ts) # instantiation with previous ewma
    >>> ts = ewmcor(a,b,10)
    >>> assert eq(new_ts, ts.iloc[5000:])
    
    """
    state = state or {}    
    return first_(_ewmcort(a, b = b, n = n, time = time, min_sample=min_sample, bias = bias, axis=axis, **state))


def ewmGLM_(a, b, n, time = None, min_sample = 0.25, bias = True, data = None, instate = None):
    """
    Equivalent to ewmGLM but returns a state parameter for instantiation of later calculations.
    See ewmGLM documentation for more details
    """
    state = instate or {}    
    return _data_state(['data', 't', 't0', 'a2', 'ab'], _ewmGLMt(a = a, b = b, n = n, time = time, min_sample=min_sample, **state))

ewmGLM_.output = ['data', 'state']


def ewmGLM(a, b, n, time = None, min_sample = 0.25, bias = True, data = None, state = None):
    """
    Calculates a General Linear Model fitting b to a.
    
    :Parameters:
    ----------
    a : a 2-d array/pd.DataFrame of values fitting b
    b : a 1-d array/pd.Series
    n : int/fraction
        The number or days (or a ratio) to scale the history
    time : Calendar, 'b/d/y/m' or a timeseries of time (use clock(a) to see output)
        If time parameter is provided, we allow multiple observations per unit of time. i.e., converging to the last observation in time unit. 
            - if we have intraday data, and set time = 'd', then 
            - the ewm calculation on last observations per day is what is retained. 
            - the ewm calculation on each intraday observation is same as an ewm(past EOD + current intraday observation)
    min_sample : floar, optional
        minimum weight of observations before we return the fitting. The default is 0.25. This ensures that we don't get silly numbers due to small population.
    data : place holder, ignore, optional
        ignore. The default is None.
    state : dict, optional
        Output from a previous run of ewmGLM_. The default is None.
        
    :Theory:
    --------
    See https://en.wikipedia.org/wiki/Generalized_linear_model for full details.
    Briefly, we assume b is single column while a is multicolumn. 
    We minimize least square error (LSE) fitting:
    
    >>> b[i] =\sum_j m_j a_j[i]    
    >>> LSE(m) = \sum_i w_i (b[i] - \sum_j m_j * a_j[i])^2

    >>> dLSE/dm_k = 0  
    >>> <==>  \sum_i w_i (b[i] - \sum_j m_j * a_j[i]) a_k[i] = 0
    >>> <==>  E(b*a_k) = m_k E(a_k^2) + sum_{j<>k} m_k E(a_j a_k) 
    
    E is expectation under weights w. And we can rewrite it as:
    
    >>> a2 x m = ab ## matrix multiplication
    >>> a2[i,j] = E(a_i * a_j)
    >>> ab[j] = E(a_j * b)
    >>> m = a2.inverse x ab ## matrix multiplication    


    :Example: simple fit
    --------------------
    >>> from pyg import *
    >>> a = pd.DataFrame(np.random.normal(0,1,(10000,10)), drange(-9999))
    >>> true_m = np.random.normal(1,1,10)
    >>> noise = np.random.normal(0,1,10000)
    >>> b = (a * true_m).sum(axis = 1) + noise
    
    >>> fitted_m = ewmGLM(a, b, 50)    
        
    """
    state = state or {}    
    return first_(_ewmGLMt(a = a, b = b, n = n, time = time, min_sample=min_sample, **state))



def ewmLR_(a, b, n, time = None, min_sample = 0.25, bias = True, axis = 0, c = None, m = None, instate = None):
    """
    Equivalent to ewmcor but returns a state parameter for instantiation of later calculations.
    See ewmcor documentation for more details
    """
    state = instate or {}    
    return _data_state(['c', 'm', 't', 't0', 'a1', 'a2', 'b1', 'b2', 'ab', 'w2'], 
                       _ewmLRt(a = a, b = b, n = n, time = time, min_sample=min_sample, bias = bias, axis=axis, **state),
                       output = ['c', 'm'])

ewmLR_.output = ['c', 'm', 'state']

def ewmLR(a, b, n, time = None, min_sample = 0.25, bias = True, axis = 0, c = None, m = None, state = None):
    """
    calculates pair-wise linear regression between a and b.
    We have a and b for which we want to fit:
    
    >>> b_i = c + m a_i 
    >>> LSE(c,m) = \sum w_i (c + m a_i - b_i)^2
    >>> dLSE/dc  = 0  <==> \sum w_i  (c + m a_i - b_i) = 0    [1]
    >>> dLSE/dm  = 0 <==> \sum w_i  a_i (c + m a_i - b_i) = 0 [2]

    >>> c     + mE(a)    = E(b)     [1]
    >>> cE(a) + mE(a^2)  = E(ab)    [2]

    >>> cE(a) + mE(a)^2  = E(a)E(n) [1] * E(a) 
    >>> m(E(a^2) - E(a)^2) = E(ab) - E(a)E(b)
    >>> m = covar(a,b)/var(a)
    >>> c = E(b) - mE(a)

    
    Parameters
    ----------
    a : array/timeseries
    b : array/timeseries
    n : int/fraction
        The number or days (or a ratio) to scale the history
    time : Calendar, 'b/d/y/m' or a timeseries of time (use clock(a) to see output)
        If time parameter is provided, we allow multiple observations per unit of time. i.e., converging to the last observation in time unit. 
            - if we have intraday data, and set time = 'd', then 
            - the ewm calculation on last observations per day is what is retained. 
            - the ewm calculation on each intraday observation is same as an ewm(past EOD + current intraday observation)
    min_sample : floar, optional
        minimum weight of observations before we return a reading. The default is 0.25. This ensures that we don't get silly numbers due to small population.
    bias : book, optional
        vol estimation for a and b should really by unbiased. Nevertheless, we track pandas and set bias = True as a default.
    axis : int, optional
        axis of calculation. The default is 0.
    c,m : place holder, ignore, optional
        ignore. The default is None.
    state : dict, optional
        Output from a previous run of ewmcor_. The default is None.
    
    :Example: numpy arrays support
    ------------------------------
    >>> assert eq(ewmLR(a.values, b.values, 10), ewmLR(a, b, 10).values)

    :Example: nan handling
    ----------------------
    >>> a[a.values<-0.1] = np.nan
    >>> ts = ewmcor(a, b, 10, time = 'i'); df = a.ewm(10).corr(b) # note: pandas assumes, 'time' pass per index entry, even if value is nan
    >>> assert abs(ts-df).max()<1e-10

    :Example: state management
    --------------------------
    >>> from pyg import *
    >>> a = pd.Series(np.random.normal(0,1,10000), drange(-9999))
    >>> b = pd.Series(np.random.normal(0,1,10000), drange(-9999))
    >>> old_a = a.iloc[:5000]; old_b = b.iloc[:5000]
    >>> new_a = a.iloc[5000:]; new_b = b.iloc[5000:]
    >>> old_ts = ewmLR_(old_a, old_b, 10)
    >>> new_ts = ewmLR(new_a, new_b, 10, **old_ts) # instantiation with previous ewma
    >>> ts = ewmLR(a,b,10)
    >>> assert eq(new_ts.c, ts.c.iloc[5000:])
    >>> assert eq(new_ts.m, ts.m.iloc[5000:])
    
    
    :Example:
    ---------
    >>> from pyg import *
    >>> a0 = pd.Series(np.random.normal(0,1,10000), drange(-9999))
    >>> a1 = pd.Series(np.random.normal(0,1,10000), drange(-9999))
    >>> b = (a0 - a1) + pd.Series(np.random.normal(0,1,10000), drange(-9999))
    >>> a = pd.concat([a0,a1], axis=1)
    >>> LR = ewmLR(a,b,50)
    >>> assert abs(LR.m.mean()[0]-1)<0.5
    >>> assert abs(LR.m.mean()[1]+1)<0.5
    """
    state = state or {}
    return _data_state(['c', 'm'], _ewmLRt(a = a, b = b, n = n, time = time, min_sample=min_sample, bias = bias, axis=axis, **state),
                       output = ['c', 'm'])

ewmLR.output = ['c', 'm']

def ewmskew_(a, n, time = None, bias = False, min_sample = 0.25, axis=0, data = None, instate = None):
    """
    Equivalent to ewmskew but returns a state parameter for instantiation of later calculations.
    See ewmskew documentation for more details
    """
    state = instate or {}
    return _data_state(['data', 't', 't0', 't1', 't2', 't3'],_ewmskewt(a, n, time = time, bias = bias, min_sample = min_sample, axis = axis, **state))

ewmskew_.output = ['data', 'state']

def ewmskew(a, n, time = None, bias = False, min_sample = 0.25, axis=0, data = None, state = None):
    """
    Equivalent to a.ewm(n).skew() but with...
    - supports np.ndarrays as well as timeseries
    - handles nan by skipping them
    - allows state-management
    - ability to supply a 'clock' to the calculation
    
    :Parameters:
    ------------
    a : array/timeseries
    n : int/fraction
        The number or days (or a ratio) to scale the history
    time : Calendar, 'b/d/y/m' or a timeseries of time (use clock(a) to see output)
        If time parameter is provided, we allow multiple observations per unit of time. i.e., converging to the last observation in time unit. 
            - if we have intraday data, and set time = 'd', then 
            - the ewm calculation on last observations per day is what is retained. 
            - the ewm calculation on each intraday observation is same as an ewm(past EOD + current intraday observation)
    
    data: None.
        unused at the moment. Allow code such as func(live, **func_(history)) to work
    state: dict, optional
        state parameters used to instantiate the internal calculations, based on history prior to 'a' provided. 
    
    :Example: matching pandas
    -------------------------
    >>> import pandas as pd; import numpy as np; from pyg import *
    >>> a = pd.Series(np.random.normal(0,1,10000), drange(-9999))
    >>> x = a.ewm(10).skew()

    >>> old = a.iloc[:10]
    >>> new = a.iloc[10:]
    >>> for f in [ewma_, ewmstd_, ewmrms_, ewmskew_, ]:
    >>>    both = f(a, 3)
    >>>    o = f(old, 3)
    >>>    n = f(new, 3, **o)
    >>>    assert eq(o.data, both.data.iloc[:10]) 
    >>>    assert eq(n.data, both.data.iloc[10:]) 
    >>>    assert both - 'data' == n - 'data'

    >>> assert abs(a.ewm(10).mean() - ewma(a,10)).max() < 1e-14
    >>> assert abs(a.ewm(10).std() - ewmstd(a,10)).max() < 1e-14

    :Example: numpy arrays support
    -------------------------------
    >>> assert eq(ewma(a.values, 10), ewma(a,10).values)

    :Example: nan handling
    ----------------------
    while panadas ffill values, timeseries skips nans:

    >>> a = pd.Series(np.random.normal(0,1,10000), drange(-9999))
    >>> a[a.values>0.1] = np.nan
    >>> ts = ewma(a,10)
    >>> assert eq(ts[~np.isnan(ts)], ewma(a[~np.isnan(a)], 10))
    
    :Example: initiating the ewma with past state
    ------------------------------------------------
    >>> old = np.random.normal(0,1,100)
    >>> new = np.random.normal(0,1,100)
    >>> old_ = ewma_(old, 10)
    >>> new_ = ewma(new, 10, t0 = old_ewma.t0, t1 = old_ewma.t1) # instantiation with previous ewma
    >>> new_2 = ewma(np.concatenate([old,new]), 10)[-100:]
    >>> assert eq(new_ewma, new_ewma2)

    :Example: Support for time & clock
    -----------------------------------
    >>> daily = pd.Series(np.random.normal(0,1,10000), drange(-9999)).cumsum()
    >>> monthly = daily.resample('M').last()
    >>> m = ewma(monthly, 3) ## 3-month ewma run on monthly data
    >>> d = ewma(daily, 3, 'm') ## 3-month ewma run on daily data
    >>> daily_resampled_to_month = d.resample('M').last()
    >>> assert abs(daily_resampled_to_month - m).max() < 1e-10

    So you can run a 3-monthly ewma on daily, where within month, most recent value is used with the EOM history.

    
    :Returns:
    -------
    an array/timeseries of ewma
    
    """
    state = state or {}
    return first_(_ewmskewt(a, n, time = time, bias = bias, min_sample=min_sample, axis=axis, **state))

