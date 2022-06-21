from pyg_base import df_reindex, is_df, pd2np, loop, Dict, is_nums, is_pd, is_arr, df_concat, is_series, is_strs
from pyg_timeseries._linalg import matmul
from pyg_timeseries._rolling import buffer
import numpy as np

def bisect(f, lb, ub, n = 0, aim = np.nan):
    """
    bisection like scipy.optimize.bisect but supports discontinous functions snapping towards our "aim"
    
    Example:
    --------
    >>> f = lambda x: 1 if x>0 else -1
    >>> assert bisect(f, 1, 3, aim = 1.3) == 1.3
    >>> assert bisect(f, -1, 3, aim = 1.3) == 1.3    

    >>> f = lambda x: 1 if x>2 else 0.5 if x>1 else -1 
    >>> assert round(bisect(f, -1, 3, 20, aim = 3),3) == 1.0 ## The crossing from positive to negative is at 1 so aim of 3 does not chain

    """
    flb = f(lb)
    fub = f(ub)
    for i in range(n or 10):
        if fub * flb >= 0 :
            break
        else:
            mid = (lb + ub) / 2
            fmid = f(mid)
            if flb * fmid < 0:
                ub = mid
                fub = fmid
            else:
                lb = mid
                flb = fmid
    mid = (lb + ub) / 2
    if np.isnan(aim):
        return mid                    
    elif f(aim) == f(mid):
        return aim
    elif aim < mid:
        return lb
    else:
        return ub


def _single_step_multibuffer(target, band, weights, vol, covariances, previous, unit = 1.0, aim = 1.0, lb = None, ub = None):    
    """
    covariances = np.array([[1,0.3], [0.3, 1]])
    weights = np.array([0.5, 0.5])
    vol = np.array([1.2, 4.3])
    mult = 3
    target = np.array([0.3, 0.8])
    previous = np.array([0., 2.])
    band = np.array([0.1, 0.1])
    from pyg import *
    a = np.array([target])
    state = Dict(pos = list(previous))
    u = list(unit)
    
    """
    w = np.matmul(np.array([weights * vol]).T, np.array([weights * vol]))
    cw = covariances * w
    cw[np.isnan(cw)] = 0.0
    target[np.isnan(target)] = 0.0
    target_risk = matmul(cw, target, target, 0.5)
    b = np.array([band])
    state = dict(pos = list(previous))
    def risk_mismatch(m):
        buffered_pos = buffer(a = np.array([target * m]), band = b, state = state, unit = unit)[0]
        buffered_risk = matmul(cw, buffered_pos, buffered_pos, 0.5)
        return buffered_risk - target_risk
    m = bisect(risk_mismatch, lb, ub, 5, aim)
    buffered_pos = buffer(a = np.array([target * m]), band = b, state = state, unit = unit)[0]
    return m, buffered_pos


@loop(list, tuple, dict)
def _from(value, n, t = None):
    if isinstance(value, np.ndarray) and (t is None or len(value) == t):
        return value[n:]
    else:
        return value

@loop(list)
def _subset(value, keys = None, ids = None):
    if keys is None and ids is None:
        return value
    elif is_df(value):
        return value.iloc[:, ids] if keys is None else value[keys]
    elif is_series(value):
        return value.iloc[ids] if keys is None else value[keys]
    elif is_arr(value):
        if ids is None:
            return value
        elif len(value.shape) == 1:
            return value[ids]
        elif len(value.shape) == 2:
            return value[:,ids]
        else:
            res = value[:,ids,:]
            res = res[:,:,ids]
            return res
    else:
         return value
    

@pd2np
def _multibuffer(target, band, unit, covariances, risks, weights, data = None, mult = None, mult_band = 0.1):
    if data is None:
        previous = np.zeros(target.shape[1])
        m = 1
    else:
        previous = data[-1]
        m = mult[-1]
        n = len(data)
        t = len(target)
        if n == t:
            return Dict(mult = mult, data = data)
        target, band, unit, covariances, risks, weights = _from([target, band, unit, covariances, risks, weights], n = n, t = t)
    ds = np.zeros(target.shape)
    ms = np.ones(target.shape[0])
    for i in range(target.shape[0]):
        m, previous = _single_step_multibuffer(target = target[i], band = band[i], weights = weights[i], vol = risks[i], 
                                               covariances = covariances[i], previous = previous, unit = unit, aim = m, lb = m - mult_band, ub = m + mult_band)
        ds[i] = previous
        ms[i] = m
        if i % 1000 == 0:
            print(i)
    if data is None:
        return Dict(mult = ms, data = ds)
    else:
        return Dict(mult = np.concatenate([mult, ms]), data = np.concatenate([data, ds]))

_multibuffer.output = ['data', 'mult']

@loop(dict)
def _subset_multibuffer(subset, target, band, unit, covariances, risks, weights, data = None, mult = None, mult_band = 0.1):
    if is_nums(subset):
        ids = subset
        keys = target.columns[ids] if is_df(target) else None
    elif is_strs(subset):
        keys = subset
        ids = Dict(zip(target.columns, range(len(target.columns))))[tuple(keys)] if is_df(target) else None
    target, band, unit, covariances, risks, weights, data , mult , mult_band = _subset([target, band, unit, covariances, risks, weights, data , mult , mult_band], ids = ids, keys = keys)
    return _multibuffer(target = target, band = band, unit = unit, covariances = covariances, risks = risks, weights = weights, data = data, mult = mult, mult_band = mult_band)


def multibuffer(target, band, unit, covariances, risks, weights, data = None, mult = None, mult_band = 0.1, subset = None):
    """
    Assumes 'a' is a vector of target positions 
    performs a buffering of a but aiming to target a given level of risk
    
    :Example:
    ---------
    >>> target = get_data('data','long_only', item = 'lots') ## grab original target lots per each asset over time
    >>> band = get_data('data','long_only', item = 'band')
    >>> unit = 1.0
    >>> risks = get_data('data','long_only', item = 'fut_vol')
    >>> weights = get_data('data','long_only', item = 'rpz')
    >>> covariances = get_data('data','long_only', item = 'covariance_zero')
    >>> buffered_lots = get_data('data','long_only', item = 'buffered_lots')
    >>> data = None ## output
    >>> mult = None ## output
    
    ## some stats

    >>> print(dictable(ticker = list(res.data.columns), new = list(tover(res.data).values), b = list(tover(buffered_lots).values)).do(f12))
    >>> ticker = 'HGA Comdty'    
    >>> df_concat([target[ticker], res.data[ticker], buffered_lots[ticker]], ['orig', 'new', 'b']).plot(title = ticker)

    
    In addition, you may want to ensure the risk in different sub-populations is maintained as well. 
    For this purpose, you can either run it with 
        - subset = [... list of all bonds]. This will subset the data to run on the bonds only
        - subset = dict(telecom = [list of telecom stocks], health = [list of health stocks], ...)
        This will return a multi-column mult dataframe with the multiplier per each sector and data = final positions for all stocks
                        
    :Example:
    ---------
    >>> res = multibuffer(target, band, unit, covariances, risks, weights, data = data, mult = mult, mult_band = mult_band, subset = ['TYA Comdty', 'ESA Index'])

    
    """
    if is_df(risks) and is_df(target):
        risks = df_reindex(risks, target, method = 'ffill')
    res = _subset_multibuffer(subset = subset, target = target, band = band, unit = unit, covariances = covariances, risks = risks, weights = weights, data = data, mult = mult, mult_band = mult_band)
    if isinstance(subset, dict):
        rtn = Dict(mult = df_concat([r.mult for r in res.values()], list(subset.keys())), data = df_concat([r.data for r in res.values()]))
        rtn = rtn[target.columns]
        return rtn
    else:
        return res

multibuffer.output = _multibuffer.output

