import pandas as pd
import numpy as np
from pyg_base import pd2np, is_pd, loop, is_df, df_reindex, df_concat, is_tss
from pyg_timeseries import matmul, reindex_3d, na2v, ffill

### single pass position calculation

@loop(list, tuple)
@pd2np
def _as_int(pos):
    res = pos.copy()
    res[np.isnan(res)] = 0    
    res = res.astype(int) ## This is rounding down
    return res

@pd2np
def _int_nearest_target(target, init = None):
    """
    makes init goes within nearest unit of target size
    
    :Parameters:
    ============
    init: int array
        initial value
        
    target: float array
        target position to get to
        
    :Example:
    =========

    This ensures our initial position is never "above" the target position
    >>> target = np.array([0.2, 0.2, 0.2])
    >>> init = np.array([3,0,-3])
    >>> assert list(_int_nearest_target(target, init)) == [1,0,0]
    >>> assert list(_int_nearest_target(target)) == [0,0,0]
    """
    if init is None:
        init = np.zeros(target.shape)
    d = (target - init).astype(int)
    res = init + d
    return res

    

def _minimize_tracking_error(covariance, target, multiplier = None, position = None, relative_cost = None, min_change = 0.01, search = 2):
    n = target.shape[0]
    ns = np.arange(n)
    res = _int_nearest_target(target, position)    
    covariance[np.isnan(covariance)] = 0.0
    if is_pd(multiplier):
        multiplier = multiplier.values
    if multiplier is not None:
        multiplier[np.isnan(multiplier)] = 0.0
        rr = np.array(np.matmul(np.reshape(multiplier, (n,1)), np.reshape(multiplier, (1,n))))
        cr = np.array(covariance) * rr
    else:
        cr = np.array(covariance)
    j = 0
    m = -1
    moved = set()
    d = na2v(target - res) # if there is a nan in target, we do not trade
    gradient = matmul(cr, d)
    base = matmul(gradient, d)

    while m < -abs(min_change) and j < 20:
        m = 0
        sgn = np.sign(d)
        gbase =  -2 * gradient * sgn ## theoretical gradient
        mask = gbase < 0
        gbase = gbase[mask] # only trade if gradient is down
        nn = ns[mask]
        gn = sorted(zip(gbase, nn)) 
        ok = True
        found = 0
        dbase = [0] * len(ns)
        for g, i in gn: 
            ## we start from the lowest gradient and work our way up. Looking for valid candidates 
            ## The moment we find a enough candidates to perform a valid reduction in tracking error, we reduce. 
            ## if search = 0; we search exhaustively manually. search = 1 is potentially less accurate but faster. 
            ## The assumption being is that if there is another candidate, it will show in the next round of reduction anyway
            if ok and g < 0 and i not in moved:
                r = res.copy()
                r[i] = r[i] + sgn[i]
                d = na2v(target - r)
                dbase[i] = m = matmul(cr, d, d) / base - 1                
                if m < -abs(min_change):
                    found +=1
                    if search > 0 and found >= search:
                        ok = False
        if found > 1:
            m = min(dbase)
            i = dbase.index(m)
            
        if m < -abs(min_change):
            moved.add(i)
            res[i] = res[i] + sgn[i]
            d = na2v(target - res) # if there is a nan in target, we do not trade
            gradient = matmul(cr, d)
            base = matmul(gradient, d)    
            if base <= 0:
                return res, base
            j = j +1

    return res, base
                
@loop(list, tuple)
def _as_np(value):
    return value.values if is_pd(value) else value
 
@loop(list, tuple)
def _post_data(value, n, i):
    if isinstance(value, np.ndarray) and len(value) == n:
        return value[i:]
    else:
        return value

def _get_i(value, n, i):
    if isinstance(value, np.ndarray) and len(value) == n:
        return value[i]
    else:
        return value
        

def minimize_tracking_error(covariance, target, multiplier = None, min_change = 0.01, covariance_index = None, search = 2, data = None, errors = None):
    """
    Tracks a target position within an int, while minimizing the tracking error defined as 
    
    Tracking error = (data-target)^T x Covariance x (data-target)

    Parameters
    ----------
    covariance : 3d np.ndarray 
        covariance of the quote time data time x n x n 
    target : 2d pd.DataFrame or np.ndarray
        The position we want to take in each of the n assets, in FRACTIONS
    multiplier : 2d pd.DataFrame, optional
        Each asset may have a multiplier. This can be due to e.g. price being in a different currency or the quote (in yield) change translating into actual value changes via dv01 multiplier.
    min_change : float<1, optional
        We stop trying to optimize when we can improve the tracking error by at most min_change. The default is 0.01.
    data : pd.DataFrame, optional
        The previous output from this function. The default is None.
        The output tracks the target positions but is always an integer.

    Returns
    -------
    data : pd.DataFrame
        A position that tracks "target" but is:
            - always an integer
            - minimizes tracking error 

    """
    if is_tss(target):
        target = df_concat(target, method = 'ffill')
    if data is not None and len(data) == len(target):
        return dict(data = data, errors = errors)

    if is_df(target):
        multiplier = df_reindex(multiplier, target, method = 'ffill')
        if covariance_index is not None:
            covariance = reindex_3d(covariance, target.index, covariance_index, method = 'ffill')

    c, t, r, d, e = _as_np([covariance, target, multiplier, data, errors])
    c = na2v(ffill(c))
    t = na2v(ffill(t))
    if d is not None and len(d):        
        c, t, r = _post_data([c, t, r], len(c), len(d))
        position = d[-1]
    else:
        position = None
    n = len(c)
    positions = []
    tracking_errors = []
    for i in range(len(c)):
        if i % 100 == 0:
            print('running %i of %i'%(i, len(c)))
        position, tracking_error = _minimize_tracking_error(covariance = _get_i(value = c, n = n, i = i), 
                                                            target = _get_i(value = t, n = n, i = i), 
                                                            multiplier = _get_i(value = r, n = n, i = i), 
                                                            position = position, 
                                                            min_change = min_change,
                                                            search = search)
        positions.append(position)
        tracking_errors.append(tracking_error)
    positions = np.array(positions)
    tracking_errors = np.array(tracking_errors)
    if d is not None and len(d):
        positions = np.concatenate([d, positions])
    if e is not None and len(e):
        tracking_errors = np.concatenate([e, tracking_errors])
    if is_df(target):
        return dict(data = pd.DataFrame(positions, target.index, target.columns), errors = pd.Series(tracking_errors, target.index))
    else:
        return dict(data = positions, errors = tracking_errors)

minimize_tracking_error.output = ['data', 'errors']