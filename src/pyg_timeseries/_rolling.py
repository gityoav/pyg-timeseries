import numpy as np
from pyg_timeseries._math import stdev_calculation, skew_calculation, _w
from pyg_timeseries._decorators import compiled, first_, _data_state
from pyg_base import pd2np, Dict, is_num, loop_all, loop, clock, is_pd, df_reindex

__all__ = ['ffill', 'bfill', 'fnna', 'na2v', 'v2na', 'diff', 'shift', 'ratio', 'rolling_mean', 'rolling_sum', 'rolling_rms', 'rolling_std', 'rolling_skew', 
           'diff_', 'shift_', 'ratio_', 'rolling_mean_', 'rolling_sum_', 'rolling_rms_', 'rolling_std_', 'rolling_skew_']

###############
##
## parameters
##
###############


@loop(list, dict)
def _vec(a, vec, n, value = np.nan, axis = 0):
    if len(a.shape)==2:
        shape = a.shape[1] if axis==0 else a.shape[0]
        if vec is None:
            return [np.full(abs(n), value)]*shape
        elif is_num(vec):
            return [np.array([vec])]*shape
        else:
            return vec.copy()
    else:        
        if vec is None:
            return np.full(abs(n), value)
        elif is_num(vec):
            return np.array([vec])
        return vec.copy()

@loop_all
@pd2np
@compiled
def _fnna(a, n):
    if n == 0:
        raise ValueError('n must be non-zero')
    i = 0
    _n = abs(n)
    s = (0,a.shape[0],1) if n>0 else (a.shape[0]-1,-1,-1)
    for j in range(*s):
        if not np.isnan(a[j]):
            i = i+1
            if i == _n:
                return j


@loop_all
@pd2np
@compiled
def _fnnz(a, n=1):
    if n == 0:
        raise ValueError('n must be non-zero')
    i = 0
    _n = abs(n)
    s = (0,a.shape[0],1) if n>0 else (a.shape[0]-1,-1,-1)
    for j in range(*s):
        if not np.isnan(a[j]) and a[j]!=0.:
            i = i+1
            if i == _n:
                return j
            

@compiled
def _ffill1d(a, n, prev, i):
    res = a.copy()
    for j in range(a.shape[0]):
        if np.isnan(a[j]):
            if n == 0:
                res[j] = prev
            else:
                i+=1
                if i>n:
                    res[j] = np.nan
                else:
                    res[j] = prev
        else:
            i = 0
            prev = a[j]
    return res, prev, i


def _ffill2d(a, n, prev, i):
    res = a.copy()
    if n:
        for j in range(a.shape[0]):
            mask = np.isnan(res[j])
            i[mask]+=1
            i[~mask] = 0
            prev[i>n] = np.nan
            res[j][mask] = prev[mask]
            prev = res[j].copy()
    else:
        for j in range(a.shape[0]):
            mask = np.isnan(res[j])
            res[j][mask] = prev[mask]
            prev = res[j]
    return res, prev, i

@loop(dict, list)
@pd2np
def _ffill(a, n = 0, prev = None, i = None):
    if len(a.shape) == 1:
        if i is None:
            i = 0
        if prev is None:
            prev = np.nan
        return _ffill1d(a, n, prev , i)
    else:            
        if i is None:
            i = np.zeros(a[0].shape)
        if prev is None:
            prev = i + np.nan
        return _ffill2d(a, n, prev , i)


@loop(dict, list)
@pd2np
def _na2v(a, new = 0.0):
    res = a.copy()
    if len(a.shape) == 1:
        res[np.isnan(res)] = new
    else:
        for j in range(a.shape[0]):
            res[j, np.isnan(res[j])] = new
    return res

@loop(dict, list)
@pd2np
def _v2na(a, old = 0.0, new = np.nan):
    res = a.copy().astype(float)
    if len(a.shape) == 1:
        res[res == old] = new
    else:
        for j in range(a.shape[0]):
            res[j, res[j] == old] = new
    return res


@loop_all
@pd2np
@compiled
def _init2v(a, n = 0, new = np.nan):
    res = a.copy()
    i = 0
    while i < a.shape[0] and n > 0:
        if ~np.isnan(a[i]):
            res[i] = new
            n = n-1
        i+=1
    return res


@loop_all
@pd2np
@compiled
def _bfill(a, limit = -1):
    """
    _bfill(np.array([np.nan, 1., np.nan])) 
    """
    res = a.copy()
    prev = np.nan
    n = limit
    for j in range(a.shape[0]-1, -1, -1):
        if np.isnan(a[j]):
            if n!=0:
                n-=1
                res[j] = prev
        else:
            n = limit
            prev = res[j]
    return res


@loop_all
@pd2np
@compiled
def _ewfill(a, fwd_n, bwd_n = np.nan, prev = np.nan, nxt = np.nan, decay_target = 0.):
    res = a.copy()
    prev = decay_target if np.isnan(prev) else prev
    nxt = decay_target if np.isnan(nxt) else nxt
    fw = _w(fwd_n)
    bw = fw if np.isnan(bwd_n) else _w(bwd_n)
    f = b = 1.
    for j in range(a.shape[0]): ## forward sweep
        if np.isnan(a[j]):
            f *= fw
            res[j] = 0.5 * (prev * f  + (1 - f) * decay_target)
        else:
            f = 1.
            prev = res[j]    
    for j in range(a.shape[0] - 1,  -1, -1):
        if np.isnan(a[j]):
            b *= bw
            res[j] += 0.5*(nxt * b  + (1 - b) * decay_target)
        else:
            b = 1.
            nxt = res[j]
    return res


###############
##
## bottleneck
##
###############

@loop_all
@pd2np
def _rolling_window(a, window, min_count, func, vec = None, axis = 0):
    vec = _vec(a,vec,0,axis=axis)
    mask = ~np.isnan(a)
    na = a[mask]
    n = len(na)
    if len(vec):
        na = np.concatenate([vec,na])
    w = na.shape[0] if window == 0 else window        
    res = func(na, w, min_count)
    if len(vec):
        res = res[-n:] 
    va = a.copy()
    va[mask] = res
    return va, va[-1:] if window == 0 else na[-(window-1):]


###############
##
## diff/shift/ratio
##
###############

@loop_all
@pd2np
@compiled
def _diff(a, n, vec, i, time, t = np.nan):
    """
    a = cumsum(np.array(range(10))) * 1.
    n = 1
    vec = np.array([np.nan])
    i = 0; t = np.nan
    time = np.array([0,0,1,1,1,2,2,2,2,3])

    """
    vec = vec.copy()
    _n = abs(n)
    s = (0,a.shape[0],1) if n>0 else (a.shape[0]-1,-1,-1)
    res = np.empty_like(a)
    prev = vec[i]
    for j in range(*s):
        if np.isnan(a[j]):
            res[j] = np.nan
        else:
            if not (time[j] == t):
                i = (i+1) % _n
                prev = vec[i]
                # print('moving forward in time, new previous is', prev)
            t = time[j]
            res[j] = a[j] - prev
            vec[i] = a[j]
            # print('prev is ', prev, 'time is ', t, 'res is', a[j], '-', prev, '=', res[j])
    return res, vec, i, t

@loop_all
@pd2np
@compiled
def _diff1(a, vec, time, i = 0, t = np.nan):
    vec = vec.copy()
    s = (0,a.shape[0],1)
    res = np.empty_like(a)
    prev = vec[i]
    for j in range(*s):
        if np.isnan(a[j]):
            res[j] = np.nan
        else:
            if not (time[j] == t):
                prev = vec[i]
            t = time[j]
            res[j] = a[j] - prev
            vec[i] = a[j]
    return res, vec, i, t


@loop_all
@pd2np
@compiled
def _buffer(a, band, unit = 0.0, pos = 0):
    """
    
    Handles buffering when the rounding into units may be significant cost
    
    
    >>> from pyg import *
    >>> a = pd.Series(cumsum(np.random.normal(0,1,1000)), drange(-999))
    >>> signal = ewmacd(a, 1, 3, vol = 18)
    >>> band = 0.1; unit = 1.5
    >>> buffered = buffer(signal, band = band, unit = unit)
    >>> sim = np.round(buffer(signal, band = band) / unit) * unit
    >>> df_concat([signal, sim, buffered], ['signal %i'%tover(signal), 'sim %i'%tover(sim), 'buffered %i'%tover(buffered), ])[dt(-200):].plot()
    """
    res = np.full(a.shape, np.nan)
    b = 0.0
    if np.isnan(pos):
        pos = 0.0
    for i in range(a.shape[0]):
        if not np.isnan(a[i]):
            if not np.isnan(band[i]): ## we forward fill band
                b = band[i]
            if pos < a[i] - b:
                aim = a[i] - b
                if unit > 0:
                    aim = np.round(aim / unit) * unit
                    if aim < a[i] - b and aim + unit < a[i] + b:
                        pos = aim + unit
                    elif aim > a[i] + b and (aim - a[i]) - (a[i] - pos) < min(b, unit/3):
                        pos = aim - unit
                    else:
                        pos = aim
                else:
                    pos = aim
            elif pos > a[i] + b:
                aim = a[i] + b
                if unit > 0:
                    aim = np.round(aim / unit) * unit
                    if aim > a[i] + b and aim - unit > a[i] - b:
                        pos = aim - unit
                    elif aim < a[i] - b and (a[i] - aim) - (pos - a[i]) < min(b, unit/3):
                        pos = aim + unit
                    else:
                        pos = aim
                else:
                    pos = aim
            res[i] = pos
    return res, pos



@loop_all
def _tdiff(a, n, vec, i, time = None, t = None):
    time = clock(a, time, t)
    t = 0 if t is None or np.isnan(t) else t
    return _diff(a = a, n = n, vec = vec, i = i, time = time, t = t)


@loop_all
def _tdiff1(a, vec, i = 0, time = None, t = None):
    time = clock(a, time, t)
    t = 0 if t is None or np.isnan(t) else t
    return _diff1(a = a, vec = vec, i = i, time = time, t = t)


@loop_all
@pd2np
@compiled
def _ratio(a, n, vec, i, time, t = np.nan):
    vec = vec.copy()
    _n = abs(n)
    s = (0,a.shape[0],1) if n>0 else (a.shape[0]-1,-1,-1)
    res = np.empty_like(a)
    prev = vec[i]
    for j in range(*s):
        if np.isnan(a[j]):
            res[j] = np.nan
        else:
            if not (time[j] == t):
                i = (i+1) % _n
                prev = vec[i]
            t = time[j]
            res[j] = np.nan if prev == 0 else a[j] / prev 
            vec[i] = a[j]
    return res, vec, i, t

@loop_all
def _tratio(a, n, vec, i, time = None, t = None):
    time = clock(a, time, t)
    t = 0 if t is None or np.isnan(t) else t
    return _ratio(a = a, n = n, vec = vec, i = i, time = time, t = t)


@loop_all
@pd2np
@compiled
def _shift(a, n, vec, i):
    vec = vec.copy()
    res = np.empty_like(a)
    _n = abs(n)
    s = (0,a.shape[0],1) if n>0 else (a.shape[0]-1,-1,-1)
    for j in range(*s):
        if np.isnan(a[j]):
            res[j] = np.nan
        else:
            res[j] = vec[i]
            vec[i] = a[j]
            i = (i+1) % _n
    return res, vec, i


@loop_all
@pd2np
@compiled
def _shift1(a, vec):
    vec = vec.copy()
    res = np.empty_like(a)
    for j in range(a.shape[0]):
        if np.isnan(a[j]):
            res[j] = np.nan
        else:
            res[j] = vec[0]
            vec[0] = a[j]
    return res, vec, 0


###############
##
## rolling
##
###############


@loop_all
@pd2np
@compiled
def _rolling_tover(a, n = 256, interval = None, positions = None, trades = None, j = 0, total_variance = 0, total_trades = 0):
    if interval is None:
        interval = 1 / 260
    v = np.empty_like(a)
    prev = positions[j]
    total_years = n * interval
    for i in range(a.shape[0]):
        jj = (j + 1) % n ## this is the position we want to be removing        
        if np.isnan(a[i]):
            positions[j] = prev
        else:
            positions[j] = a[i]
        trades[j] = abs(positions[j] - prev)
        total_variance += positions[j] ** 2 - positions[jj] ** 2
        total_trades += (trades[j] - trades[jj])        
        annual_variance = (total_variance * interval) / total_years
        annual_trading = total_trades / total_years
        if annual_variance <= 0:
            v[i] = np.nan
        else:
            v[i] = annual_trading / (annual_variance ** 0.5)
        prev = positions[j]
        j = jj
    return v, positions, trades, j, total_variance, total_trades


@pd2np
@compiled
def _rolling_mean(a, n, time, t0, t1, vec, i, denom, t = np.nan):
    vec = vec.copy()
    res = np.empty_like(a)
    _n = abs(n)
    s = (0,a.shape[0],1) if n>0 else (a.shape[0]-1,-1,-1)
    for j in range(*s):
        if np.isnan(a[j]):
            res[j] = np.nan
        else:
            if not (time[j] == t):
                i = (i+1) % _n
                t0 +=1
            t1 += a[j]-vec[i]
            vec[i] = a[j]
            res[j] = np.nan if t0<n else t1/denom
            t = time[j]
    return res, t0, t1, vec, i, t


@loop_all
def _trolling_mean(a, n, t0, t1, vec, i, denom, t = np.nan, time = None):
    time = clock(a, time, t)
    return _rolling_mean(a, n, time = time, t = t, t0 = t0, t1 = t1, vec = vec, i = i, denom = denom)


@pd2np
@compiled
def _rolling_rms(a, n, time, t0, t2, vec, i, denom, t = np.nan):
    vec = vec.copy()
    res = np.empty_like(a)
    _n = abs(n)
    s = (0,a.shape[0],1) if n>0 else (a.shape[0]-1,-1,-1)
    for j in range(*s):
        if np.isnan(a[j]):
            res[j] = np.nan
        else:
            if not (time[j] == t):
                i = (i+1) % _n
                t0 +=1
            t2 += a[j]**2-vec[i]**2
            vec[i] = a[j]
            res[j] = np.nan if t0<n else np.sqrt(t2/denom)
            t = time[j]
    return res, t0, t2, vec, i, t

@loop_all
def _trolling_rms(a, n, t0, t2, vec, i, denom, t = np.nan, time = None):
    time = clock(a, time, t)
    return _rolling_rms(a, n, time = time, t = t, t0 = t0, t2 = t2, vec = vec, i = i, denom = denom)

@pd2np
@compiled
def _rolling_std(a, n, time, t0, t1, t2, vec, i, denom, t = np.nan):
    vec = vec.copy()
    res = np.empty_like(a)
    _n = abs(n)
    s = (0,a.shape[0],1) if n>0 else (a.shape[0]-1,-1,-1)
    for j in range(*s):
        if np.isnan(a[j]):
            res[j] = np.nan
        else:
            if not (time[j] == t):
                i = (i+1) % _n
                t0 +=1
            t1 += a[j]-vec[i]
            t2 += a[j]**2-vec[i]**2
            vec[i] = a[j]
            res[j] = np.nan if t0<n else stdev_calculation(t0 = n, t1 = t1, t2 = t2)
            t = time[j]
    return res, t0, t1, t2, vec, i, t

@loop_all
def _trolling_std(a, n, t0, t1, t2, vec, i, denom, t = np.nan, time = None):
    time = clock(a, time, t)
    return _rolling_std(a, n, time = time, t = t, t0 = t0, t1 = t1, t2 = t2, vec = vec, i = i, denom = denom)


@pd2np
@compiled
def _rolling_skew(a, n, time, bias, t0, t1, t2, t3, vec, i, denom, t = np.nan):
    vec = vec.copy()
    res = np.empty_like(a)
    _n = abs(n)
    s = (0,a.shape[0],1) if n>0 else (a.shape[0]-1,-1,-1)
    for j in range(*s):
        if np.isnan(a[j]):
            res[j] = np.nan
        else:
            if not (time[j] == t):
                i = (i+1) % _n
                t0 +=1
            t1 += a[j]-vec[i]
            t2 += a[j]**2-vec[i]**2
            t3 += a[j]**3-vec[i]**3
            vec[i] = a[j]
            res[j] = np.nan if t0<_n else skew_calculation(t0 = _n, t1 = t1, t2 = t2, t3 = t3, bias = bias, min_sample = 1)
            t = time[j]
    return res, t0, t1, t2, t3, vec, i, t


@loop_all
def _trolling_skew(a, n, t0, t1, t2, t3, vec, i, denom, bias = False, t = np.nan, time = None):
    time = clock(a, time, t)
    return _rolling_skew(a, n = n, time = time, bias = bias, t = t, t0 = t0, t1 = t1, t2 = t2, t3 = t3, vec = vec, i = i, denom = denom)


###############
##
## API
##
###############


def fnna(a, n=1, axis = 0):
    """
    returns the index in a of the nth first non-nan.
    
    :Parameters:
    ------------
    a : array/timeseries
    n: int, optional, default = 1

    :Example:
    ---------
    >>> a = np.array([np.nan,np.nan,1,np.nan,np.nan,2,np.nan,np.nan,np.nan])
    >>> fnna(a,n=-2)
    
    """
    return _fnna(a, n, axis = axis)


def fnnz(a, n=1, axis = 0):
    """
    returns the index in a of the nth first non-nan and non-zero.
    
    :Parameters:
    ------------
    a : array/timeseries
    n: int, optional, default = 1

    :Example:
    ---------
    >>> a = np.array([[np.nan,0.,1.,np.nan,np.nan,2,np.nan,np.nan,np.nan], [0.,0.,0.,np.nan,np.nan,2,np.nan,np.nan,np.nan]]).T
    >>> #                        ^                                             ^                    ^
    >>> #                      2nd pos                                 zeros dont count      first non-zero

    >>> assert list(fnnz(a)) == [2,5] 
    
    """
    return _fnnz(a, n, axis = axis)


def ewfill(a, fwd_n, bwd_n = None, axis = 0, decay_target = 0.0):
    """
    returns a double-direction exponentially-weighted filled result.
    
    The idea is that we decay value to long term target (say 0). 
    The weight we get from previous observation decays exponentiall (using fwd_n)
    The weight we get from next observation decays exponentiall backwards (using bwd_n)
    The remaining weight is assigned to long term decay_target
    
    :Parameters:
    ------------
    a : array/timeseries
        array/timeseries
    fwd_n: int/float, optional
        speed of fwd filling decay weight
    bwd_n: int/float, optional
        speed of bwd_n filling decay weight
    state: dict, optional
        state parameters used to instantiate the internal calculations, based on history prior to 'a' provided. 

    :Example:
    ---------
    >>> a = np.array([np.nan,np.nan,1,np.nan,np.nan,2,np.nan,np.nan,np.nan])
    >>> fnna(a, n=-2)
    """
    bwd_n = bwd_n or fwd_n
    return _ewfill(a, fwd_n = fwd_n, bwd_n = bwd_n, decay_target = decay_target)



def bfill(a, n = -1, axis = 0):
    """
    equivalent to a.fillna('bfill'). There is no state-aware as this function is forward looking

    :Example:
    -------
    >>> from pyg import *
    >>> a = np.array([np.nan, 1., np.nan])
    >>> b = np.array([1., 1., np.nan])
    >>> assert eq(bfill(a),  b)

    :Example: pd.Series
    -------
    >>> ts = pd.Series(a, drange(-2))
    >>> assert eq(bfill(ts).values, b)
    """
    return _bfill(a, limit = n, axis = axis)


def ffill(a, n=0, axis = 0, data = None, state = None):
    """
    returns a forward filled array, up to n values forward. 
    supports state manegement which is needed if we want only nth

    
    :Parameters:
    ------------
    a : array/timeseries
        array/timeseries
    n: int, optional, default = 1
        window size
    data: None.
        unused at the moment. Allow code such as func(live, **func_(history)) to work
    state: dict, optional
        state parameters used to instantiate the internal calculations, based on history prior to 'a' provided. 

    :Example:
    ---------
    >>> a = np.array([np.nan,np.nan,1,np.nan,np.nan,2,np.nan,np.nan,np.nan])
    >>> fnna(a, n=-2)
    """
    state = state or Dict(prev = None, i = None)
    return first_(_ffill(a, n=n, axis = axis, **state))

def ffill_(a, n=0, axis = 0, instate = None):
    """
    returns a forward filled array, up to n values forward. 
    supports state manegement
    
    """
    state = instate or dict(prev = None, i = None)
    return _data_state(['data', 'prev', 'i'],_ffill(a, n=n, axis = axis, **state))

ffill_.output = ['data', 'state']


def v2na(a, old = 0.0, new = np.nan):
    """
    replaces an old value with a new value (default is nan)

    :Examples:
    --------------
    >>> from pyg import *
    >>> a = np.array([1., np.nan, 1., 0.])
    >>> assert eq(v2na(a), np.array([1., np.nan, 1., np.nan]))
    >>> assert eq(v2na(a,1), np.array([np.nan, np.nan, np.nan, 0]))
    >>> assert eq(v2na(a,1,0), np.array([0., np.nan, 0., 0.]))
    
    :Parameters:
    ----------------
    a : array/timeseries
    old: float
        value to be replaced
    new : float, optional
        new value to be used, The default is np.nan.

    :Returns:
    -------
    array/timeseries

    """
    if is_num(a):
        return new if a == old else a
    return _v2na(a, old = old, new = new)

def na2v(a, new = 0.0):
    """
    replaces a nan with a new value
    
    :Example:
    -------
    >>> from pyg import *
    >>> a = np.array([1., np.nan, 1.])
    >>> assert eq(na2v(a), np.array([1., 0.0, 1.]))
    >>> assert eq(na2v(a,1), np.array([1., 1., 1.]))
    
    :Parameters:
    ----------------
    a : array/timeseries
    new : float, optional
        DESCRIPTION. The default is 0.0.

    :Returns:
    -------
    array/timeseries

    """
    if is_num(a):
        return new if np.isnan(a) else a
    return _na2v(a, new)
    

def init2v(a, n = 0, new = np.nan):
    """
    replaces initial non-nan values with a new value. This is primarily to remove initial results of a volatile fit etc.
    
    :Example:
    -------
    >>> from pyg import *
    >>> a = np.arange(100) * 1.
    >>> assert np.sum(a[np.isnan(init2v(a,10))]) == 0+1+2+3+4+5+6+7+8+9
    
    :Parameters:
    ----------------
    a : array/timeseries
        data to be nanned
    new : float, optional
        DESCRIPTION. The default is np.nan.
    n: int
        number of entries to go to new value

    :Returns:
    -------
    array/timeseries

    """
    if n == 0:
        return a
    return _init2v(a, n, new)


    

def diff(a, n=1, time = None, axis = 0, data = None, state = None):
    """
    equivalent to a.diff(n) in pandas if there are no nans. If there are, we SKIP nans rather than propagate them.

    :Parameters:
    ------------
    a : array/timeseries
        array/timeseries
    n: int, optional, default = 1
        window size
    data: None.
        unused at the moment. Allow code such as func(live, **func_(history)) to work
    state: dict, optional
        state parameters used to instantiate the internal calculations, based on history prior to 'a' provided. 


    :Example: : matching pandas no nan's
    ----------------------------------------------------------
    >>> from pyg import *; import pandas as pd; import numpy as np
    >>> a = pd.Series(np.random.normal(0,1,10000), drange(-9999))
    >>> assert eq(timer(diff, 1000)(a), timer(lambda a, n=1: a.diff(n), 1000)(a))

    :Example: : nan skipping
    ----------------------------------
    >>> a = np.array([1., np.nan, 3., 9.])
    >>> assert eq(diff(a),                      np.array([np.nan, np.nan, 2.0,   6.0]))
    >>> assert eq(pd.Series(a).diff().values,   np.array([np.nan, np.nan, np.nan,6.0]))
    
    """
    if n == 0:
        return a - a
    if is_num(state) and n == 1:
        state = Dict(vec = np.array([state]), i = 0)
    state = state or Dict(vec = None, i = 0, t = np.nan)
    state.vec = _vec(a,state.vec, n, axis=axis)
    return first_(_tdiff1(a, time = time, axis = axis, **state) if n == 1 
                  else _tdiff(a, n = n, time = time, axis = axis, **state))

def diff_(a, n=1, time = None, axis = 0, data = None, instate = None):
    """
    returns a forward filled array, up to n values forward. 
    Equivalent to diff(a,n) but returns the full state. See diff for full details
  
    """
    if n == 0:
        return Dict(data = a - a, state = instate)
    state = instate or Dict(vec = None, i = 0, t = np.nan) 
    state.vec = _vec(a, state['vec'], n, axis=axis)
    return _data_state(['data', 'vec', 'i', 't'], 
                       _tdiff1(a, time = time, axis = axis, **state) if n == 1 
                       else _tdiff(a, n=n, time = time, axis = axis, **state))

diff_.output = ['data', 'state']

def buffer_(a, band, unit = 0.0, data = None, instate = None, rms = None):
    if is_num(instate):
        instate = Dict(pos = instate)
    elif instate is None:
        instate = Dict(pos = 0.0)
    if is_num(band):
        band = np.full(a.shape, band)
    if is_num(rms):
        r = rolling_rms(a, rms)
        band = band * 2 * np.exp(-abs(a)/r)
    return _data_state(['data', 'pos'], _buffer(a = a, band = band, unit = unit, **instate))
        
buffer_.output = ['data', 'state']

def buffer(a, band, unit = 0.0, data = None, state = None, rms = None):
    """
    buffer performs two functions:
        - ensures the result is stated in 'units' so if unit == 1, output is integers
        - slows down the trading to ensure we are always within [a-band, a+band] but does not change values while we are within the band


    Explanation: rounding zone band
    -----------
    Handles buffering when the rounding into units may be significant part of the cost. 
    Issue is: supppose unit = 1 and buffer = 0.1
    
    a = [0.49, 0.51, 0.49], the rounding will switch to [0,1,0] 
    while actually not gaining...
    the rounded time series is always "0.49 from nearest"
    
    buffer is our tolerance of "being away from target" so... 
    we should stay at 0 until switching to 1 benefits us at least 0.1
    
    It means the "no-trade zones" are:
        - when we are within [target-band, target_band] : The "target zone band"
        - when we are withing [0.5 of unit - band, 0.5 unit + band] : "The rounding zone band"
    
    This does mean that if the band is very wide viz a viz unit, we end up not trading at all!
    So we impose rounding_band < min(band, unit/3)
  
  
    :Parameters:
    ------------
    a : array/timeseries
        array/timeseries
    band: float
        band size
    unit: float
        the resulting timeseries will always be whole number of units
    data: None.
        unused at the moment. 
    state: dict, float, optional
        state parameters of last position, used to instantiate the internal calculations, based on history prior to 'a' provided. 
        please provide the variable "pos" if a dict, or just a float, indicating previous position.
    """
    if is_num(state):
        state = Dict(pos = state)
    elif state is None:
        state = Dict(pos = 0.0)
    if is_num(band):
        band = np.full(a.shape, band)
    if is_pd(band) and is_pd(a):
        band = df_reindex(band, a, method = 'ffill')
    if is_num(rms):
        r = rolling_rms(a, rms)
        band = band * 2 * np.exp(-abs(a)/r)
    return first_(_buffer(a = a, band = band, unit = unit, **state))
    
        

        
def shift(a, n=1, axis = 0, data = None, state = None):
    """
    Equivalent to a.shift() with support to arra
    
    :Parameters:
    ------------
    a : array, pd.Series, pd.DataFrame or list/dict of these
        timeseries
    n: int
        size of rolling window
    data: None.
        unused at the moment. Allow code such as func(live, **func_(history)) to work
    state: dict, optional
        state parameters used to instantiate the internal calculations, based on history prior to 'a' provided. 
        
    :Example:
    ---------
    >>> from pyg import *; import pandas as pd; import numpy as np
    >>> a = pd.Series([1.,2,3,4,5], drange(-4))
    >>> assert eq(shift(a), pd.Series([np.nan,1,2,3,4], drange(-4)))
    >>> assert eq(shift(a,2), pd.Series([np.nan,np.nan,1,2,3], drange(-4)))
    >>> assert eq(shift(a,-1), pd.Series([2,3,4,5,np.nan], drange(-4)))

    :Example: np.ndarrays
    ---------------------
    >>> assert eq(shift(a.values), shift(a).values)

    :Example: nan skipping
    ---------------------
    >>> a = pd.Series([1.,2,np.nan,3,4], drange(-4))
    >>> assert eq(shift(a), pd.Series([np.nan,1,np.nan, 2,3], drange(-4)))
    >>> assert eq(a.shift(), pd.Series([np.nan,1,2,np.nan,3], drange(-4))) # the location of the nan changes

    :Example: state management
    --------------------------
    >>> old = a.iloc[:3]
    >>> new = a.iloc[3:]
    >>> old_ts = shift_(old)
    >>> new_ts = shift(new, **old_ts)
    >>> assert eq(new_ts, shift(a).iloc[3:])
    """
    if n == 0 or is_num(a):
        return a
    state = state or Dict(vec = None, i = 0,)
    state.vec = _vec(a, state.vec, n, axis=axis)
    return first_(_shift1(a, state.vec, axis = axis) if n == 1 else _shift(a, n, axis = axis, **state))

def shift_(a, n=1, axis = 0, instate = None):
    """
    Equivalent to shift(a,n) but returns the full state. See shift for full details
  
    """
    if n == 0 or is_num(a):
        return Dict(data = a, state = instate)
    state = instate or Dict(vec = None, i = 0,)
    state.vec = _vec(a, state.vec, n, axis=axis)
    return _data_state(['data', 'vec', 'i'], _shift1(a, vec = state.vec, axis = axis) if n == 1 else _shift(a, n, axis = axis, **state))

shift_.output = ['data', 'state']
        
def ratio(a, n=1, time = None, data = None, state = None, axis = 0):
    """
    Equivalent to a.diff() but in log-space..
    
    :Parameters:
    ------------
    a : array, pd.Series, pd.DataFrame or list/dict of these
        timeseries
    n: int
        size of rolling window

    data: None.
        unused at the moment. Allow code such as func(live, **func_(history)) to work
    state: dict, optional
        state parameters used to instantiate the internal calculations, based on history prior to 'a' provided. 
            
    :Example:
    ---------
    >>> from pyg import *; import pandas as pd; import numpy as np
    >>> a = pd.Series([1.,2,3,4,5], drange(-4))
    >>> assert eq(ratio(a), pd.Series([np.nan, 2, 1.5, 4/3,1.25], drange(-4)))
    >>> assert eq(ratio(a,2), pd.Series([np.nan, np.nan, 3, 2, 5/3], drange(-4)))
    """
    if is_num(a):
        return 1
    state = state or Dict(vec = None, i = 0, t = np.nan)
    state.vec = _vec(a, state.vec, n, axis=axis)
    return first_(_tratio(a, n, axis = axis, time = time, **state))

def ratio_(a, n=1, time = None, axis = 0, data = None, instate = None):        
    state = instate or Dict(vec = None, i = 0, t = np.nan) 
    state.vec = _vec(a, state.vec, n, axis=axis)
    if is_num(a):
        return Dict(data = 1, state = state)
    else:
        return _data_state(['data', 'vec', 'i', 't'], _tratio(a, n, time = time, **state))

ratio_.output = ['data', 'state']

def rolling_tover(a, n = 256, data = None, state = None, interval = None):
    if interval is None:
        if is_pd(a) and len(a) > 2:
            elapsed = a.index[-1] - a.index[0]
            total_years = elapsed.days / 365
            interval = total_years / (len(a) - 1) 
        else:
            interval = 1 / 260
    if state is None:
        state = Dict(j = 0, total_variance = 0, total_trades = 0, positions = None, trades = None)
    state.positions = _vec(a, state.positions, n, 0.)
    state.trades = _vec(a, state.trades, n, 0.)
    return first_(_rolling_tover(a = a , n = n, interval = interval, **state))


def rolling_mean(a, n, time = None, axis = 0, data = None, state = None):
    """
    equivalent to pandas a.rolling(n).mean().
    
    - works with np.arrays
    - handles nan without forward filling.
    - supports state parameters
    
    :Parameters:
    ------------
    a : array, pd.Series, pd.DataFrame or list/dict of these
        timeseries
    n: int
        size of rolling window
    time: a sequence of rising values of time
        passage of time
    axis : int, optional
        0/1/-1. The default is 0.    
    data: None.
        unused at the moment. Allow code such as func(live, **func_(history)) to work
    state: dict, optional
        state parameters used to instantiate the internal calculations, based on history prior to 'a' provided. 
        
    :Example: agreement with pandas
    --------------------------------
    >>> from pyg import *; import pandas as pd; import numpy as np
    >>> a = pd.Series(np.random.normal(0,1,10000), drange(-9999))
    >>> panda = a.rolling(10).mean(); ts = rolling_mean(a,10)
    >>> assert abs(ts-panda).max()<1e-10   

    :Example: nan handling
    ----------------------
    Unlike pandas, timeseries does not include the nans in the rolling calculation: it skips them.
    Since pandas rolling engine does not skip nans, they propagate. 
    In fact, having removed half the data points, rolling(10) will return 99% of nans

    >>> a[a<0.1] = np.nan
    >>> panda = a.rolling(10).mean(); ts = rolling_mean(a,10)
    >>> print('#original:', len(nona(a)), 'timeseries:', len(nona(ts)), 'panda:', len(nona(panda)), 'data points')
    >>> #original: 4534 timeseries: 4525 panda: 6 data points

    :Example: state management
    --------------------------
    One can split the calculation and run old and new data separately.

    >>> old = a.iloc[:5000]        
    >>> new = a.iloc[5000:]    
    >>> ts = rolling_mean(a,10)
    >>> old_ts = rolling_mean_(old,10)
    >>> new_ts = rolling_mean(new, 10, **old_ts)    
    >>> assert eq(new_ts, ts.iloc[5000:])

    :Example: dict/list inputs
    ---------------------------
    >>> assert eq(rolling_mean(dict(x = a, y = a**2),10), dict(x = rolling_mean(a,10), y = rolling_mean(a**2,10)))
    >>> assert eq(rolling_mean([a,a**2],10), [rolling_mean(a,10), rolling_mean(a**2,10)])

    :Example: passage of time
    --------------------------
    >>> a = np.array([1.,2.,3.,4.])
    >>> time = np.array([0,0,1,1]) ## i.e. the first two observations are from day 0 and the next two are from day 1
    >>> rolling_mean(a,2,time)
    >>> array([nan, nan, 2.5, 3. ])  
    
    ## The first two observations are from day 0 so cannot have a mean until we get a second point
    ## The mean is then calculated from last observation in day 0 (i.e. 2) and then 3. and then with 4. since these are again, from same day
            

    """
    state = state or Dict(t0 = 0, t1 = 0., vec = None, i = 0, t = np.nan)
    state.vec = _vec(a, state.vec, n, 0., axis=axis)
    return first_(_trolling_mean(a, n, time = time, denom = n, axis = axis, **state))

def rolling_rms(a, n, time = None, axis = 0, data = None, state = None):
    """
    equivalent to pandas (a**2).rolling(n).mean()**0.5.
    
    - works with np.arrays
    - handles nan without forward filling.
    - supports state parameters
    
    :Parameters:
    ------------
    a : array, pd.Series, pd.DataFrame or list/dict of these
        timeseries
    n: int
        size of rolling window
    time: a sequence of rising values of time
        passage of time
    axis : int, optional
        0/1/-1. The default is 0.    
    data: None.
        unused at the moment. Allow code such as func(live, **func_(history)) to work
    state: dict, optional
        state parameters used to instantiate the internal calculations, based on history prior to 'a' provided. 
        
    :Example: agreement with pandas
    --------------------------------
    >>> from pyg import *; import pandas as pd; import numpy as np
    >>> a = pd.Series(np.random.normal(0,1,10000), drange(-9999))
    >>> panda = (a**2).rolling(10).mean()**0.5; ts = rolling_rms(a,10)
    >>> assert abs(ts-panda).max()<1e-10   

    :Example: nan handling
    ----------------------
    Unlike pandas, timeseries does not include the nans in the rolling calculation: it skips them.
    Since pandas rolling engine does not skip nans, they propagate. 
    In fact, having removed half the data points, rolling(10) will return 99% of nans

    >>> a[a<0.1] = np.nan
    >>> panda = (a**2).rolling(10).mean()**0.5; ts = rolling_rms(a,10)
    >>> print('#original:', len(nona(a)), 'timeseries:', len(nona(ts)), 'panda:', len(nona(panda)), 'data points')
    >>> #original: 4534 timeseries: 4525 panda: 6 data points

    :Example: state management
    --------------------------
    One can split the calculation and run old and new data separately.

    >>> old = a.iloc[:5000]        
    >>> new = a.iloc[5000:]    
    >>> ts = rolling_rms(a,10)
    >>> old_ts = rolling_rms_(old,10)
    >>> new_ts = rolling_rms(new, 10, **old_ts)    
    >>> assert eq(new_ts, ts.iloc[5000:])

    :Example: dict/list inputs
    ---------------------------
    >>> assert eq(rolling_rms(dict(x = a, y = a**2),10), dict(x = rolling_rms(a,10), y = rolling_rms(a**2,10)))
    >>> assert eq(rolling_rms([a,a**2],10), [rolling_rms(a,10), rolling_rms(a**2,10)])

    :Example: passage of time
    --------------------------
    >>> a = np.array([1.,2.,3.,4.])
    >>> time = np.array([0,0,1,1]) ## i.e. the first two observations are from day 0 and the next two are from day 1
    >>> (rolling_rms(a,2,time) ** 2) * 2
    >>> array([nan, nan, 13., 20.])   == array([nan, nan, 4 + 9, 4 + 16])
    
    ## The first two observations are from day 0 so cannot have a mean until we get a second point
    ## The rms is then calculated from last observation in day 0 (i.e. 2) and then 3. and then with 4. since these are again, from same day
            

    """
    state = state or Dict(t0 = 0, t2 = 0., vec = None, i = 0, t = np.nan)
    state.vec = _vec(a, state.vec, n, 0., axis=axis)
    return first_(_trolling_rms(a, n, time = time, denom = n, axis = axis, **state))

def rolling_sum(a, n, time = None, axis = 0, data = None, state = None):
    """
    equivalent to pandas a.rolling(n).sum().
    
    - works with np.arrays
    - handles nan without forward filling.
    - supports state parameters
    
    :Parameters:
    ------------
    a : array, pd.Series, pd.DataFrame or list/dict of these
        timeseries
    n: int
        size of rolling window
    time: a sequence of rising values of time
        passage of time
    axis : int, optional
        0/1/-1. The default is 0.    
    data: None.
        unused at the moment. Allow code such as func(live, **func_(history)) to work
    state: dict, optional
        state parameters used to instantiate the internal calculations, based on history prior to 'a' provided. 
        
    :Example: agreement with pandas
    --------------------------------
    >>> from pyg import *; import pandas as pd; import numpy as np
    >>> a = pd.Series(np.random.normal(0,1,10000), drange(-9999))
    >>> panda = a.rolling(10).sum(); ts = rolling_sum(a,10)
    >>> assert abs(ts-panda).max()<1e-10   

    :Example: nan handling
    ----------------------
    Unlike pandas, timeseries does not include the nans in the rolling calculation: it skips them.
    Since pandas rolling engine does not skip nans, they propagate. 
    In fact, having removed half the data points, rolling(10) will return 99.9% nans

    >>> a[a<0.1] = np.nan
    >>> panda = a.rolling(10).sum(); ts = rolling_sum(a,10)
    >>> print('#original:', len(nona(a)), 'timeseries:', len(nona(ts)), 'panda:', len(nona(panda)), 'data points')
    >>> #original: 4534 timeseries: 4525 panda: 2 data points

    :Example: state management
    --------------------------
    One can split the calculation and run old and new data separately.

    >>> old = a.iloc[:5000]        
    >>> new = a.iloc[5000:]    
    >>> ts = rolling_sum(a,10)
    >>> old_ts = rolling_sum_(old,10)
    >>> new_ts = rolling_sum(new, 10, **old_ts)    
    >>> assert eq(new_ts, ts.iloc[5000:])

    :Example: dict/list inputs
    ---------------------------
    >>> assert eq(rolling_sum(dict(x = a, y = a**2),10), dict(x = rolling_sum(a,10), y = rolling_sum(a**2,10)))
    >>> assert eq(rolling_sum([a,a**2],10), [rolling_sum(a,10), rolling_sum(a**2,10)])

    :Example: passage of time
    --------------------------
    >>> a = np.array([1.,2.,3.,4.])
    >>> time = np.array([0,0,1,1]) ## i.e. the first two observations are from day 0 and the next two are from day 1
    >>> rolling_sum(a,2,time) 
    >>> array([nan, nan, 5., 6.])   == array([nan, nan, 2+3, 2+4])
    
    ## The first two observations are from day 0 so cannot have a mean until we get a second point
    ## The sum is then calculated from last observation in day 0 (i.e. 2) and then 3. and then with 4. since these are again, from same day
    """
    state = state or Dict(t0 = 0, t1 = 0., vec = None, i = 0, t = np.nan)
    state.vec = _vec(a, state.vec, n, 0., axis=axis)
    return first_(_trolling_mean(a, n, time = time, denom = 1, axis = axis, **state))

def rolling_std(a, n, time = None, axis = 0, data = None, state = None):
    """
    equivalent to pandas a.rolling(n).std().
    
    - works with np.arrays
    - handles nan without forward filling.
    - supports state parameters
    
    :Parameters:
    ------------
    a : array, pd.Series, pd.DataFrame or list/dict of these
        timeseries
    n: int
        size of rolling window
    time: a sequence of rising values of time
        passage of time
    axis : int, optional
        0/1/-1. The default is 0.    
    data: None.
        unused at the moment. Allow code such as func(live, **func_(history)) to work
    state: dict, optional
        state parameters used to instantiate the internal calculations, based on history prior to 'a' provided. 
        
    :Example: agreement with pandas
    --------------------------------
    >>> from pyg import *; import pandas as pd; import numpy as np
    >>> a = pd.Series(np.random.normal(0,1,10000), drange(-9999))
    >>> panda = a.rolling(10).std(); ts = rolling_std(a,10)
    >>> assert abs(ts-panda).max()<1e-10   

    :Example: nan handling
    ----------------------
    Unlike pandas, timeseries does not include the nans in the rolling calculation: it skips them.
    Since pandas rolling engine does not skip nans, they propagate. 
    In fact, having removed half the data points, rolling(10) will return 99.9% nans

    >>> a[a<0.1] = np.nan
    >>> panda = a.rolling(10).std(); ts = rolling_std(a,10)
    >>> print('#original:', len(nona(a)), 'timeseries:', len(nona(ts)), 'panda:', len(nona(panda)), 'data points')
    >>> #original: 4534 timeseries: 4525 panda: 2 data points

    :Example: state management
    --------------------------
    One can split the calculation and run old and new data separately.

    >>> old = a.iloc[:5000]        
    >>> new = a.iloc[5000:]    
    >>> ts = rolling_std(a,10)
    >>> old_ts = rolling_std_(old,10)
    >>> new_ts = rolling_std(new, 10, **old_ts)    
    >>> assert eq(new_ts, ts.iloc[5000:])

    :Example: dict/list inputs
    ---------------------------
    >>> assert eq(rolling_std(dict(x = a, y = a**2),10), dict(x = rolling_std(a,10), y = rolling_std(a**2,10)))
    >>> assert eq(rolling_std([a,a**2],10), [rolling_std(a,10), rolling_std(a**2,10)])

    :Example: passage of time
    --------------------------
    >>> a = np.array([1.,2.,3.,4.])
    >>> time = np.array([0,0,1,1]) ## i.e. the first two observations are from day 0 and the next two are from day 1
    >>> 2*(rolling_std(a,2,time) ** 2)
    >>> array([nan, nan, 1., 4.])
    
    ## The first two observations are from day 0 so cannot have a mean until we get a second point
    ## The sum is then calculated from last observation in day 0 (i.e. 2) and then 3. and then with 4. since these are again, from same day
    """    
    state = state or Dict(t0 = 0, t1 = 0, t2 = 0., vec = None, i = 0, t = np.nan)
    state.vec = _vec(a, state.vec, n, 0., axis=axis)
    return first_(_trolling_std(a, n, time = time, denom = n, axis = axis, **state))

def rolling_skew(a, n, bias = False, time = None, axis = 0, data = None, state = None):
    """
    equivalent to pandas a.rolling(n).skew().
    
    - works with np.arrays
    - handles nan without forward filling.
    - supports state parameters
    
    :Parameters:
    ------------
    a : array, pd.Series, pd.DataFrame or list/dict of these
        timeseries
    n: int
        size of rolling window
    time: a sequence of rising values of time
        passage of time
    bias: 
        affects the skew calculation definition, see scipy documentation for details.
    axis : int, optional
        0/1/-1. The default is 0.    
    data: None.
        unused at the moment. Allow code such as func(live, **func_(history)) to work
    state: dict, optional
        state parameters used to instantiate the internal calculations, based on history prior to 'a' provided. 
        
    :Example: agreement with pandas
    --------------------------------
    >>> from pyg import *; import pandas as pd; import numpy as np
    >>> a = pd.Series(np.random.normal(0,1,10000), drange(-9999))
    >>> panda = a.rolling(10).skew(); ts = rolling_skew(a,10)
    >>> assert abs(ts-panda).max()<1e-10   

    :Example: nan handling
    ----------------------
    Unlike pandas, timeseries does not include the nans in the rolling calculation: it skips them.
    Since pandas rolling engine does not skip nans, they propagate. 
    In fact, having removed half the data points, rolling(10) will return 99.9% nans

    >>> a[a<0.1] = np.nan
    >>> panda = a.rolling(10).skew(); ts = rolling_skew(a,10)
    >>> print('#original:', len(nona(a)), 'timeseries:', len(nona(ts)), 'panda:', len(nona(panda)), 'data points')
    >>> #original: 4534 timeseries: 4525 panda: 2 data points

    :Example: state management
    --------------------------
    One can split the calculation and run old and new data separately.

    >>> old = a.iloc[:5000]        
    >>> new = a.iloc[5000:]    
    >>> ts = rolling_skew(a,10)
    >>> old_ts = rolling_skew_(old,10)
    >>> new_ts = rolling_skew(new, 10, **old_ts)    
    >>> assert eq(new_ts, ts.iloc[5000:])

    :Example: dict/list inputs
    ---------------------------
    >>> assert eq(rolling_skew(dict(x = a, y = a**2),10), dict(x = rolling_skew(a,10), y = rolling_skew(a**2,10)))
    >>> assert eq(rolling_skew([a,a**2],10), [rolling_skew(a,10), rolling_skew(a**2,10)])
    """
    state = state or Dict(t0 = 0, t1 = 0, t2 = 0., t3 = 0, vec = None, i = 0, t = np.nan)
    state.vec = _vec(a, state.vec, n, 0., axis=axis)
    return first_(_trolling_skew(a, n, time = time, bias = bias, denom = n, axis = axis, **state))


def rolling_mean_(a, n, time = None, axis = 0, data = None, instate = None):
    """
    Equivalent to rolling_mean(a) but returns also the state variables t0,t1 etc. 
    For full documentation, look at rolling_mean.__doc__
    """
    state = instate or Dict(t0 = 0, t1 = 0., vec = None, i = 0, t = np.nan)
    state.vec = _vec(a, state.vec, n, 0., axis=axis)
    return _data_state(['data','t0','t1', 'vec','i', 't'],_trolling_mean(a, n, time = time, denom = n, axis = axis, **state))

rolling_mean_.output = ['data','state']

def rolling_rms_(a, n, time = None, axis = 0, data = None, instate = None):
    """
    Equivalent to rolling_rms(a) but returns also the state variables t0,t1 etc. 
    For full documentation, look at rolling_rms.__doc__
    """
    state = instate or Dict(t0 = 0, t2 = 0., vec = None, i = 0, t = np.nan)
    state.vec = _vec(a, state.vec, n, 0., axis=axis)
    return _data_state(['data','t0','t2', 'vec','i', 't'],_trolling_rms(a, n, time = time, denom = n, axis = axis, **state))

rolling_rms_.output = ['data','state']

def rolling_sum_(a, n, time = None, axis = 0, data = None, instate = None):
    """
    Equivalent to rolling_sum(a) but returns also the state variables t0,t1 etc. 
    For full documentation, look at rolling_sum.__doc__
    """
    state = instate or Dict(t0 = 0, t1 = 0., vec = None, i = 0, t = np.nan)
    state.vec = _vec(a, state.vec, n, 0., axis=axis)
    return _data_state(['data','t0','t1', 'vec','i', 't'], _trolling_mean(a, n, denom = 1, axis = axis, **state))

rolling_sum_.output = ['data','state']

def rolling_std_(a, n, time = None, axis = 0, data = None, instate = None):
    """
    Equivalent to rolling_std(a) but returns also the state variables t0,t1 etc. 
    For full documentation, look at rolling_std.__doc__
    """
    state = instate or Dict(t0 = 0, t1 = 0, t2 = 0., vec = None, i = 0, t = np.nan)
    state.vec = _vec(a, state.vec, n, 0., axis=axis)
    return _data_state(['data','t0', 't1', 't2', 'vec', 'i', 't'],_trolling_std(a, n, time = time, denom = n, axis = axis, **state))


rolling_std_.output = ['data','state']

def rolling_skew_(a, n, time = None, bias = False, axis = 0, data = None, instate = None):
    """
    Equivalent to rolling_skew(a) but returns also the state variables t0,t1 etc. 
    For full documentation, look at rolling_skew.__doc__
    """
    state = instate or Dict(t0 = 0, t1 = 0, t2 = 0., t3 = 0, vec = None, i = 0, t = np.nan)
    state.vec = _vec(a, state.vec, n, 0., axis=axis)
    return _data_state(['data','t0', 't1', 't2', 't3', 'vec','i', 't'], _trolling_skew(a, n, time = time, bias = bias, denom = n, axis = axis, **state))

rolling_skew_.output = ['data','state']
