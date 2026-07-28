import numpy as np
from pyg_timeseries._rolling import _fnna, _vec
from pyg_timeseries._decorators import _data_state, first_
from pyg_base import pd2np, as_list, loop_all, loop, is_pd, is_num, as_series, is_df


def _as_strided(a, L, S=1):
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))


def _cast_strided_result(a, va, res):
    """
    casts the values from res, to the last values of a that are not nan
    """
    if len(res):
        mask = ~np.isnan(a)
        msk = mask[mask]
        msk[: -len(res)] = False
        mask[mask] = msk
        va[mask] = res
    return va


@loop_all
@pd2np
def _rolling_stride(a, n, vec=None, min_periods=None, width = 0, function = np.quantile, **function_parameters):
    """
    >>> vec = None; quantile = [0.1, 0.2]; n = 100; a = np.arange(1000) * 1.
    >>> a[np.random.normal(0,1,1000) > 1.5] = np.nan
    >>> min_periods = 10
    >>> _rolling_quantile(a, n, quantile, vec = None, min_periods = min_periods)
    >>> rolling_quantile(a, n, quantile, min_periods = min_periods)
    >>> rolling_quantile_(a, n, quantile, min_periods = min_periods)
    """
    vec = _vec(a, vec, 0)
    mask = ~np.isnan(a)
    na = a[mask]
    if len(vec):
        na = np.concatenate([vec,na])
    else:
        na = a
    va = np.full(a.shape, np.nan) if width == 0 else np.full((a.shape[0], width), np.nan)
    strided = _as_strided(na, abs(n), 1)
    res = function(strided, axis=1, **function_parameters).T
    if min_periods is not None:
        if min_periods > n:
            res[:min_periods-n] = np.nan
        else:
            n_ = min(n, len(res))-1
            min_periods_ = min_periods + len(vec)        
            if n_ > min_periods_:
                initial = np.array([function(na[:i], **function_parameters) for i in range(min_periods_, n_)])
                res = np.concatenate([initial, res])        
    rtn = _cast_strided_result(a, va, res)
    return rtn, na[-(n-1):]


def rolling_quantile(
    a, n, quantile=0.5, axis=0, data=None, state=None, min_periods=None, interpolation  = 'linear',
):
    """
    equivalent to a.rolling(n).quantile(q) except...
    - supports numpy arrays
    - supports multiple q values

    :Example:
    -------
    >>> from pyg import *; import pandas as pd; import numpy as np
    >>> a = pd.Series(np.random.normal(0,1,10000), drange(-9999)); n = 100; quantile = 0.3; state = None
    >>> res = rolling_quantile(a, 100, 0.3)
    >>> res
    Out[128]:
    1999-03-07         NaN
    1999-03-08         NaN
    1999-03-09         NaN
    1999-03-10         NaN
    1999-03-11         NaN

    2026-07-18   -0.628205
    2026-07-19   -0.671164
    2026-07-20   -0.628205
    2026-07-21   -0.615389
    2026-07-22   -0.611659

    >>> res = rolling_quantile(a, n=100, quantile = 0.3, min_periods = 0)
    >>> res
    1999-03-07   -0.506515
    1999-03-08   -0.176311
    1999-03-09   -0.313078
    1999-03-10   -0.216360
    1999-03-11   -0.061241

    2026-07-18   -0.587122
    2026-07-19   -0.602696
    2026-07-20   -0.602696
    2026-07-21   -0.626753
    2026-07-22   -0.626753

    >>> a.rolling(100, min_periods=0).quantile(0.3)
    1999-03-07   -0.506515
    1999-03-08   -0.176311
    1999-03-09   -0.313078
    1999-03-10   -0.216360
    1999-03-11   -0.061241

    2026-07-18   -0.587122
    2026-07-19   -0.602696
    2026-07-20   -0.602696
    2026-07-21   -0.626753
    2026-07-22   -0.626753

    :Example: multiple quantiles
    ---------------------------------------------
    >>> res = rolling_quantile(a, 100, [0.3, 0.5, 0.75])
    >>> assert abs(res[0.3] - a.rolling(100).quantile(0.3)).max() < 1e-13

    :Example: state management
    ---------------------------------------------
    >>> res = rolling_quantile(a, 100, 0.3)
    >>> old = rolling_quantile_(a.iloc[:2000], 100, 0.3)
    >>> new = rolling_quantile(a.iloc[2000:], 100, 0.3, **old)
    >>> both = pd.concat([old.data, new])
    >>> assert eq(both, res)

    Example:
    ----
    a = pd.Series(np.random.normal(0,1,300), drange(-299))
    rolling_quantile(a, 500, quantile = [0.1, 0.2], min_periods = 100)



    :Parameters:
    ----------------
    a : array/timeseries
    n : integer
        window size.
    q : float or list of floats in [0,1]
        quantile(s).
    data: None.
        unused at the moment. Allow code such as func(live, **func_(history)) to work
    state: dict, optional
        state parameters used to instantiate the internal calculations, based on history prior to 'a' provided.

    :Returns:
    -------
    timeseries/array of quantile(s)

    """
    if len(getattr(a, 'shape', [])) == 2 and a.shape[1] > 1 and len(as_list(quantile)) > 1:
        raise ValueError('Can do multiple quantiles %s only for single-column data'%quantile)
    state = state or {}
    qs = as_list(quantile)
    res = first_(_rolling_stride(a, n=n, q=np.array(quantile), width = 0 if is_num(quantile) else len(qs), axis=axis, min_periods=min_periods, method = interpolation ,**state))
    if is_num(quantile) and len(a.shape) == 1:  ## cast back to a series
        @loop(list, dict, tuple)
        def add_qs(res):
            return as_series(res) if is_df(res) else res[:, 0] if isinstance(res, np.ndarray) and len(res.shape) == 2 else res
    else:
        @loop(list, dict, tuple)
        def add_qs(res):
            if is_pd(res) and len(res.shape) == 2 and res.shape[1] == len(as_list(qs)):
                res.columns = as_list(qs)
            return res
    return add_qs(res)


def rolling_quantile_(a, n, quantile=0.5, axis=0, min_periods=None, data=None, instate=None, interpolation = 'linear'):
    """
    Equivalent to rolling_quantile(a) but returns also the state.
    For full documentation, look at rolling_quantile.__doc__
    """
    if len(getattr(a, 'shape', [])) == 2 and a.shape[1] > 1 and len(as_list(quantile)) > 1:
        raise ValueError('Can do multiple quantiles %s only for single-column data'%quantile)
    state = instate  or {}
    qs = as_list(quantile)
    res = _data_state(["data", "vec"],_rolling_stride(a, n=n, q=np.array(quantile), width = 0 if is_num(quantile) else len(qs), min_periods=min_periods, axis=axis, method = interpolation , **state))
    if is_num(quantile) and len(a.shape) == 1:  ## cast back to a series
        @loop(list, dict, tuple)
        def add_qs(res):
            return as_series(res) if is_df(res) else res[:, 0] if isinstance(res, np.ndarray) and len(res.shape) == 2 else res
    else:
        @loop(list, dict, tuple)
        def add_qs(res):
            if is_pd(res) and len(res.shape) == 2 and res.shape[1] == len(as_list(qs)):
                res.columns = as_list(qs)
            return res
    return add_qs(res)


rolling_quantile_.ouput = ["data", "state"]
