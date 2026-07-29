import numpy as np
from pyg_timeseries._rolling import _vec
from pyg_timeseries._decorators import _data_state, first_
from pyg_base import pd2np, as_list, loop_all, loop, is_pd, is_num, as_series, is_df


def _as_strided(a, L, S=1):
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))


def _trim_state(values, n):
    """
    the state only ever needs the last n-1 observations. Trimming also protects against a state that
    was built with a longer window, so len(state) < n always holds.
    """
    return values[len(values) - n + 1:] if len(values) >= n else values


def _strided_results(history, n_vec, t, n, min_periods, width, function, function_parameters):
    """
    applies function to a rolling window of the last n values of history, returning one result per
    element of history beyond the leading n_vec (which are the state carried in from a previous call).

    history[:n_vec] is the state and history[n_vec:] this call's observations. The i'th result is
    computed off the last min(n, n_vec + i + 1) values of the history up to it: everything available
    while the window is still filling up (the warm-up ramp) and the last n values thereafter.

    t is the number of observations seen before this call, so the i'th result is observation t + i + 1
    and is only emitted once that reaches min_periods. t exceeds n_vec whenever the history is longer
    than the n-1 values the state keeps, which matters when min_periods > n.
    """
    n_res = len(history) - n_vec
    res = np.full((n_res,) if width == 0 else (n_res, width), np.nan)
    window = np.minimum(n, n_vec + 1 + np.arange(n_res))    ## values available at each result point
    full_window = window == n
    if full_window.any():
        res[full_window] = function(_as_strided(history, n), axis=1, **function_parameters).T
    for i in np.where(~full_window)[0]:
        res[i] = function(history[:window[i]], **function_parameters)
    res[t + 1 + np.arange(n_res) < min_periods] = np.nan
    return res


@loop_all
@pd2np
def _rolling_stride(a, n, vec=None, t=None, min_periods=None, width = 0, function = np.quantile,
                    **function_parameters):
    """
    nan-skipping rolling application of function to a window of the last n VALID observations of a.
    Returns (data, vec, t) where vec is the last n-1 valid observations and t the number of valid
    observations seen, so that resuming from a state reproduces the whole-series result.

    >>> quantile = [0.1, 0.2]; n = 100; a = np.arange(1000) * 1.
    >>> a[np.random.normal(0,1,1000) > 1.5] = np.nan
    >>> min_periods = 10
    >>> _rolling_stride(a, n, q = np.array(quantile), width = 2, min_periods = min_periods)
    >>> rolling_quantile(a, n, quantile, min_periods = min_periods)
    >>> rolling_quantile_(a, n, quantile, min_periods = min_periods)
    """
    n = abs(n)
    min_periods = n if min_periods is None else min_periods
    vec = _trim_state(_vec(a, vec, 0), n)
    t = len(vec) if t is None else t
    mask = ~np.isnan(a)
    history = np.concatenate([vec, a[mask]])    ## previous valid observations followed by a's own
    data = np.full(a.shape if width == 0 else (a.shape[0], width), np.nan)
    data[mask] = _strided_results(history, len(vec), t, n, min_periods, width, function, function_parameters)
    return data, _trim_state(history, n), t + len(history) - len(vec)


def _quantile_width(a, quantile):
    """
    the number of columns _rolling_stride must produce per column of a: 0 for a single quantile (the
    result has the shape of a) and one column per quantile otherwise
    """
    if len(getattr(a, 'shape', [])) == 2 and a.shape[1] > 1 and len(as_list(quantile)) > 1:
        raise ValueError('Can do multiple quantiles %s only for single-column data'%quantile)
    return 0 if is_num(quantile) else len(as_list(quantile))


def _cast_quantile_result(res, a, quantile):
    """
    a single quantile of a 1d input is cast back to a series/1d array; multiple quantiles are labelled
    with the quantile they represent
    """
    qs = as_list(quantile)
    @loop(list, dict, tuple)
    def cast(res):
        if is_num(quantile) and len(a.shape) == 1:  ## cast back to a series
            return as_series(res) if is_df(res) else res[:, 0] if isinstance(res, np.ndarray) and len(res.shape) == 2 else res
        if is_pd(res) and len(res.shape) == 2 and res.shape[1] == len(qs):
            res.columns = qs
        return res
    return cast(res)


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
    width = _quantile_width(a, quantile)
    res = first_(_rolling_stride(a, n=n,
                                 q=np.array(quantile),
                                 width = width,
                                 axis=axis,
                                 min_periods=min_periods,
                                 method = interpolation , **(state or {})))
    return _cast_quantile_result(res, a, quantile)


def rolling_quantile_(a, n, quantile=0.5, axis=0, min_periods=None, data=None, instate=None, interpolation = 'linear'):
    """
    Equivalent to rolling_quantile(a) but returns also the state.
    For full documentation, look at rolling_quantile.__doc__
    """
    width = _quantile_width(a, quantile)
    res = _data_state(["data", "vec", "t"],_rolling_stride(a, n=n,
                                                      q = np.array(quantile),
                                                      width = width,
                                                      min_periods=min_periods,
                                                      axis=axis,
                                                      function = np.quantile,
                                                      method = interpolation , **(instate or {})))
    return _cast_quantile_result(res, a, quantile)


rolling_quantile_.output = ["data", "state"]
