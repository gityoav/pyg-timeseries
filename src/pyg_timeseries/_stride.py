from typing import Any, Callable, Dict

import numpy as np
from numba import njit
from pyg_base import pd2np, as_list, loop_all, loop, is_pd, is_num, as_series, is_df
from pyg_timeseries._decorators import _data_state, first_
from pyg_timeseries._rolling import _vec


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
    res = np.full((n_res,) if width == 0 else (n_res, width), np.nan)1
    window = np.minimum(n, n_vec + 1 + np.arange(n_res))    ## values available at each result point
    full_window = window == n
    if full_window.any():
        res[full_window] = function(_as_strided(history, n), axis=1, **function_parameters).T
    for i in np.where(~full_window)[0]:
        res[i] = function(history[:window[i]], **function_parameters)
    res[t + 1 + np.arange(n_res) < min_periods] = np.nan
    return res


def _is_moving_quantile(function: Callable, function_parameters: Dict[str, Any]) -> bool:
    """
    whether _moving_quantile can stand in for function: a plain np.quantile, linearly interpolated and
    with nothing else passed to it. np.quantile's other interpolation methods stay on the strided path.
    """
    return (function is np.quantile
            and not set(function_parameters) - {"q", "method"}
            and function_parameters.get("method", "linear") == "linear")


@njit(cache=True)
def _sift_up(heap: np.ndarray, values: np.ndarray, where: np.ndarray, sign: float, j: int) -> int:
    """
    moves heap[j] towards the root until its parent is no greater, returning the index it settles at.

    heap holds slots into values and is a min-heap on sign * value, so the lower heap passes sign = -1
    and keeps its LARGEST value at the root. where[slot] is the inverse of heap, tracking each slot's
    position so that a value replaced in place can be sifted without searching for it.
    """
    slot = heap[j]
    key = sign * values[slot]
    while j > 0:
        parent = (j - 1) // 2
        parent_slot = heap[parent]
        if sign * values[parent_slot] <= key:
            break
        heap[j] = parent_slot
        where[parent_slot] = j
        j = parent
    heap[j] = slot
    where[slot] = j
    return j


@njit(cache=True)
def _sift_down(
        heap: np.ndarray,
        size: int,
        values: np.ndarray,
        where: np.ndarray,
        sign: float,
        j: int,
) -> int:
    """
    moves heap[j] away from the root until both its children are no smaller, returning its final index
    """
    slot = heap[j]
    key = sign * values[slot]
    while True:
        child = 2 * j + 1
        if child >= size:
            break
        if child + 1 < size and sign * values[heap[child + 1]] < sign * values[heap[child]]:
            child += 1
        child_slot = heap[child]
        if key <= sign * values[child_slot]:
            break
        heap[j] = child_slot
        where[child_slot] = j
        j = child
    heap[j] = slot
    where[slot] = j
    return j


@njit(cache=True)
def _heap_moving_quantile(history: np.ndarray, n: int, quantile: float) -> np.ndarray:
    """
    the quantile of every trailing window of up to n values of history, one result per element.

    The window is split across two heaps - a lower one keeping its largest value at the root and an
    upper one keeping its smallest there - rebalanced after every step so the lower heap holds exactly
    lo + 1 values, where lo = floor(quantile * (m - 1)) over a window of m values. The two roots are
    then the order statistics either side of the quantile, interpolated exactly as np.quantile's
    'linear' method does. Insert, evict and rebalance are all O(log n), so the whole series costs
    O(len(history) * log n) against the O(len(history) * n * log n) of sorting each window separately.

    history must not contain NaN (_rolling_stride masks them out before calling).
    """
    out = np.empty(len(history), dtype=np.float64)
    values = np.empty(n, dtype=np.float64)       ## the window, by arrival slot: slot i % n is oldest
    where = np.empty(n, dtype=np.int64)          ## each slot's index within whichever heap holds it
    lower = np.empty(n, dtype=np.int64)          ## slots, min-heap on -value, so root = largest
    upper = np.empty(n, dtype=np.int64)          ## slots, min-heap on +value, so root = smallest
    in_lower = np.empty(n, dtype=np.bool_)
    n_lower = 0
    n_upper = 0
    for i in range(len(history)):
        value = history[i]
        if i < n:                                ## the window is still filling, so take a fresh slot
            slot = i
            values[slot] = value
            if n_lower == 0 or value <= values[lower[0]]:
                in_lower[slot] = True
                lower[n_lower] = slot
                where[slot] = n_lower
                n_lower += 1
                _sift_up(lower, values, where, -1.0, n_lower - 1)
            else:
                in_lower[slot] = False
                upper[n_upper] = slot
                where[slot] = n_upper
                n_upper += 1
                _sift_up(upper, values, where, 1.0, n_upper - 1)
            m = i + 1
        else:                                    ## overwrite the slot leaving the window in place
            slot = i % n
            values[slot] = value
            if in_lower[slot]:
                j = _sift_up(lower, values, where, -1.0, where[slot])
                _sift_down(lower, n_lower, values, where, -1.0, j)
            else:
                j = _sift_up(upper, values, where, 1.0, where[slot])
                _sift_down(upper, n_upper, values, where, 1.0, j)
            ## each heap is valid internally now, but the replacement may have pushed a value past the
            ## other heap's root; exchanging the two roots restores max(lower) <= min(upper)
            if n_lower > 0 and n_upper > 0 and values[lower[0]] > values[upper[0]]:
                was_lower = lower[0]
                was_upper = upper[0]
                lower[0] = was_upper
                where[was_upper] = 0
                in_lower[was_upper] = True
                upper[0] = was_lower
                where[was_lower] = 0
                in_lower[was_lower] = False
                _sift_down(lower, n_lower, values, where, -1.0, 0)
                _sift_down(upper, n_upper, values, where, 1.0, 0)
            m = n
        position = quantile * (m - 1)
        lo = int(np.floor(position))
        while n_lower < lo + 1:                  ## promote min(upper) into the lower heap
            promoted = upper[0]
            n_upper -= 1
            if n_upper > 0:
                upper[0] = upper[n_upper]
                where[upper[0]] = 0
                _sift_down(upper, n_upper, values, where, 1.0, 0)
            lower[n_lower] = promoted
            where[promoted] = n_lower
            in_lower[promoted] = True
            n_lower += 1
            _sift_up(lower, values, where, -1.0, n_lower - 1)
        while n_lower > lo + 1:                  ## demote max(lower) into the upper heap
            demoted = lower[0]
            n_lower -= 1
            if n_lower > 0:
                lower[0] = lower[n_lower]
                where[lower[0]] = 0
                _sift_down(lower, n_lower, values, where, -1.0, 0)
            upper[n_upper] = demoted
            where[demoted] = n_upper
            in_lower[demoted] = False
            n_upper += 1
            _sift_up(upper, values, where, 1.0, n_upper - 1)
        below = values[lower[0]]
        if position == lo:
            out[i] = below
        else:
            above = values[upper[0]]
            gamma = position - lo
            ## np.quantile's linear method lerps from whichever end is nearer, which costs it a
            ## different rounding than a plain below + gamma * (above - below); match it exactly so
            ## results stay identical to the last bit rather than merely to a tolerance
            if gamma < 0.5:
                out[i] = below + (above - below) * gamma
            else:
                out[i] = above - (above - below) * (1.0 - gamma)
    return out


def _moving_quantile(history: np.ndarray, n: int, quantile: float) -> np.ndarray:
    """
    the quantile of every trailing window of up to n values of history, one result per element: the
    leading ones are the warm-up ramp over however many values are available, matching
    np.quantile(history[:i + 1]) there and np.quantile(history[i + 1 - n:i + 1]) once the window fills.

    A window longer than the history never fills, so capping it keeps the heaps no larger than the data.
    """
    history = np.asarray(history, dtype=np.float64)
    return _heap_moving_quantile(history, min(n, len(history)), quantile)


def _moving_quantile_results(
        history: np.ndarray,
        n_vec: int,
        t: int,
        n: int,
        min_periods: int,
        width: int,
        function: Callable,
        function_parameters: Dict[str, Any],
) -> np.ndarray:
    """
    _strided_results specialised to np.quantile, computing the same results incrementally rather than
    by sorting each window. Interchangeable with it: same arguments, same shape, same values, which is
    why it takes the function it has already been chosen to replace and ignores it.
    """
    quantiles = np.ravel(function_parameters["q"])
    columns = [_moving_quantile(history, n, q)[n_vec:] for q in quantiles]
    res = columns[0] if width == 0 else np.column_stack(columns)
    res[t + 1 + np.arange(len(res)) < min_periods] = np.nan
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
    results = _moving_quantile_results if _is_moving_quantile(function, function_parameters) else _strided_results
    data[mask] = results(history, len(vec), t, n, min_periods, width, function, function_parameters)
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
