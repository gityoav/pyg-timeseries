from pyg_base import pd2np, is_num, is_rng, is_pd, loop, is_arr, df_reindex
from pyg_timeseries._rolling import fnna, na2v
import pandas as pd
import numpy as np

@loop(list)
@pd2np
def _fnna_like(source, target = 1., default = np.nan):
    """
    takes a target value, list or an array and match source shape. Then matches the initial nan pattern

    Parameters
    ----------
    source : dataframe
        the source to copy
    target : value/list/array, optional
        The source we want to fill with nan when source is nan
    default : float/int, optional
        The value before nan

    Returns
    -------
    a : array matching shape of source

    :Example:
    --------
    
    >>> source = np.array([[np.nan, 1.], [1.,1.]])

    >>> fnna_like(source, 2)
    >>> array([[nan,  2.],
    >>>        [ 2.,  2.]])

    >>> fnna_like(source, [4,5])
    >>> array([[nan,  5.],
    >>>        [ 4.,  5.]])

    >>> import pandas as pd
    >>> source = pd.DataFrame(np.array([[np.nan, np.nan, 1.], [np.nan,2., 3.], [np.nan, np.nan, np.nan, ]]), [1,2,3], ['a', 'b', 'c'])

    >>> fnna_like(source, 1)
    >>>     a    b    c
    >>> 1 NaN  NaN  1.0
    >>> 2 NaN  1.0  1.0
    >>> 3 NaN  1.0  1.0

    """
    f = fnna(source)
    if is_num(target) or is_rng(target) or (is_arr(target) and len(target.shape) == 1 and target.shape[0] == source.shape[1]):
        a = np.full(source.shape, fill_value = target)
    else:
        a = target.copy()
    a = a.astype(type(default))
    if len(source.shape) == 1:
        a[:f] = default
    else:
        for i, j in enumerate(f):
            a[:j,i] = default
    return a

def fnna_like(source, target = 1., default = np.nan):
    if is_pd(source) and is_pd(target):
        target = target.reindex(source.index)
    return _fnna_like(source, target, default)


def reindex_3d(array, index, original_index, method = None, limit = None):
    if len(array.shape)<3:
        if isinstance(array, np.ndarray):
            ts = pd.Series(array, index = original_index) if len(array.shape) == 1 else pd.DataFrame(array, index = original_index)
        else:
            ts = array
        return df_reindex(ts, index = index, method = method, limit = limit)                    
    default = np.full(array[0].shape, method if is_num(method) else np.nan)
    idx = df_reindex(pd.Series(range(array.shape[0]), original_index), index, method = None if is_num(method) else method, limit = limit).values
    res = np.array([default if np.isnan(idx[i]) else array[int(idx[i])] for i in range(len(idx))])
    if is_num(method):
        res = na2v(res, method)
    return res
    
