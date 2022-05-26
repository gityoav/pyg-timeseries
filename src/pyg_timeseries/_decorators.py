from pyg_base import getargspec, first, Dict, loop, zipper, as_list, getargs, wrapper
from numba import njit
import numpy as np

__all__ = ['compiled']

def compiled(function):
    res = njit(nogil = True)(function)
    res.fullargspec = getargspec(function)
    return res

#from pyg.base import passthru; compiled = passthru ## use this to get valid code-coverage for compiled functions

first_ = loop(dict, list)(first)


def _data_state(keys, values, output = 'data'):
    if isinstance(values, dict):
        return type(values)({k: _data_state(keys, v, output) for k, v in values.items()})
    elif isinstance(values, list):
        return type(values)([_data_state(keys, v, output) for v in values])
    output = as_list(output)
    assert keys[:len(output)] == output
    res = Dict(zip(output,values))
    if len(keys) > len(output):
        res['state'] = Dict(zipper(keys[len(output):], values[len(output):]))
    return res


def _ignore(ignore, data, state):
    state = {} if state is None else state
    if ignore is True:
        return None, {}
    elif ignore is False:
        return data, state
    ignore = as_list(ignore)
    if 'data' in ignore:
        data = None
    if 'state' in ignore:
        state = {}
    return data, state



@loop(dict, tuple, list)
def _mask(arg, mask, apply_axis):
    """
    removes nans based on a 1-d mask

    """
    res = arg
    f = len(mask)
    if isinstance(arg, np.ndarray):
        if (len(arg.shape) == 1 or apply_axis == 0) and arg.shape[0] == f:
            res = res[mask]
        elif len(arg.shape) > 1 and apply_axis == 1 and arg.shape[1] == f:
            res = res[:, mask]
        elif len(arg.shape) > 2 and apply_axis == 2 and arg.shape[2] == f:
            res = res[:, mask]
    return res

def _unmask1(arg, mask, apply_axis):
    n = len(mask[mask]) # len of no nans
    f = len(mask) # full length
    res = arg
    if (len(arg.shape) > 0 and apply_axis == 0) and arg.shape[0] == n:
        res = np.full( (f,) + arg.shape[1:], np.nan)
        res[mask] = arg
    elif len(arg.shape) > 1 and apply_axis == 1 and arg.shape[1] == n:
        res = np.full((arg.shape[0], f) + arg.shape[2:], np.nan)
        res[:, mask] = arg
    elif len(arg.shape) > 2 and apply_axis == 2 and arg.shape[2] == n:
        res = np.full((arg.shape[0], arg.shape[1], f) + arg.shape[3:], np.nan)
        res[:, mask] = arg
    return res
    

@loop(dict, tuple, list)
def _unmask(arg, masks):
    """
    returns nans based on a 1-d mask
    
    :Example:
    ----------
    >>> mask0 = np.array([True, False, True, True, False, True])
    >>> mask1 = np.array([True, False, True, True])
    >>> mask2 = np.array([True, True, False, True])
    >>> arg = np.array([[1,2,3.], [4,5,6.], [7,8,9.], [10,11,12]])
    >>> arg_ = _unmask1(_unmask1(arg, mask0, 0), mask1, 1)
    >>> arg1 = np.array([[1,2,3.], [4,5,6.], [7,8,9.]])
    >>> arg1_ = _unmask1(_unmask1(arg1, mask1, 0), mask1, 1)
    >>> arg2_ = _unmask1(_unmask1(arg1, mask2, 0), mask1, 1)
    
    
    >>> mask_nan_in_array(np.sum)(arg_, axis = 0)
    >>> mask_nan_in_array(np.sum)(arg, axis = 0)
    _unmask(mask_nan_in_array(np.sum)(arg, axis = 0), mask1, 1)
    
    >>> arg = _mask(_mask(arg, mask, apply_axis = 0), mask, 1)
    
    

    """
    res = arg
    if isinstance(arg, np.ndarray):
        if len(arg.shape) == 1: ## we give priorities to earlier dimensions
            masks = {apply_axis: mask for apply_axis, mask in masks.items() if len(mask[mask]) == arg.shape[0]}
            if len(masks) == 0:
                return res
            elif len(masks) > 1:
                unique_masks = set([tuple(mask) for mask in masks.values()])
                if len(unique_masks) > 1:
                    raise ValueError('There are multiple dimension possible to unmask the result %s'%masks)
            else:
                for apply_axis, mask in masks.items():
                    f = len(mask)
                    res = np.full(f, np.nan)
                    res[mask] = arg
                    return res
        elif len(arg.shape) == 2 and max(mask.keys()) == 3:
            raise ValueError('not implemented')
        for apply_axis, mask in masks.items():
            res = _unmask1(res, mask, apply_axis)
    return res


def _arg(function, args, kwargs):
    if len(args):
        arg = args[0]
    else:
        arg = kwargs[getargs(function)[0]]
    return arg
    
class mask_nans(wrapper):
    """
    This wrapper allows us to operate in as if nan's are not actually provided
    """
    def __init__(self, function = None, apply_axis = None, exclude_any_nan = False):
        return super(mask_nans, self).__init__(function = function, apply_axis = apply_axis, exclude_any_nan = exclude_any_nan)

    def wrapped(self, *args, **kwargs):
        arg = _arg(self.function, args, kwargs)
        if not isinstance(arg, np.ndarray):
            return self.function(*args, **kwargs)        
        masks = {}
        mask = ~np.isnan(arg)
        args_, kwargs_ = args, kwargs
        for apply_axis in range(len(arg.shape)):
            if self.apply_axis is None or apply_axis in as_list(self.apply_axis):
                msk = mask
                for other_axis in range(len(arg.shape)):
                    if other_axis!=apply_axis:
                        msk = msk.min(axis = other_axis) if self.exclude_any_nan else msk.max(axis = other_axis)
                if not msk.min():
                    masks[apply_axis] = msk
                    args_, kwargs_ = _mask((args_, kwargs_), mask = msk, apply_axis = apply_axis)
                    arg = _arg(self.function, args_, kwargs_)

        res = self.function(*args_, **kwargs_)
        res = _unmask(res, masks)
        return res



# import pandas as pd
# from pyg.base import wrapper, is_ts, is_dict, is_tuple, getargspec, loop, first, Dict, zipper

# class persist_data(wrapper):
#     """
#     We work with state-persisting function that return a variable called 'data'
    
#     :Example:
#     -------
#     >>> from pyg import *
#     >>> ts = pd.Series(np.random.normal(0,1,(1000)), drange(-999))
#     >>> ts[ts<0.1] = np.nan

#     We can run ts in one go:
        
#     >>> both = ffill_(ts)

#     Or split it into two parts:

#     >>> old = ffill_(ts.iloc[:500])
    
#     old.data is the data for first 500 entries. We can use this data to speed up later calculations...
    
#     >>> new = ffill_(ts.iloc[500:], **(old-'data')) 
    
#     We note: 
#     1) the 'data' variable must be remove
#     2) We need to glue together old.data and new.data
    
#     However... persist_data does the work for you:
        
#     >>> persist_ffill_ = persist_data(ffill_)
#     >>> glued_with_old = persist_ffill_(ts, **old)  # Note that we pass on data and the full ts, but calculation is done only on the 'new' bit of ts
#     >>> glued_just_new = persist_ffill_(ts.iloc[500:], **old)  

#     >>> assert eq(glued_with_old, both)    
#     >>> assert eq(glued_just_new, both)    
    
#     """
#     def wrapped(self, *args, **kwargs):
#         data = kwargs.pop('data', None)
#         if data is not None and len(data)>0 and is_ts(data):
#             cutoff = data.index[-1]
#             args_ = [v[v.index>cutoff] if is_ts(v) and len(v) else v for v in args]
#             kwargs_ = {k : v[v.index>cutoff] if is_ts(v) and len(v) else v for k, v in kwargs.items()}
#             res = self.function(*args_, **kwargs_)
#             if res is None:
#                 return data
#             if is_ts(res):
#                 new = res
#             elif is_dict(res):
#                 new = res.get('data')
#             elif is_tuple(res):
#                 new = res[0]
#             else:
#                 raise ValueError('result of persistence data must be a timeseries or a dict \n%s '%res)
#             if new is None:
#                 both = data
#             elif is_ts(new):
#                 both = data if len(new) == 0 else pd.concat([data[data.index<new.index[0]], new])
#             else:
#                 raise ValueError('data must be None or a timeseries')                
#             if is_ts(res):
#                 return both
#             elif is_dict(res):
#                 res['data'] = both
#                 return res
#             elif is_tuple(res):
#                 return (both,) + res[1:]
#         else:
#             return self.function(*args, **kwargs)

