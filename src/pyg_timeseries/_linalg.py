import pandas as pd
import numpy as np
from scipy.optimize import minimize
TOLERANCE = 1e-10
from pyg_base import pd2np, is_df, skip_if_data_pd_or_np, try_nan, as_list, reducer, is_num, is_nums, is_pd, is_tss, is_list, df_concat, df_reindex
from pyg_timeseries._rolling import v2na, na2v, ffill
from pyg_timeseries._decorators import mask_nans, apply_along_first_axis


@skip_if_data_pd_or_np
@pd2np
@apply_along_first_axis(base_shape = 2)
@mask_nans
def _eigenvalues(matrix, sort = -1):
    try:
        evals, evec = np.linalg.eig(matrix)
    except Exception:
        return np.full(matrix.shape[0], np.nan)
    if sort!=0 and len(evals):
        _, ns = zip(*sorted(zip(evals, range(len(evals))))[::sort])
        ns = np.array(ns)
        evals = evals[ns]
    return evals    

    
def eigenvalues(matrix, index = None, data = None):
    """
    A thin wrapper for numpy np.linalg.eig
    By default, eigen values (and vectors) are sorted in decreasing order
    
    :Parameters:
    ------------
    matrix:
        A timeseries (t x n x n numpy array) or a single n x n matrix 
    index: 
        The result can be put into a timeseries if index is provided
    sort: int
        sorting mechanism for size of eigenvalues

    """
    return _eigenvalues(matrix, index = index, data = data)


@skip_if_data_pd_or_np
@pd2np
@apply_along_first_axis(base_shape = 2)
@mask_nans
def _eigenvectors(matrix, n = None, sort = -1):
    try:
        evals, evec = np.linalg.eig(matrix)
    except Exception:
        return np.full(matrix.shape if n is None else matrix.shape[0], np.nan)
    if sort!=0 and len(evals):
        _, ns = zip(*sorted(zip(evals, range(len(evals))))[::sort])
        ns = np.array(ns)
        evec = evec[:,ns]        
    if n is None:
        return evec
    else:
        if n < len(evec):
            return evec[:, n]
        else:
            return np.full(len(evec), np.nan)
    
def eigenvectors(matrix, n = None, sort = -1, columns = None, index = None, data = None):
    """
    A thin wrapper for numpy np.linalg.eig
    By default, eigen values (and vectors) are sorted in decreasing order
    
    :Parameters:
    ------------
    matrix:
        A timeseries (t x n x n numpy array) or a single n x n matrix 
    n: int or None
        if provided, will return the nth eigen vector (by sorted size)
    
    index: list/pd.Index of dates
        The result can be put into a timeseries if index is provided

    columns: list of components for matrix
        If n is provided, the result can be put into a timeseries dataframe if index is provided. Columns represent the projection of the eigen-vector

    sort: int
        sorting mechanism for size of eigenvalues

    """
    return _eigenvectors(matrix, n = n, columns = columns, index = index, data = data)


@skip_if_data_pd_or_np
@pd2np
@apply_along_first_axis(base_shape = 2)
@mask_nans
@try_nan
def _det(matrix):
    # if len(matrix.shape) == 1 and int(np.sqrt(matrix.shape[0]))**2 == matrix.shape[0]:
    #     n = int(np.sqrt(matrix.shape[0]))
    #     matrix = matrix.reshape((n,n))
    return np.linalg.det(matrix)    

def det(matrix, data = None):
    """
    A thin wrapper around np.linalg.det.
    
    - handles a timeseries of matrices
    - handles nan's a little better

    :Example:
    ---------
    >>> matrix = np.array([[1,2],[3,4]])
    >>> assert round(det(matrix),10) == -2.

    >>> matrices = np.array([matrix] * 5)   ### 5 x 2 x 2 is viewed as a timeseries of matrices
    >>> assert len(det(matrices)) == 5      ### get 5 x 1 back

    >>> matrix = np.array([[1,np.nan],[np.nan,np.nan]])     ## we drop rows and columns (2nd ones here) that are all nan
    >>> assert det(matrix) == 1
    """
    return _det(matrix, data = data)


def _mult(a, b):
    """
    A much more forgiving np.matmul implementation.
    - supports multiplying by numbers
    - supports multiplying by vectors posing as 1xn matrices
    """
    if is_pd(a):
        a = a.values
    if is_pd(b):
        b = b.values
    if is_num(a) or is_num(b):
        return a * b
    if len(a.shape) == 1 and len(b.shape) == 1:
        return np.matmul(a,b)
    if len(a.shape) == 1:
        a = np.array([a])
    if len(b.shape) == 1:
        b = np.array([b])
    if len(a.shape) == 2 and len(b.shape) == 2 and a.shape[1]!=b.shape[0]: 
        ## we first flip and fake vectors
        if a.shape[1] == 1:
            a = a.T
        if b.shape[0] == 1:
            b = b.T
    if len(a.shape) == 2 and len(b.shape) == 2 and a.shape[1]!=b.shape[0] and a.shape[0] == b.shape[0]:
        res = np.array([_mult(a[i], b[i]) for i in range(a.shape[0])]) ## we interpret the first dimension as a timeseries multiplication
    else:
        res = np.matmul(a, b)
    if len(res.shape)>= 1 and res.shape[0] == 1:
        res = res[0]
    elif len(res.shape) >= 2 and res.shape[1] == 1:
        res = res[:, 0]
    if len(res.shape)>= 1 and res.shape[0] == 1:
        res = res[0]
    return res
    
def _as_list(value):
    if value is None:
        return []
    elif is_num(value):
        return [value]
    elif is_list(value) and is_nums(value):
        return [np.array(value)]
    elif is_pd(value):
        return [value]
    else:
        return as_list(value)

@pd2np
@apply_along_first_axis(base_shape = 2)
@mask_nans
def _matmul(matrix, rhs = None, lhs = None):
    matrices = _as_list(lhs) + _as_list(matrix) + _as_list(rhs)
    res = reducer(_mult, matrices)
    return res

def matmul(matrix, rhs = None, lhs = None, data = None, index = None, columns = None, power = 1):
    """

    :Example: simple matrix multiplication
    ----------
    >>> from pyg import *
    >>> rhs = np.array([1,2,3])
    >>> lhs = np.array([1,2,3])
    >>> matrix = np.array([[1,2,3],[4,5,6], [7,8,9]])    

    >>> assert matmul([1,2,3], [4,5,6]) == 32
    >>> assert eq(matmul(rhs,2), np.array([2, 4, 6]))
    >>> assert matmul(rhs,lhs) == 14
    >>> assert eq(matmul(matrix,rhs), np.array([14, 32, 50]))
    >>> assert matmul(matrix, rhs, lhs) == 228
    

    :Example: timeseries multiplication
    ----------
    >>> m = np.array([matrix] * 10)    
    >>> r = np.array([rhs] * 10)
    >>> l = np.array([lhs] * 10)
    
    >>> assert eq(matmul(r, l), np.array([14]*10))
    >>> assert eq(matmul(m, r, l), np.array([228]*10))
    >>> assert eq(matmul(matrix, r, l), np.array([228]*10))
    >>> assert eq(matmul(m, rhs, lhs), np.array([228]*10))
    >>> assert eq(matmul(m, r, lhs), np.array([228]*10))
    >>> assert eq(matmul(m, r), np.array([[14, 32, 50]] * 10))
    >>> assert eq(matmul(m, rhs), np.array([[14, 32, 50]] * 10))


    :Example: handling nans
    -----------------------
    from numpy import nan
    >>> matrix = np.array([[1,2,nan], [3,4,nan], [nan, nan, nan]])
    >>> rhs = np.array([1,2,np.nan])
    >>> assert eq(matmul(matrix, rhs), np.array([ 5., 11., nan]))
    >>> assert eq(matmul(matrix, rhs, rhs), 27)
    >>> assert matmul(rhs, lhs) == 5
    """    
    lhs_    = _as_list(lhs) 
    matrix_ = _as_list(matrix) 
    rhs_    = _as_list(rhs)
    pds = [m for m in lhs_ + matrix_ + rhs_ if is_pd(m)]
    lhs_ = [m.values if is_pd(m) else m for m in lhs_]
    rhs_ = [m.values if is_pd(m) else m for m in rhs_]
    if index is None:
        pd0 = [m for m in pds if m.shape[0] == matrix.shape[0]]
        if len(pd0):
            index = pd0[0].index
    if columns is None:
        pd1 = [m for m in pds if len(m.shape) == 2 and m.shape[1] == matrix.shape[1]]
        if len(pd1):
            columns = pd1[0].columns        
    res = _matmul(matrix = matrix, rhs = rhs_, lhs = lhs_)
    if power!=1:
        if power<0:
            res = v2na(res)
        res = res ** power
    if len(res.shape) == 1 and index is not None:
        res = pd.Series(res, index)
    elif len(res.shape) == 2 and (index is not None or columns is not None):
        res = pd.DataFrame(res, index, columns)        
    return res

