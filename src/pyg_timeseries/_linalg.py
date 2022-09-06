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
