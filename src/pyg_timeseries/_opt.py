import numpy as np
from pyg_base import is_ts, is_pd

def least_squares(A, b, cost = None, **kwargs):
    """
    single value least-squares, 
    Optimizing ||Ax - b|| in L2 Norm

    :Parameters:
    ----------
    A: 1d or 2d np.array
        nans in A's are converted to zeros
        
    b : 1d np.array
        nans in b yields to removed rows of both A and b
        
    Returns
    -------
    np.array 1d
        optimal solution

    :Example:
    ---------
    >>> m = 20
    >>> n = 15
    >>> t = 10
    >>> np.random.seed(1)
    
    >>> A = np.random.randn(t, m, n)
    >>> b = np.random.randn(t, m)
    
    # Add noise...
    
    >>> b[np.random.randn(t, m)>1] = np.nan
    >>> A[np.random.randn(t, m, n)>1] = np.nan

    >>> dfA = pd.DataFrame(A.reshape(t, m * n), drange(1-t))
    >>> dfb = pd.DataFrame(b, drange(1-t))

    # solve...

    >>> solution = ts_iterate(functor = least_squares, variables = dict(A = dfA, b = dfb), dates = dfA)

    # merge with previous solution:
    
    >>> solution2 = ts_iterate(functor = least_squares, variables = dict(A = dfA, b = dfb), dates = dfA, data = pd_solution.drop(solution.index[::3]))
    >>> assert eq(solution2, solution)
    
    """
    import cvxpy as cp
    if cost is None:
        cost = cp.sum_squares
    m = b.shape[0]
    n = A.shape[0] // m if len(A.shape) == 1 else A.shape[1]
    if is_ts(A) or is_ts(b):
        raise ValueError('cannot get a timeseries')
    if is_pd(A):
        A = A.values
    if is_pd(b):
        b = b.values
    A = A.reshape(m,n) if len(A.shape) == 1 else A
    mask = ~np.isnan(b)
    b = b[mask]
    A = A[mask]
    A[np.isnan(A)] = 0
    x = cp.Variable(n)
    costs = cost(A @ x - b)
    prob = cp.Problem(cp.Minimize(costs))
    prob.solve()
    return x.value

