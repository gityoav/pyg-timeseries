import pandas as pd
import numpy as np
from scipy.optimize import minimize
TOLERANCE = 1e-10
from pyg_base import pd2np, is_df, skip_if_data_pd_or_np, try_nan, as_list, reducer, is_num, is_nums, is_pd, is_tss, is_list, df_concat, df_reindex
from pyg_timeseries._rolling import v2na, na2v, ffill
from pyg_timeseries._decorators import mask_nans, apply_along_first_axis

__all__ = ['riskparity']



def _risk_budget_objective_error(weights, args):

    # The covariance matrix occupies the first position in the variable
    covariances = args[0]

    # The desired contribution of each asset to the portfolio risk occupies the
    # second position
    assets_risk_budget = args[1]

    # We convert the weights to a matrix
    weights = np.matrix(weights)

    # We calculate the risk of the weights distribution

    gradient = covariances * weights.T
    portfolio_risk = np.sqrt(weights * gradient)[0,0]

    # portfolio_risk = _allocation_risk(weights, covariances)

    # We calculate the contribution of each asset to the risk of the weights
    # distribution
    assets_risk_contribution = np.multiply(weights.T, gradient) / portfolio_risk

    # We calculate the desired contribution of each asset to the risk of the
    # weights distribution
    assets_risk_target = np.asmatrix(np.multiply(portfolio_risk, assets_risk_budget))

    # Error between the desired contribution and the calculated contribution of
    # each asset
    error = sum(np.square(assets_risk_contribution - assets_risk_target.T))[0, 0]

    # It returns the calculated error
    return error


def _risk_budget_objective_error_no_gradient(weights, args):
    covariances = args[0]
    n = covariances.shape[0]  ** 2
    assets_risk_budget = args[1]
    w = np.matrix(weights).T ## 1 x n matrix
    gradient = covariances * w ## n x 1
    assets_risk_contribution = np.multiply(w , gradient)
    total = np.sum(assets_risk_contribution)
    assets_risk_contribution /= total
    difference = assets_risk_contribution.T - assets_risk_budget
    error = n * np.sum(np.square(difference))
    return error


def _risk_budget_objective_error_with_gradient(weights, args):
    covariances = args[0]
    n = covariances.shape[0]  ** 2
    assets_risk_budget = args[1]
    w = np.matrix(weights) ## 1 x n matrix
    gradient = 0.5 * np.array(covariances * w.T)[:,0] ## n x 1
    assets_risk_contribution = weights * gradient
    total = np.sum(assets_risk_contribution)
    assets_risk_contribution /= total
    difference = assets_risk_contribution - assets_risk_budget
    error = n * np.sum(np.square(difference))
    derror_by_dweights = n * (difference * assets_risk_contribution) * (gradient / total)
    return error, derror_by_dweights


_constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                   {'type': 'ineq', 'fun': lambda x: x})


@pd2np
@apply_along_first_axis(base_shape = 2, state = 'weights', message = 100)
def _riskparity(covariances, assets_risk_budget = None, weights = None, tol = None, method = 'pyrb'):

    # Restrictions to consider in the optimisation: only long positions whose
    # sum equals 100%

    if assets_risk_budget is None:
        n = covariances.shape[0]
        arb = np.full(n, 1. / n)
    else:
        arb = assets_risk_budget.copy()
        arb[np.isnan(arb)] = 0

    d = np.diagonal(covariances)
    msk = ~np.maximum(np.isnan(d), d <=0, arb == 0)

    masked_c = covariances[msk][:,msk]
    masked_a = arb[msk]
    masked_w = None if weights is None else weights[msk]

    fit = np.full_like(arb, np.nan)
    n = masked_c.shape[0]
    if n == 0:
        return fit
    
    if method == 'pyrb':
        try:
            import pyrb
        except Exception:
            raise ValueError('need to install pyrb from https://github.com/jcrichard/pyrb \npip install git+https://github.com/jcrichard/pyrb \nTo')
        try:
            RB = pyrb.RiskBudgeting(masked_c, masked_a)
            RB.solve()
            fit[msk] = RB.x
        except ValueError:
            pass
        return fit #, matmul(masked_c, RB.x, RB.x)

    if masked_w is None:
        x0 = np.full(n,  1./n)
    else:
        x0_mask = np.isnan(masked_w)
        if x0_mask.min():
            x0 = np.full(n, 1./n)
        else:
            x0 = masked_w
            x0[x0_mask] = np.mean(x0[~x0_mask])
    x0 = x0 / np.sum(x0)
    tol = tol or TOLERANCE
    
    # Optimisation process in scipy
    if method == 'old':
        optimize_result = minimize(fun = _risk_budget_objective_error,
                                    x0 = x0,
                                    args = [covariances, arb],
                                    method = 'SLSQP',
                                    constraints = _constraints,
                                    tol = tol,
                                    options= {'disp': False})
    elif method == 'gradient':        
        optimize_result = minimize(fun = _risk_budget_objective_error_with_gradient,
                                    x0 = x0,
                                    args = [covariances, arb],
                                    method = 'SLSQP',
                                    jac = True,
                                    constraints = _constraints,
                                    tol = tol,
                                    options= {'disp': False})
    elif method == 'sum_to_one':
        optimize_result = minimize(fun = _risk_budget_objective_error_no_gradient,
                                    x0 = x0,
                                    args = [covariances, arb],
                                    method = 'SLSQP',
                                    constraints = _constraints,
                                    tol = tol,
                                    options= {'disp': False})
    # Recover the weights from the optimised object
    fit[msk] = optimize_result.x
    return fit #, matmul(masked_c, optimize_result.x, optimize_result.x)

def riskparity(covariances, assets_risk_budget = None, columns = None, index = None, data = None, tol = None, method = 'pyrb'):
    """
    Designed to take the output from the ewmcovar calculation or a simple covariances matrix.
    Calculates risk parity weights given covariances and target budgets.
    If data is provided, will skip calculations for previously done optimizations
    
    See
    https://thequantmba.wordpress.com/2016/12/14/risk-parityrisk-budgeting-portfolio-in-python/
    or 
    https://quantdare.com/risk-parity-in-python/ 
    
    :Example:
    ---------
    >>> covariances = np.array([ [1,0.8,0.1], [0.8,1,0.1], [0.1, 0.1, 1] ]) 
    >>> weights = riskparity(covariances)
    >>> print(weights) 

    :Example: np.nans
    ------------------
    >>> covariances = np.array([ [1,0.8,0.1, np.nan], [0.8,1,0.1,np.nan], [0.1, 0.1, 1, np.nan] , [np.nan] * 4]) 
    >>> weights = riskparity(covariances)
    >>> print(weights) 

    :Parameters:
    ------------
    covariances : matrix
        covariance matrix or a timeseries of covariances
    assets_risk_budget : np.array, optional
        target risk budget per asset
    weights : np.array, optional
        initial weights for estimated allocation. The default is None.
    tol : float, optional
        Calculation tolerance. The default is None.
    columns : list, optional
        columns names. The default is None.
    index : list, optional
        index dates. The default is None.

    :Returns:
    -------
    weights : np.array
        The risk parity weights
        
        
    :Example:
    ---------
    >>> from pyg import *
    >>> rs = dictable(ticker = ['ESA Index', 'TYA Comdty', 'CLA Comdty', 'ECA Curncy'])
    >>> rs = rs(vrtn = lambda ticker: get_data('ticker', 'bbg', item = 'vrtn', ticker = ticker))
    >>> rs = rs(vrtn = lambda vrtn: vrtn[0] if is_df(vrtn) else vrtn)
    >>> vrtn = df_concat(rs.vrtn, rs.ticker)
    >>> vrtn1w = v2na(vrtn.resample('w').sum())


    >>> cor = ewmcorr_(vrtn1w, 52)['data']
    >>> cov = ewmcovar_(vrtn1w, 52)
    >>> res = riskparity(cor['data']**2, columns = cor['columns'], index = cor['index'])
    >>> res[dt(1985):].plot()
    >>> res1 = riskparity(cov['data'], columns = cov['columns'], index = cov['index'])
    >>> res[dt(1985):].plot()
    >>> res1[dt(1985):].plot()

    args = (cov['data'],)
    kwargs = dict(index = cov['index'], columns = cov['columns'])
    self = _eigenvalues
    >>> eigenvalues(cov['data'], index = cov['index']).plot()
    >>> eigenvalues(cor['data'], index = cor['index']).plot()
    >>> eigenvectors(cor['data'], n = 3, columns = cor['columns'], index = cor['index'], sort = -1)

    :Example: repeating the calculation, this time 
    ---------
    >>> res2 = riskparity(cov['data'], columns = cov['columns'], index = cov['index'], data = res1.iloc[:1800])
    >>> ratio = res2/res1
    >>> assert ratio.max().max() < 1.0001 and ratio.min().min() > 0.9999
    """
    arb = assets_risk_budget; cov = covariances; idx = index; weights = None
    cov = ffill(cov)
    if data is not None:
        if is_df(data) and index is not None: ## previous result is a timeseries
            data = data.iloc[:-1] ## remove last calculation
            mask = ~np.array([date in data.index for date in index])
            cov = cov[mask]
            idx = index[mask]
            weights = data.iloc[-1].values
            if is_df(arb):
                arb = arb[mask]
        elif not is_df(data):
            data = data[:-1]
            n = len(data)
            cov = cov[n:]
            if arb is not None and len(arb.shape) == 2:
                arb = arb[n:]
            if idx is not None:
                idx = idx[n:]
            weights = data[-1]

    if isinstance(arb, list):
        if is_nums(arb):
            arb = np.array(arb)
        elif is_tss(arb):
            arb = v2na(df_reindex(df_concat(arb), index, method = 'ffill'))
    if is_pd(arb):
        arb = arb.values
        if len(arb.shape) == 2:
            arb = ffill(arb)
        arb = na2v(arb)
        
    res = _riskparity(covariances = cov, assets_risk_budget = arb, weights = weights, tol = tol, method = method)
    
    if len(res.shape) == 1:
        return res if index is None else pd.Series(res, idx)
    else:
        res = ffill(res)
        if columns is not None or index is not None:
            res = pd.DataFrame(data = res, columns = columns, index = idx) 
            if data is not None:
                res = pd.concat([data, res]).sort_index()
            return res
        else:
            if data is not None:
                res = np.concatenate([data, res])
            return res

    

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
@skip_if_data_pd_or_np
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
    res = _matmul(matrix = matrix, rhs = rhs_[0], lhs = lhs_[0], data = data, index = index, columns = columns)
    if power!=1:
        if power<0:
            res = v2na(res)
        res = res ** power
    return res

