import numpy as np
import pandas as pd
from scipy.optimize import minimize
TOLERANCE = 1e-10
from pyg_base import pd2np, is_df, skip_if_data_pd_or_np, try_nan, as_list, reducer, is_num, is_nums, is_pd, is_tss, is_list, df_concat, df_reindex
from pyg_timeseries._rolling import v2na, na2v, ffill
from pyg_timeseries._decorators import mask_nans, apply_along_first_axis


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

def _data_ok(data, covariances, columns):
    if data is None or len(data.shape)!=2: ## shape OK
        return False
    if data.shape[1] != covariances.shape[1]: ## matching covariances
        return False
    if is_df(data) and columns is not None and list(columns)!=list(data.columns): # match columns if available
        return False
    return True
        

def riskparity(covariances, assets_risk_budget = None, columns = None, index = None, data = None, tol = None, method = 'pyrb'):
    """
    Designed to take the output from the ewmcovar calculation or a simple covariances matrix.
    Calculates risk parity weights given covariances and target budgets.
    If data is provided, will skip calculations for previously done optimizations.
    
    The faster implementation is using pyrb available from https://github.com/jcrichard/pyrb
    This relies on quadprog https://github.com/quadprog/quadprog which in turns requires a C compiler like the MS one.
    
    The slower implementation is using scipy:
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
    if _data_ok(data, covariances, columns):      
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
    else:
        data = None

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

    # -*- coding: utf-8 -*-

