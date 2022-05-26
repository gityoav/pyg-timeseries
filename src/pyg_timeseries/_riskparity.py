import pandas as pd
import numpy as np
import datetime
from scipy.optimize import minimize
TOLERANCE = 1e-10
from pyg_base import pd2np, is_df
from pyg_timeseries._decorators import mask_nans

__all__ = ['riskparity']

def _allocation_risk(weights, covariances):

    # We calculate the risk of the weights distribution
    portfolio_risk = np.sqrt((weights * covariances * weights.T))[0, 0]

    # It returns the risk of the weights distribution
    return portfolio_risk


def _assets_risk_contribution_to_allocation_risk(weights, covariances):

    # We calculate the risk of the weights distribution
    portfolio_risk = _allocation_risk(weights, covariances)

    # We calculate the contribution of each asset to the risk of the weights
    # distribution
    assets_risk_contribution = np.multiply(weights.T, covariances * weights.T) / portfolio_risk

    # It returns the contribution of each asset to the risk of the weights
    # distribution
    return assets_risk_contribution


def _risk_budget_objective_error(weights, args):

    # The covariance matrix occupies the first position in the variable
    covariances = args[0]

    # The desired contribution of each asset to the portfolio risk occupies the
    # second position
    assets_risk_budget = args[1]

    # We convert the weights to a matrix
    weights = np.matrix(weights)

    # We calculate the risk of the weights distribution
    portfolio_risk = _allocation_risk(weights, covariances)

    # We calculate the contribution of each asset to the risk of the weights
    # distribution
    assets_risk_contribution = _assets_risk_contribution_to_allocation_risk(weights, covariances)

    # We calculate the desired contribution of each asset to the risk of the
    # weights distribution
    assets_risk_target = np.asmatrix(np.multiply(portfolio_risk, assets_risk_budget))

    # Error between the desired contribution and the calculated contribution of
    # each asset
    error = sum(np.square(assets_risk_contribution - assets_risk_target.T))[0, 0]

    # It returns the calculated error
    return error


_constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                   {'type': 'ineq', 'fun': lambda x: x})


@mask_nans
def _riskparity(covariances, assets_risk_budget = None, initial_weights = None, tol = None):

    # Restrictions to consider in the optimisation: only long positions whose
    # sum equals 100%
    n = covariances.shape[0]
    if n == 0:
        return np.array([]).astype(float)
    if initial_weights is None:
        x0 = np.full(n,  1./n)
    else:
        x0_mask = np.isnan(initial_weights)
        if x0_mask.min():
            x0 = np.full(n, 1./n)
        else:
            x0 = initial_weights
            x0[x0_mask] = np.mean(x0[~x0_mask])
    x0 = x0 / np.sum(x0)

    if assets_risk_budget is None:
        arb = np.full(n, 1. / n)
    else:
        arb = assets_risk_budget.copy()
        arb[np.isnan(arb)] = 0
        if np.sum(arb) == 0:
            arb = np.full(n, 1./n)
        else:
            arb = arb / np.sum(arb)
        
    tol = tol or TOLERANCE
    
   # Optimisation process in scipy
    optimize_result = minimize(fun = _risk_budget_objective_error,
                               x0 = x0,
                               args = [covariances, arb],
                               method = 'SLSQP',
                               constraints = _constraints,
                               tol = tol,
                               options= {'disp': False})

    # Recover the weights from the optimised object
    weights = optimize_result.x
    return weights

@pd2np
def _riskparity3d(covariances, assets_risk_budget = None, initial_weights = None, tol = None):
    if len(covariances.shape) == 3:
        t = covariances.shape[0]; n = covariances.shape[1]
        if assets_risk_budget is None:
            assets_risk_budget = np.full(n, 1)
        weights = np.empty((t, n))
        for i in range(t):
            arb = assets_risk_budget[i] if len(assets_risk_budget.shape) == 2 else assets_risk_budget
            initial_weights = weights[i] = _riskparity(covariances[i], assets_risk_budget = arb, initial_weights = initial_weights, tol = tol)
        return weights            
    else:
        return _riskparity(covariances, assets_risk_budget = assets_risk_budget, initial_weights = initial_weights, tol = tol)


def riskparity(covariances, assets_risk_budget = None, tol = None, columns = None, index = None, data = None):
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
    initial_weights : np.array, optional
        The default is None.
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
    >>> rs = dictable(ticker = ['ESA Index', 'TYA Comdty', 'CLA Comdty', 'ECA Curncy'])
    >>> rs = rs(vrtn = lambda ticker: get_data('ticker', 'bbg', item = 'vrtn', ticker = ticker))
    >>> rs = rs(vrtn = lambda vrtn: vrtn[0] if is_df(vrtn) else vrtn)
    >>> vrtn = df_concat(rs.vrtn, rs.ticker)
    >>> vrtn1w = v2na(vrtn.resample('w').sum())

    >>> from pyg_timeseries import *
    >>> cor = ewmcorr_(vrtn1w, 52)['data']
    >>> cov = ewmcovar_(vrtn1w, 52)
    >>> res = riskparity(cor['data']**2, columns = cor['columns'], index = cor['index'])
    >>> res[dt(1985):].plot()
    >>> res1 = riskparity(cov['data'], columns = cov['columns'], index = cov['index'])
    >>> res[dt(1985):].plot()
    >>> res1[dt(1985):].plot()

    :Example: repeating the calculation, this time 
    ---------
    >>> res2 = riskparity(cov['data'], columns = cov['columns'], index = cov['index'], data = res1.iloc[:1800])
    >>> ratio = res2/res1
    >>> assert ratio.max().max() < 1.0001 and ratio.min().min() > 0.9999
    """
    arb = assets_risk_budget; cov = covariances; idx = index; initial_weights = None
    if data is not None:
        if is_df(data) and index is not None: ## previous result is a timeseries
            data = data.iloc[:-1] ## remove last calculation
            mask = ~np.array([date in data.index for date in index])
            cov = cov[mask]
            idx = index[mask]
            initial_weights = data.iloc[-1].values
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
            initial_weights = data[-1]

    res = _riskparity3d(covariances = cov, assets_risk_budget = arb, initial_weights = initial_weights, tol = tol)
    if len(res.shape) == 1:
        return res if index is None else pd.Series(res, idx)
    else:
        if columns is not None or index is not None:
            res = pd.DataFrame(data = res, columns = columns, index = idx) 
            if data is not None:
                res = pd.concat([data, res]).sort_index()
            return res
        else:
            if data is not None:
                res = np.concatenate([data, res])
            return res
        