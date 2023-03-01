from scipy.optimize import minimize
from pyg_timeseries._rolling import ffill
from pyg_base import is_num
import numpy as np

def residual(c):
    return (1-c**2)**0.5


def _variance(weights, c):
    return weights.dot(c.dot(weights.transpose()))

def _addem(weights):
    return 1.0 - np.sum(weights)

def _ensure_initial_weights_verify_bounds(weights, min_weights, max_weights):
    return weights



def _single_period_max_diversify(c, min_weights, max_weights, weights = None, tol = 0.001):
    """
    This implements a one-step optimization of weights. We handle nan in various columns

    see also 
    robcarver17/march2023_independent_bets.py
    https://gist.github.com/robcarver17/c350ca7c73018c175393e44ad839c061

    """
    bad = np.isnan(c[0])
    good = ~bad
    t = c.shape[0]
    n = sum(good[good])
    if n == 1:
        weights = np.zeros(t)
        weights[good] = 1/n
        return weights
    if weights is None:
        weights = np.zeros(t)
        weights[good] = 1/n
    
    ## now we project onto the good area
    weights_ = weights[good]
    min_weights_ = min_weights[good]
    max_weights_ = max_weights[good]
    c_ = c[good,good]

    weights = _ensure_initial_weights_verify_bounds(weights = weights_, 
                                                    min_weights = min_weights_, 
                                                    max_weights=max_weights_)
    bounds = list(zip(min_weights_, max_weights_))
    cdict = [{"type": "eq", "fun": _addem}]
    ans = minimize(
           _variance,
           weights_,
           (c_,),
           method="SLSQP",
           bounds=bounds,
           constraints=cdict,
           tol=tol)
    rtn = np.zeros(t)
    rtn[good] = ans['x']
    return rtn
        
    


def maxdiv(correlation, min_weights = None, max_weights = None, max_change = None, weights = None):
    """
    ### let us set up the assets
    
    >>> sp = pd.Series(np.random.normal(0,1, 10000), drange(-9999))
    >>> dax_c = 0.3; dax = add_(pd.Series(np.random.normal(0,1, 5000), drange(-4999)) * residual(dax_c),  dax_c * sp)
    >>> cac_c = 0.1; cac = add_(pd.Series(np.random.normal(0,1, 4000), drange(-3999)) * residual(cac_c),  cac_c * sp)
    >>> ty_c = -0.2; ty = add_(pd.Series(np.random.normal(0,1, 8000), drange(-7999)) * residual(ty_c),  ty_c * sp)

    >>> dax[np.random.normal(0,1,len(dax)) > 3] = np.nan
    >>> cac[np.random.normal(0,1,len(cac)) > 3] = np.nan
    >>> ty[np.random.normal(0,1,len(ty)) > 3] = np.nan
    
    >>> rtn = df_concat([sp,dax,cac,ty], ['sp', 'dax', 'cac', 'ty'])
    >>> correlation = ewmcorr(rtn, 255)
    >>> assert correlation.shape == (10000,4,4)

    correlation[np.isnan(rtn.dax).values]
    correlation = ffill(correlation)
    """
    correlation = ffill(correlation)
    c = correlation[0]
    t = c.shape[0]
    if min_weights is None:
        min_weights = 0.
    if max_weights is None:
        max_weights = 1.
    if is_num(min_weights):
        min_weights = np.array([min_weights] * t)
    if is_num(max_weights):
        max_weights = np.array([max_weights] * t)
    if max_change is None:
        max_change = 1.
    if is_num(max_change):
        max_change = np.array([max_change] * t)

