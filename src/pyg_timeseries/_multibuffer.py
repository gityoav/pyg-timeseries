from pyg_base import df_reindex, is_df, pd2np, loop, Dict, is_nums, is_pd, is_arr, is_num, df_concat, is_series, is_strs, dt, is_tss, is_ts
from pyg_timeseries._linalg import matmul
from pyg_timeseries._rolling import buffer, ffill, v2na
from pyg_timeseries._pandas import reindex_3d
import numpy as np
from functools import partial

def bisect(f, lb, ub, n = 0, aim = np.nan):
    """
    bisection like scipy.optimize.bisect but supports discontinous functions snapping towards our "aim"
    
    Example:
    --------
    >>> f = lambda x: 1 if x>0 else -1
    >>> assert bisect(f, 1, 3, aim = 1.3) == 1.3
    >>> assert bisect(f, -1, 3, aim = 1.3) == 1.3    

    >>> f = lambda x: 1 if x>2 else 0.5 if x>1 else -1 
    >>> assert round(bisect(f, -1, 3, 20, aim = 3),3) == 1.0 ## The crossing from positive to negative is at 1 so aim of 3 does not chain

    """
    flb = f(lb)
    fub = f(ub)
    for i in range(n or 10):
        if fub * flb >= 0 :
            break
        else:
            mid = (lb + ub) / 2
            fmid = f(mid)
            if flb * fmid < 0:
                ub = mid
                fub = fmid
            else:
                lb = mid
                flb = fmid
    mid = (lb + ub) / 2
    if np.isnan(aim):
        return mid                    
    elif f(aim) == f(mid):
        return aim
    elif aim < mid:
        return lb
    else:
        return ub


def _single_step_multibuffer(target, band, point_values, vol, correlations, previous, unit = 1.0, prev_m = 1.0, risk_band = 0.1, rounding_band = 0):    
    """
    Since actual positions are discrete while the target positions can be a fraction, 
    _single_step_multibuffer adjusts the target risk using a multiplier so that the realised (integer valued) buffered positions match the overall risk within [1-risk_band, 1+risk_band]
    
    :Parameters:
    ------------
    - target : array
        target positions
    - band : float
        bands for the target positions of individual markets (see buffer function)
    - point_values : array
        a multiplier per each position. This is useful if for example, a unit position in a market actually corresponds to much higher value (e.g. in Futures, point_values will be fx_rate * full_point_value)
    - vol: array
        price volatility
    - correlations: 
        actually, correlations between the markets
    - previous: array
        previous integer valued positions
    - unit: int
        The minimum amount of trading in an asset (e.g 1 future lots)
    - prev_m: float
        The previous adjustment multiplier
    - risk_band: float
        The band of risk we would like to keep overall
        
        
    :Example:
    ---------
    >>> correlations = np.array([[1,0.3], [0.3, 1]])
    >>> point_values = np.array([0.5, 0.5])
    >>> vol = np.array([1.2, 4.3])
    >>> mult = 3
    >>> target = np.array([0.3, 0.8]) ## current target positions
    >>> previous = np.array([0., 2.]) ## previous position
    >>> band = np.array([0.1, 0.1])
    >>> prev_m = 1.0
    >>> risk_band = 0.1
    
    
    >>> from pyg import *
    >>> a = np.array([target])
    >>> state = Dict(pos = list(previous))
    >>> u = list(unit)
    
    """
    pv = point_values * vol
    w = np.matmul(np.array([pv]).T, np.array([pv]))
    cw = correlations * w
    cw[np.isnan(cw)] = 0.0
    target[np.isnan(target)] = 0.0
    target_risk = matmul(cw, target, target, 0.5)
    b = np.array([band])
    state = dict(pos = list(previous))
        
    if target_risk == 0:
        def _mismatch(buffered_pos, target_mismatch = 1.0):
            return 0.
        def do_buffer_and_calculate_mismatch(m, target_mismatch = 1.0):
            return 0.
    else:        
        
        def _mismatch(buffered_pos, target_mismatch = 1.0):
            buffered_risk = matmul(cw, buffered_pos, buffered_pos, 0.5)
            return buffered_risk / target_risk - target_mismatch
        
        def do_buffer_and_calculate_mismatch(m, target_mismatch = 1.0):
            buffered_pos = buffer(a = np.array([target * m]), band = b, state = state, unit = unit, rounding_band = rounding_band)[0]
            return _mismatch(buffered_pos, target_mismatch)

    m = prev_m
    buffered_pos = buffer(a = np.array([target * m]), band = b, state = state, unit = unit, rounding_band = rounding_band)[0]
    mismatch = _mismatch(buffered_pos)

    if mismatch < - risk_band:
        function = partial(do_buffer_and_calculate_mismatch, target_mismatch = 1 - risk_band) ## we need to drive up risk but only to 1-risk_band
        m = bisect(function, lb = prev_m, ub = prev_m + risk_band, n = 5, aim = prev_m)        
        buffered_pos = buffer(a = np.array([target * m]), band = b, state = state, unit = unit)[0]
        # m = prev_m * (1 - risk_band) / mismatch
        mismatch = _mismatch(buffered_pos)
        if mismatch > risk_band:
            m = prev_m
            buffered_pos = buffer(a = np.array([target * m]), band = b, state = state, unit = unit, rounding_band = rounding_band)[0]
            

    elif mismatch > risk_band:
        function = partial(do_buffer_and_calculate_mismatch, target_mismatch = 1 + risk_band)
        m = bisect(function, lb = prev_m - risk_band, ub = prev_m, n = 5, aim = prev_m)                
        # m = prev_m * (1 + risk_band) / mismatch
        buffered_pos = buffer(a = np.array([target * m]), band = b, state = state, unit = unit, rounding_band = rounding_band)[0]
        mismatch = _mismatch(buffered_pos)
        if mismatch < - risk_band:
            m = prev_m
            buffered_pos = buffer(a = np.array([target * m]), band = b, state = state, unit = unit, rounding_band = rounding_band)[0]
    
    return m, buffered_pos, _mismatch(buffered_pos)


@loop(list, tuple, dict)
def _from(value, n, t = None):
    if isinstance(value, np.ndarray) and (t is None or len(value) == t):
        return value[n:]
    else:
        return value

@loop(list)
def _subset(value, keys = None, ids = None):
    if keys is None and ids is None:
        return value
    elif is_df(value):
        return value.iloc[:, ids] if keys is None else value[keys]
    elif is_series(value):
        if is_ts(value):
            return value
        else:
            return value.iloc[ids] if keys is None else value[keys]
    elif is_arr(value):
        if ids is None:
            return value
        elif len(value.shape) == 1:
            return value[ids]
        elif len(value.shape) == 2:
            return value[:,ids]
        else:
            res = value[:,ids,:]
            res = res[:,:,ids]
            return res
    else:
         return value
    

@pd2np
def _multibuffer(target, band, unit, correlations, volatilities, point_values, data = None, mult = None, mismatch = None, risk_band = 0.1, rounding_band = 0):
    if data is None:
        previous = np.zeros(target.shape[1])
        m = 1
    else:
        previous = data[-1]
        m = mult[-1]
        n = len(data)
        t = len(target)
        if n == t:
            return Dict(mult = mult, data = data, mismatch = mismatch)
        target, band, unit, correlations, volatilities, point_values = _from([target, band, unit, correlations, volatilities, point_values], n = n, t = t)
    ds = np.zeros(target.shape)
    ms = np.ones(target.shape[0])
    mismatches = np.zeros(target.shape[0])
    t0 = dt()
    for i in range(target.shape[0]):
        m, previous, match = _single_step_multibuffer(target = target[i], 
                                                      band = band if is_num(band) else band[i], 
                                                      rounding_band = rounding_band,
                                                      point_values = point_values[i], vol = volatilities[i], 
                                               correlations = correlations[i], previous = previous, unit = unit, prev_m = m, risk_band = risk_band)
        ds[i] = previous
        ms[i] = m
        mismatches[i] = match
        if i % 1000 == 0:
            t1 = dt()
            print(i, t1 - t0)
            t0 = t1
    if data is None:
        return Dict(mult = ms, data = ds, mismatch = mismatches)
    else:
        return Dict(mult = np.concatenate([mult, ms]), 
                    data = np.concatenate([data, ds]), 
                    mismatch = np.concatenate([mismatch, mismatches]))

_multibuffer.output = ['data', 'mult', 'mismatch']

def _subset_multibuffer(subset, target, band, unit, correlations, volatilities, point_values, data = None, mult = None, mismatch = None, risk_band = 0.1, rounding_band = 0):
    if is_nums(subset):
        ids = subset
        keys = target.columns[ids] if is_df(target) else None
    elif is_strs(subset):
        keys = subset
        ids = Dict(zip(target.columns, range(len(target.columns))))[tuple(keys)] if is_df(target) else None
    else:
        keys = ids = None
    target, band, unit, correlations, volatilities, point_values, data , mult, mismatch, risk_band, rounding_band = _subset([target, band, unit, correlations, volatilities, point_values, data, mult, mismatch, risk_band, rounding_band], ids = ids, keys = keys)
    return _multibuffer(target = target, band = band, unit = unit, correlations = correlations, volatilities = volatilities, point_values = point_values, data = data, mult = mult, mismatch = mismatch, risk_band = risk_band, rounding_band = rounding_band)


@loop(tuple)
def _to_target(value, target = None):
    if isinstance(value, list) and is_tss(value):
        value = df_concat(value)
    if is_df(target) and is_df(value):
        value = ffill(df_reindex(value, target))
    return value
        

def multibuffer(target, band, unit, correlations, volatilities, point_values, data = None, mult = None, mismatch = None, 
                risk_band = 0.1, subset = None, subset_mult = None, subset_mismatch = None, correlations_index = None, rounding_band = 0):
    """
    performs a buffering of a target position but aiming to target a given level of risk
    
    TargetRisk is assumed to be given as 
    
    single_position_volatilities = target * point_values * volatilities
    portfolio_risk = single_position_volatilities^T x  correlations x single_position_volatilities

    first_attempt = buffer(target, band = band / (volatilities * point_values))
    
    EXCEPT
    
    first_attempt_risk = 
    
    
    :Parameters:
    ------------
    - target: df
        The target position
    
    - band: float/df
        The buffer on the position. See single buffer function for explanation
    
    - 
        
        
    
    :Example:
    ---------
    >>> from pyg import *
    >>> target = get_data('data','long_only', item = 'lots') ## grab original target lots per each asset over time
    >>> band = get_data('data','long_only', item = 'band')
    >>> unit = 1.0
    >>> volatilities = get_data('data','long_only', item = 'fut_vol')
    >>> point_values = get_data('data','long_only', item = 'rpz')
    >>> correlations = get_data('data','long_only', item = 'covariance_zero')
    >>> buffered_lots = get_data('data','long_only', item = 'buffered_lots')
    >>> risk_band = 0.1
    >>> data = None ## output
    >>> mult = None ## output
    >>> subset = None
    
    >>> res = multibuffer(target, band, unit, correlations, volatilities, point_values, data = data, mult = mult, risk_band = risk_band, subset = subset)


    
    >>> res.mult.plot(title = 'adjustment over time to the risk')

    >>> print(dictable(ticker = list(res.data.columns), new = list(tover(res.data).values), b = list(tover(buffered_lots).values)).do(f12))
    >>> ticker = 'HGA Comdty'    
    >>> df_concat([target[ticker], res.data[ticker], buffered_lots[ticker]], ['fractional position', 'multi asset buffer', 'single asset buffer']).plot(title = ticker)
    
    
    In addition, you may want to ensure the risk in different sub-populations is maintained as well. 
    For this purpose, you can either run it with 
        - subset = [... list of all bonds]. This will subset the data to run on the bonds only
        - subset = dict(telecom = [list of telecom stocks], health = [list of health stocks], ...)
        This will return a multi-column mult dataframe with the multiplier per each sector and data = final positions for all stocks
                        
    :Example:
    ---------
    >>> res = multibuffer(target, band, unit, correlations, volatilities, point_values, data = data, mult = mult, risk_band = risk_band, subset = ['TYA Comdty', 'ESA Index'])
    
    
    """
    target = _to_target(target)
    volatilities, point_values, band = _to_target((volatilities, point_values, band), target)
    if is_df(target) and len(correlations.shape) == 3 and correlations_index is not None and len(correlations) == len(correlations_index):
        correlations = reindex_3d(correlations, index = target.index, original_index = correlations_index)

    if isinstance(subset, dict):
        res = {}
        for key in subset:
            value = subset[key]
            print('running', key, 'for', value)
            res[key] = _subset_multibuffer(subset = value, 
                                           target = target, 
                                           band = band, 
                                           unit = unit, 
                                           correlations = correlations, 
                                           volatilities = volatilities, 
                                           point_values = point_values, 
                                           data = data, 
                                           mult = subset_mult[key] if is_df(subset_mult) else mult, 
                                           mismatch = subset_mismatch[key] if is_df(subset_mismatch) else subset_mismatch, 
                                           risk_band = risk_band,
                                           rounding_band = rounding_band)
        rtn = Dict(subset_mult = df_concat([r.mult for r in res.values()], list(res.keys())),
                   subset_mismatch = df_concat([r.mismatch for r in res.values()], list(res.keys())),
                   data = df_concat([r.data for r in res.values()]),
                   mult = df_concat(sum([[res[s].mult] * len(subset[s]) for s in subset], []), sum(subset.values(), [])),
                   mismatch = df_concat(sum([[res[s].mismatch] * len(subset[s]) for s in subset], []), sum(subset.values(), []))
                   )
        rtn['data'] = rtn['data'][target.columns]
        rtn['mult'] = rtn['mult'][target.columns]
        rtn['mismatch'] = rtn['mismatch'][target.columns]
        return rtn
    else:
        res = _subset_multibuffer(subset = subset, target = target, band = band, unit = unit, correlations = correlations, volatilities = volatilities, point_values = point_values, 
                              data = data, mult = mult, mismatch = mismatch, risk_band = risk_band, rounding_band = rounding_band)
        res['subset_mult'] = None
        res['subset_mismatch'] = None
    return res

multibuffer.output = _multibuffer.output + ['subset_mult', 'subset_mismatch']

