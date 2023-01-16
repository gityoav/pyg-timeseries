import numpy as np
from pyg_base import calendar
from pyg_timeseries._rolling import shift, diff
from pyg_timeseries._ewm import ewma
import pandas as pd

ln2 = np.log(2)


def bar_daily_to_eod(bar, o = 'open', v = 'volume', cal = None, day_start = None, day_end = None):
    """
    Constructs an eod-bar from intraday data bar ASSUMING the bar provided at each intraday is actually a daily intraday bar... 
    i.e. every bar is actually the "day so far" bar.
    - hence, a day changes when the open changes
    - or when the volume drops
    
    if neither is provided, uses simple date to decide transition.
    
    ASSUMPTION: conversion to actual date is done using the calendar.trade_date function.
    
    Parameters
    ----------
    bar : pd.DataFrame
        bar timeseries
    o : str, optional
        name of the open column, default 'open'.
    v : str, optional
        name of the volume colume. The default is 'volume'.

    >>> from pyg import *; from pysys import *
    >>> bar = pd_read_npy(data_path('bc/ESA Index/2023/H/intraday.npy')); v = 'volume'; o = 'open'; cal = None
    >>> bar_eod(bar)

    >>> bar = pd_read_npy(data_path('bc/TYA Comdty/2023/H/intraday.npy')); v = 'volume'; o = 'open'; cal = None
    >>> bar_eod(bar)
    >>> o = 'open'; v = 'volume'

    """
    if len(bar) == 0:
        return bar
    elif len(bar) == 1:
        eod = bar
    else: #remove duplicates based on open/oi, always keep latest point
        if not v and not o:
            mask = diff(bar.index.day.values.astype(float),-1)!=0
        elif v and o:
            vmask = diff(bar[v],-1) > 0
            omask = diff(bar[o],-1)!=0
            omask.iloc[-1] = False
            mask = np.maximum(omask, vmask)        
        elif o:
            omask = diff(bar[o],-1)!=0
            omask.iloc[-1] = False
            mask = omask
        elif v:
            mask = vmask = diff(bar[v],-1) > 0  
        mask.iloc[-1] = True
        eod = bar[mask]
    cal = calendar(cal)    
    eod.index = [cal.trade_date(d, adj = 'f', day_start = day_start, day_end = day_end) for d in eod.index] 
    date_repeated = np.concatenate([eod.index[:-1] == eod.index[1:], np.array([False])])
    eod = eod[~date_repeated]
    last_date = cal.trade_date(bar.index[-1], adj = 'f', day_start = day_start, day_end = day_end)
    if last_date > eod.index[-1]:
        last_bar = bar.iloc[-1:]
        last_bar.index = [last_date]
        eod = pd.concat([eod, last_bar])
    return eod


def bar_r2(bar, o = 'open', h = 'high', l = 'low', c = 'close', model = 'yz', alpha = None, short_range = None, long_range = None):
    """

    Calculates a point-in-time R^2 estimation for variance from a bar.
    
    Yang Zhang original paper is here: http://www.atmif.com/papers/range.pdf
    Rogers et al, defining V_rs can be found here: http://www.skokholm.co.uk/wp-content/uploads/2016/01/RS_Yoon0.pdf (survey) 
    and here https://www.skokholm.co.uk/wp-content/uploads/2016/01/R_Satchell_HLOC.pdf
    
    We follow Yang Zhang notations but be aware of implementation differences:

    1) we choose not to remove drift... i.e.     
        In YZ paper, estimate for variance is given by     
        V_c = mean((c - mean(c))**2)
        We ignore the drift correction and use
        V_c = mean(c**2)

    2) We do not calculate the variance estimate, we calculate the daily R^2
    The process of calculating the average is left for later as (unbiased) mean of daily R^2 over a period.
    This is useful as user may want to 
        - "stitch together" the R^2 across multiple time series (e.g. future contracts) prior to variance calculation
        - calculate an ewma average rather than a linear average in the paper
        
    3) The distribution assumptions of all the fancy estimators may be false as there may be jumps, auto correlation etc. 
    The "reference" vol estimation is actually the close-close so you can choose to calibrate to it.
    Calibration matches short-range vol estimation over a rolling long range. 
    We find that calibration 
        - brings the estimators together in the mean.
        - hardly changing the turnover of the slow yz/rs/gk estimators.

    :Parameters:
    ============
    bar: pd.DataFrame
        bar data
    
    o,h,l,c: str
        column names for ohlc
    
    model: str 
        type of r2 estimator required:
        'cc': close-close estimator
        'p' : Parkinson estimator 
        'rs': Roger Satchel estimator 
        'gk': Garman Klass estimator
        'yz': Yang Zhang (default) 
    
    alpha: float
        Only applied to yz estimator, determines ratio between o-c variance and o-h-c & o-l-c variance estimators
    
    short_range, long_range: int
        If supplied, will be used to calibrate estimator so that long-term vol estimation is close to the c-c estimator
        This is to compensate for distribution assumptions made by gk, yz and rs estimators.
        We find this makes the estimators match the mean better while hardly impacting turnover    

    :Example:
    =========
    >>> from pyg import * 
    >>> rtn = pd.Series(np.random.normal(0,1,10000), [date for date in drange(-2500, 0, '1h') if date.hour>5 and date.hour<22][:10000]) ## 16 observations per day each N(0,1), so daily vol is 4
    >>> price = rtn.cumsum()
    >>> price.plot()
    >>> daily = price.resample('d')
    >>> daily.count()
    >>> bar = df_concat([daily.first(), daily.max(), daily.min(), daily.last()], ['open', 'high', 'low', 'close'])
    >>> yz_raw = bar_r2(bar)

    >>> long_range = 100
    >>> yz = bar_r2(bar, short_range = 40, long_range = long_range)
    >>> cc = bar_r2(bar, model = 'cc', short_range = 20, long_range = long_range)
    >>> rs = bar_r2(bar, model = 'rs', short_range = 20, long_range = long_range)
    >>> gk = bar_r2(bar, model = 'gk', short_range = 20, long_range = long_range)
    >>> yz_raw_vol = ewma(yz_raw, 20) ** 0.5
    >>> yz_vol = ewma(yz, 20) ** 0.5
    >>> cc_vol = ewma(cc, 20) ** 0.5
    >>> rs_vol = ewma(rs, 20) ** 0.5
    >>> gk_vol = ewma(gk, 20) ** 0.5

    >>> print('\nmean vol was generated to be near 4 but yz with no calibration underestimates vol:\n', dict(yz_raw_mean = yz_raw_vol.mean(), yz_mean = yz_vol.mean(), cc_mean = cc_vol.mean()))
    >>> print('\nturnover of yz and yz_raw should be similar:\n', dict(cc_tover = tover(cc_vol), rs_tover = tover(rs_vol), gk_tover = tover(gk_vol), yz_tover = tover(yz_vol), yz_raw_tover = tover(yz_raw_vol)))
    
    >>> df_concat([cc_vol, yz_vol, yz_raw_vol, rs_vol, gk_vol], ['cc', 'yz', 'yz_raw', 'rs', 'gk']).plot()    

    """
    o1 = bar[o]
    h1 = bar[h]
    l1 = bar[l]
    c1 = bar[c]
    c0 = shift(c1)
    
    o = o1 - c0
    c = c1 - o1    
    cc = c1 - c0 # = o + c
    u = h1 - o1
    d = l1 - o1
    hl = h1 - l1
    
    
    V_cc = cc ** 2
    V_rs = u * (u - c) + d * (d - c) # Eq 3 # This equation seriously underestimates daily variance in practice
    alpha = alpha or 1.34 ## eq 10 and following discussion
    k = (alpha - 1) / (alpha + 1) # eq 10, note that for our cases, n+1/n-1 == 1 approx
    
    V_o = o ** 2  # Eq 5 
    V_c = c ** 2  # Eq 6
    V_p = (u-d) ** 2 / (4 * ln2)  # Eq 2
    V_gk = V_o - 0.383 * V_c + 1.364 * V_p + 0.019 * V_rs # Eq 4, Garman Klass estimator
    V_yz = V_o + k * V_c + (1-k) * V_rs # Eq 9 

    ### some cleaning choice
    V_yz[np.isnan(V_yz)] = V_gk ## if we have some nan, default to Garman Klass variance
    V_yz[np.isnan(V_yz)] = V_cc ## if still nan, use close-close variance
    V_yz[hl <= 0] = V_cc ## if we have bad high-low data, revert to close-close variance
    res = dict(yz = V_yz, gk = V_gk, p = V_p, c = V_c, cc = V_cc, rs = V_rs, o = V_o)[model.lower()]
    if long_range and short_range and model.lower()!='cc':
        cc_vol_avg = ewma(ewma(V_cc, short_range) ** 0.5, long_range)
        res_vol_avg = ewma(ewma(res, short_range) ** 0.5, long_range)
        ratio = (cc_vol_avg/res_vol_avg)**2
        res = ratio * res
    return res

