import numpy as np
from pyg_timeseries._rolling import shift
from pyg_timeseries._ewm import ewma

ln2 = np.log(2)

def bar_r2(bar, o = 'open', h = 'high', l = 'low', c = 'close', out = 'yz', alpha = None, short_range = None, long_range = None):
    """

    Calculates a point-in-time R^2 estimation for variance from a bar.
    
    Yang Zhang original paper is here: http://www.atmif.com/papers/range.pdf
    Rogers et al, defining V_rs can be found here: http://www.skokholm.co.uk/wp-content/uploads/2016/01/RS_Yoon0.pdf (survey) and here https://www.skokholm.co.uk/wp-content/uploads/2016/01/R_Satchell_HLOC.pdf
    
    We follow its notations but 

    1) we choose not to remove drift... i.e.     
        In paper, estimate for variance is given by     
        V_c = mean((c - mean(c))**2)
        We ignore the drift correction and use
        V_c = mean(c**2)

    2) We do not actually calculate the mean, we calculate the daily R^2
    The process of calculating the average is left for later
    This is useful as user may want to 
        - "roll" the R^2 across multiple periods (future contracts rolling) prior to variance calculation
        - calculate an ewma average rather than a linear average in the paper
        
    Out:


    3) The distribution assumptions of all the fancy estimators may be false as there may be jumps etc. 
    The "true" vol estimation is actually close-close so you can calibrate to it.
    Calibration matches short-range vol estimation over a rolling long range. 
    We find that brings the estimators together in the mean while hardly touching the turnover of the slow yz estimator.

    :Parameters:
    ============
    df: pd.DataFrame
        bar 
    
    o,h,l,c: str
        column names for ohlc
    
    out: str default 'yz'
        type of r2 estimator required
        'rs': Roger Satchel estimator
        'cc': close-close estimator
        'gk': Garman Klass
        'yz': Yang Zhang 
    
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

    >>> long_range = 80
    >>> yz = bar_r2(bar, short_range = 20, long_range = long_range)
    >>> cc = bar_r2(bar, out = 'cc', short_range = 20, long_range = long_range)
    >>> rs = bar_r2(bar, out = 'rs', short_range = 20, long_range = long_range)
    >>> gk = bar_r2(bar, out = 'gk', short_range = 20, long_range = long_range)
    >>> yz_raw_vol = ewma(yz_raw, 20) ** 0.5
    >>> yz_vol = ewma(yz, 20) ** 0.5
    >>> cc_vol = ewma(cc, 20) ** 0.5
    >>> rs_vol = ewma(rs, 20) ** 0.5
    >>> gk_vol = ewma(gk, 20) ** 0.5

    >>> dict(yz_raw_mean = yz_raw_vol.mean(), yz_mean = yz_vol.mean(), cc_mean = cc_vol.mean())
    >>> dict(cc_to = tover(cc_vol), gk_to = tover(gk_vol), rs_to = tover(rs_vol), yz_to = tover(yz_vol), yz_raw_to = tover(yz_raw_vol))
    
    >>> assert tover(cc_vol)  > tover(yz_vol)        
    df_concat([cc_vol, yz_vol, yz_raw_vol, rs_vol, gk_vol], ['cc', 'yz', 'yz_raw', 'rs', 'gk']).plot()    

    df_concat([c_vol, rs_vol], ['c', 'rs']).plot()    
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
    res = dict(yz = V_yz, gk = V_gk, p = V_p, c = V_c, cc = V_cc, rs = V_rs, o = V_o)[out]
    if long_range and short_range and out!='cc':
        cc = ewma(ewma(V_cc, short_range) ** 0.5, long_range)
        rr = ewma(ewma(res, short_range) ** 0.5, long_range)
        ratio = cc/rr
        res = ratio * res
    return res

