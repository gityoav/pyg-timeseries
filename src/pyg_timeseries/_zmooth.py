from pyg_base import is_int, pd2np
from pyg_timeseries._median import rolling_median
from pyg_timeseries._decorators import compiled, _data_state, first_
from pyg_timeseries._ewm import _w
import numpy as np

@pd2np
@compiled
def _zmooth(a, smooth, n, max_move, t0 = 0, t2 = 0, prev = np.nan, exc_zero = False):
    p = w = _w(n)
    res = np.empty_like(a)
    vol = 0 if t0 == 0 else np.sqrt(t2/t0)
    if max_move == 0:
        raise ValueError('must provide a positive max_move')
    for i in range(a.shape[0]):
        if np.isnan(a[i]):
            res[i] = np.nan
        else:
            if np.isnan(prev): ## prev value is nan, no reference to calculate "change" from
                res[i] = a[i]
            else:
                v = a[i] - prev
                sign = np.sign(v)
                if vol > 0 and abs(v) > max_move * vol:
                    if np.isnan(smooth[i]):
                        v = sign * max_move * vol 
                    elif np.sign(smooth[i] - prev) == sign:
                        v = smooth[i] - prev
                    else:
                        v = 0
                res[i] = prev + v
                ### now we update vol
                if exc_zero and v == 0:
                    pass
                else:
                    t0 = t0 * p + (1-w)
                    t2 = t2 * p + (1-w) * v**2
                    vol = 0 if t0 == 0 else np.sqrt(t2/t0)
            prev = res[i]
    return res, t0, t2, prev


def zmooth_(a, n, smooth = None, max_move = 4.2, exc_zero = False, instate = None):
    """
    A z-filter + smooth with state management
    See zmooth documentation for more details
    """
    state = instate or {}
    if smooth is None:
        smooth = a + np.nan
    elif is_int(smooth):
        smooth = rolling_median(a, smooth)
    return _data_state(['data', 't0', 't2', 'prev'], _zmooth(a = a, n = n, smooth = smooth, exc_zero = exc_zero, max_move = max_move,**state))

zmooth_.output = ['data', 'state']


def zmooth(a, n, smooth = None, max_move = 4.2, exc_zero = False, state = None):
    """
    This function combines a standard z-filter with a smooth
    smoothing smoothes data but also introduces a delay which we don't usually want
    
    So what we want to do is as follows:

    if a[i] - res[i-1] exceeds a threshold, 
        res[i] = smooth[i] provided it is in the direction of a[i] else res[i-1] + threshold[i]
    else:        
        res[i] = a[i]
    
    Finally:
    threshold[i] is defined as a multiple of rolling rms of ewmrms(a[i]-a[i-1],n)
    
    Note: 
    The usual z-filter is when smooth is not provided. 
    The problem with a z-filter is that it can get stuck near the old value if the jump is many standard deviations away
    

    >>> from pyg import * 
    >>> from pyg_timeseries import * 
    >>> rtn = np.random.normal(0,1,10000)
    >>> a = cumsum(rtn)
    
    ### now we add some noise..
    >>> mask5 = np.random.normal(0,1,10000)>1.5 
    >>> a[mask5] = (a + np.random.normal(0,5,10000))[mask5]
    >>> mask10 = np.random.normal(0,1,10000)>2    
    >>> a[mask10] = (a + np.random.normal(0,10,10000))[mask10]
    >>> smooth = rolling_median(a, 10)
    >>> z = zmooth(a = a, smooth = smooth, n = 18)
    >>> assert ts_cor(diff(z), diff(a)) > 0.5
    >>> assert ts_cor(diff(smooth), diff(a)) < 0.3    

    ## another way is by looking at when the maximum correlation occurs
    >>> shifted_cor = [ts_cor(shift(a,i), smooth) for i in range(10)]
    >>> assert shifted_cor.index(max(shifted_cor)) > 0 ## there is a delay introduced...
    >>> shifted_cor = [ts_cor(shift(a,i), z) for i in range(10)]
    >>> assert shifted_cor.index(max(shifted_cor)) == 0

    :Example: using prev_value + 4 stdev as next value
    ---------    
    >>> z = zmooth(a = a, smooth = None, n = 18, max_move = 2)
    >>> assert ts_cor(diff(z), diff(a)) > 0.8
    >>> shifted_cor = [ts_cor(shift(a,i), z) for i in range(10)]
    >>> assert shifted_cor.index(max(shifted_cor)) == 0
   
    :Example:
    --------
    >>> n = 10; max_move = 4
    """    
    state = state or {}
    if smooth is None:
        smooth = a + np.nan
    elif is_int(smooth):
        smooth = rolling_median(a, smooth)
    return first_(_zmooth(a = a, n = n, smooth = smooth, exc_zero = exc_zero, max_move = max_move, **state))

