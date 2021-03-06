from pyg_timeseries._rolling import _vec, _rolling_window
from pyg_base import Dict
from pyg_timeseries._decorators import first_, _data_state

__all__ = ['rolling_rank', 'rolling_rank_', 'expanding_rank']

def rolling_rank(a, n, axis = 0, data = None, state = None):
    """
    returns a rank of the current value within a given window, scaled to be -1 if it is the smallest and +1 if it is the largest
    - works on mumpy arrays too
    - skips nan, no ffill
    
    :Parameters:
    ------------
    a : array, pd.Series, pd.DataFrame or list/dict of these
        timeseries
        
    n: int
        window size 

    axis : int, optional
        0/1/-1. The default is 0.

    data: None.
        unused at the moment. Allow code such as func(live, **func_(history)) to work
    state: dict, optional
        state parameters used to instantiate the internal calculations, based on history prior to 'a' provided. 
    
    :Example:
    -------
    >>> from pyg import *; import pandas as pd; import numpy as np
    >>> a = pd.Series([1.,2., np.nan, 0., 4., 2., 3., 1., 2.], drange(-8))
    >>> rank = rolling_rank(a, 3)
    >>> assert eq(rank.values, np.array([np.nan, np.nan, np.nan, -1, 1, 0, 0, -1, 0]))
    >>> # 0 is smallest in [1,2,0] so goes to -1
    >>> # 4 is largest in [2,0,4] so goes to +1
    >>> # 2 is middle of [0,4,2] so goes to 0
    
    :Example: numpy equivalent
    ------------------------------------------
    >>> assert eq(rolling_rank(a.values, 10), rolling_rank(a, 10).values)  
    
    :Example: state management
    --------------------------------------------
    >>> a = np.random.normal(0,1,10000)
    >>> old = rolling_rank_(a[:5000], 10) # grab both data and state
    >>> new = rolling_rank(a[5000:], 10, **old)
    >>> assert eq(np.concatenate([old.data,new]), rolling_rank(a, 10))

    """
    import bottleneck as bn
    state = state or {}
    return first_(_rolling_window(a,n,n, func = bn.move_rank, axis = axis, **state))

def rolling_rank_(a, n, axis = 0, data = None, instate = None):
    """
    Equivalent to rolling_rank(a) but returns also the state variables.
    For full documentation, look at rolling_rank.__doc__    
    """
    import bottleneck as bn
    state = instate or {}
    return _data_state(['data','vec'],_rolling_window(a,n,n,func = bn.move_rank, axis = axis, **state))

rolling_rank_.output = ['data','state']


def expanding_rank(a, axis = 0):
    """
    returns a rank of the current value within history, scaled to be -1 if it is the smallest and +1 if it is the largest
    - works on mumpy arrays too
    - skips nan, no ffill
    
    :Parameters:
    ------------
    a : array, pd.Series, pd.DataFrame or list/dict of these
        timeseries
        
    axis : int, optional
        0/1/-1. The default is 0.

    :Example:
    -------
    >>> from pyg import *; import pandas as pd; import numpy as np
    >>> a = pd.Series([1.,2., np.nan, 0.,4.,2.], drange(-5))
    >>> rank = expanding_rank(a)
    >>> assert eq(rank, pd.Series([0, 1, np.nan, -1, 1, 0.25], drange(-5)))
    >>> #
    >>> # 2 is largest in [1,2] so goes to 1; 
    >>> # 0 is smallest in [1,2,0] so goes to -1 etc.
    
    :Example: numpy equivalent
    ------------------------------------------
    >>> assert eq(expanding_rank(a.values), expanding_rank(a).values)  
    

    """
    import bottleneck as bn
    return first_(_rolling_window(a,0,1, vec = None, func = bn.move_rank, axis = axis))
