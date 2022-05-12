from pyg_base import is_ts, loop
import pandas as pd

@loop(list, dict, tuple)
def _latest_value(value, date, default = None):
    if is_ts(value):
        v = value.loc[:date]
        if len(v):
            return v.iloc[-1].values 
        else:
            return default
    else:
        return value

def _latest_date_value(value, date, default = None):
    if is_ts(value):
        v = value.loc[:date]
        if len(v):    
            return v.index[-1], v.iloc[-1].values 
        else:
            return None, default
    else:
        return None, value


def ts_iterate(functor, variables, dates, data = None):
    """
    This is a generic functor iterating over timeseries.
    The output of previous calculation is a pd.DataFrame
    """
    if is_ts(dates):
        dates = dates.index
    ds = dates.drop(sorted(set(dates) & set(data.index))) if data is not None else dates
    previous_date = None
    values = []
    data_ = None
    maximal_date = data.index[-1] if is_ts(data) and len(data) else None
    for date in ds:
        variables_ = _latest_value(variables, date)
        if maximal_date and maximal_date > date:  ## we are back-filling here
            latest_date, latest_data = _latest_date_value(data, date)
            if latest_data is not None and latest_date is not None:
                if previous_date is None or latest_date > previous_date:
                    data_ = latest_data
        data_ = functor(data = data_, date = date, **variables_)
        previous_date  = date
        values.append(data_)
    result = pd.DataFrame(values, ds)
    if is_ts(data):
        result = pd.concat([data, result]).sort_index()
    return result       

