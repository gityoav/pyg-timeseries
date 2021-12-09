# pyg-timeseries
pandas is great but pyg-timeseries introduces a few improvements. 

* pyg is designed so that for dataframes/series without nans, it matches pandas exactly 
* consistent treatments of nan's: unlike pandas, pyg ignores nans everywhere in its calculations.
* np.ndarray and pandas dataframes are treated the same and pyg operates on np.arrays seemlessly
* state-management: pyg introduces a framework for returning not just the timeseries, but also the state of the calculation. This can be fed into the next calculation batch, allowing us not to have to 're-run' everything from the distant past.

pip install from https://pypi.org/project/pyg-timeseries/
