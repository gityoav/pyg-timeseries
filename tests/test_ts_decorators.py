from pyg_timeseries import compiled
from pyg_base import getargspec


def test_ts_compiled():
    def function(a, b):
        return a + b
    f = compiled(function)
    assert f.fullargspec == getargspec(f)
    assert f(1,2) == 3
