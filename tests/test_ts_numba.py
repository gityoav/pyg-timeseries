import numba
import numpy as np

def example(arr, i, j):
    arr = arr.copy()
    arr[i,j,1:] = arr[i,j,:-1]
    return arr    
    
numba_example = numba.jit(example)

def test_numba():
    arr = np.random.normal(0,1,(3,3,3))
    i = 0; j = 1
    expect = example(arr, 0,1)    
    get = numba_example(arr, 0,1)
    get[i,j,:]
    expect[i,j,:]
    assert get[i,j,-1]!=expect[i,j,-1]
