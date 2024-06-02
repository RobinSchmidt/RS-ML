'''
Functions to deal with data, i.e. to read, write, convert, format, extract, 
clean up, etc.
'''

import numpy as np 


def delayed_values(s, n, d):
    '''
    Given a time series signal s, an index n into s and an array of delay 
    values d = [d0, d1, d2, ...], this function extracts the signal values 
    s[n-d0], s[n-d1], s[n-d2], ... and puts them into an array and returns that 
    array.

    Parameters
    ----------
    s : array of signal values (typically float)
        The signal from which we want to extract delayed values.
    n : int
        The index with respect to which we wnat to extract the delayed values.
    d : array of int
        The delay values/times. They should all be greater than zero.
        
    Returns
    -------
    x : array of signal values (typically float, same as s)
        The array of delayed values from the signal s.
    '''
    assert(n <  len(s))
    assert(n >= max(d))
    M = len(d)       
    x = np.zeros(M)
    for k in range(0, M):
        x[k] = s[n - d[k]]
    return x

# ToDo:
#
# - Maybe lift the n >= max(d) restriction for n by assuming s[n-d[k]] = 0 for 
#   n-d[k] < 0. That means, we imagine the signal values to be zero for indices
#   less that zero.


def signal_ar_to_nn(s, d):
    N = len(s)
    M = len(d)    
    D = max(d)  
    X = np.zeros((N-D, M))
    y = np.zeros( N-D)
    for n in range(D, N):
        y[n-D] = s[n]
        X[n-D] = delayed_values(s, n, d)
    return X, y    
    
# ToDo:
#
# - Maybe write the loop as "for n in range(0, N-D)". It's just a csometic 
#   thing and I'm not sure what is better, though. Maybe it's better the way it
#   currently is. Dunno.

#==============================================================================
# Unit Tests:

def test_datatools():
    ok = True
    
    # Test signal_ar_to_nn:
    d  = [3,5]                                        # Array of delay times
    s  = [1,2,3,4,5,6,7,8,9]                          # Input signal
    Xt = np.array([[3.,1.],[4.,2.],[5.,3.],[6.,4.]])  # Target for X
    yt = np.array([6.,7.,8.,9.])                      # Target for y
    X, y = signal_ar_to_nn(s, d) 
    ok &= np.array_equal(X, Xt)
    ok &= np.array_equal(y, yt)
    
    return ok

# If the current file is being run as script, we execute the unit tests:
if __name__ == "__main__":
    ok = test_datatools()
    if ok:
        print("Unit tests for datatools.py passed without errors.")
    else:
        print("!!! WARNING! Unit tests for datatools.py FAILED !!!")
    #assert(ok)

    

