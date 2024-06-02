'''
Functions to deal with data, i.e. to read, write, convert, format, extract, 
clean up, etc.
'''

import numpy as np 


def delayed_values(s, n, d):
    N = len(s)
    M = len(d)    
    D = max(d)
    assert(n <  N)
    assert(n >= D)
    x = np.zeros(M)
    for k in range(0, M):
        x[k] = s[n - d[k]]
    return x

# ToDo:
#
# - Maybe lift the n >= D restriction for n by assuming s[n-d[k]] = 0 for 
#   n-d[k] < 0. That means, we imagine the signal values to be zero for indices
#   less that zero.
# - Maybe get rid of N,D - they are used only once


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
# Write a function delayed_values(s, n, d) and use it here like 
# X[n-D, :] = delayed_values(s, n, d) to replace the inner for-loop. The 
# rationale is that this delayed_values function will be needed during 
# recursive signal generation as well

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

    

