'''
Functions to deal with data, i.e. read, write, convert, format, ...
'''

import numpy as np 

def signal_ar_to_nn(s, d):
    N = len(s)
    M = len(d)    
    D = max(d)  
    X = np.zeros((N-D, M))
    y = np.zeros( N-D)
    for n in range(D, N):
        y[n-D] = s[n]
        for k in range(0, M):
            X[n-D, k] = s[n - d[k]]
    return X, y    
    
# NEEDS UNIT TESTS!
# Example:
# d = [3,5]
# s = [1,2,3,4,5,6,7,8,9]
# y = [6,7,8,9]
# X = [[3,1],[4,2],[5,3],[6,4]]

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

if __name__ == "__main__":
    ok = test_datatools()
    assert(ok)

    

