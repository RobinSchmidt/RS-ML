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
    
    
    # for n in range(0, N-D):
    #     y[n] = s[n+D]
    #     for k in range(0, M):
    #         X[n, k] = s[n + d[k]]
    # return X, y

            

