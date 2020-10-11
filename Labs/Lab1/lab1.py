import numpy as np
import matplotlib.pyplot as plt

def poly_fit(x, y, p):
    
    X = np.zeros([len(x), p])

    for row in range(0, X.shape[0]):
        for col in range(0, X.shape[1]):
            X[row, col] = x[row] ** col
    
    return np.matmul (np.matmul( np.linalg.inv( np.matmul(np.transpose(X), X) ), np.transpose(X) ), y)



def poly(a, x):
    res = 0
    for p in range(0, len(a)):
        res += a[p] * x**p
        
    return res