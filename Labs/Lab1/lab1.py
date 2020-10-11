import numpy as np
import matplotlib.pyplot as plt

def poly_fit(x, y, p):
    
    X = np.zeros([len(x), p])

    for row in range(0, X.shape[0]):
        for col in range(0, X.shape[1]):
            X[row, col] = x[row] ** col
    
    return np.matmul (np.matmul( np.linalg.inv( np.matmul(np.transpose(X), X) ), np.transpose(X) ), y).squeeze()



def poly(a, x):
    res = 0
    for p in range(0, len(a)):
        res += a[p] * x**p
        
    return res



def poly_vector(a, x_):
    y_ = np.zeros([len(x_)])
    for i in range(0, len(x_)):
        y_[i] = poly(a, x_[i])
        
    return y_


def SSE(y, y_):
    if len(y.shape)>1:
        y = np.squeeze(y)

    return np.linalg.norm(y-y_, ord=2)**2



def compute_SSE(y,x,a):
    y_hat=poly(a,x)
    SSE=np.linalg.norm(y-y_hat, ord=2)**2
    return SSE

def indic(y,x,a):
    SSE=compute_SSE(y,x,a)
    print('\n')
    print('B coeficients are:')
    for i in range(0,len(a)):
        print(a[i])
    print('\n')
    print('SSE is:')
    print(SSE)
    return
    