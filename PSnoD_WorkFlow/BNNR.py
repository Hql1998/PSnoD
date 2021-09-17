from scipy.linalg import svd
from numpy.linalg import norm
import numpy as np


def svt(Y, T):
    
    [U, S, V] = svd(Y, full_matrices=False)
    S_t = S
    S = np.eye(S_t.shape[0])
    for i in range(S_t.shape[0]):
        S[i, i] = S_t[i]
    tem = (abs(S) - T) * ((abs(S) - T)>0).astype(np.int)
    S = (S > 0).astype(np.int) * tem
    X = U.dot(S).dot(V)
    return X

def bnnr(
    T, mask,
    
    alpha=1, beta=10, 
    
    tol1=2*10**-3, tol2=10**-5,
    maxiter = 500, 
    a=0, b=1
):
    """ BNNR: bounded nuclear norm regularization.
    Inputs:
           T                  - the target matrix with only known entries and the unobserved entries are 0.
           mask               - a matrix recording the observed positions in the target matrix.
           alpha, beta        - parameters needed to give.
           tol1, tol2         - tolerance of termination conditions.
           maxiter            - maximum number of iterations.
           a, b               - the left and right endpoints of the bounded interval.

    Outputs:
           T_recovery         - the completed matrix.
           iter               - the number of iterations."""
    X = T
    W = X
    Y = X
    
    i = 0
    stop1 = 1
    stop2 = 1
    
    while (stop1 > tol1 or stop2 > tol2 ):
        tran = (1/beta) * (Y + alpha * (T * mask)) + X
        W = tran - (alpha / (alpha + beta)) * (tran * mask)

        W[W < a] = a
        W[W > b] = b

        #the process of computing X
        X_1 = svt(W - 1/beta * Y, 1/beta)

        #the process of computing Y
        Y = Y + beta * (X_1 - W)

        stop1_0 = stop1
        stop1 = norm(X_1 - X) / norm(X)
        stop2 = abs(stop1 - stop1_0) / max(1, abs(stop1_0))

        X = X_1
        i = i+1

        if i >= maxiter:
            print('reach maximum iteration~~do not converge!!!')
            break

    return W, i
