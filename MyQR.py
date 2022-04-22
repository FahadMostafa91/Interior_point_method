import numpy as np
import math
#from scipy import io, integrate, linalg, signal
#import scipy.linalg as la
#from scipy.sparse.linalg import eigs


def QR_householder(A):
    
    '''
     First find the size of the matrix
    '''
    m = A.shape[0]
    n = A.shape[1]
    beta = np.zeros(n)
    j=0
    for j in range(0, n):
        [v,beta[j]] = myhouse(A[j:m,j],m-j) #v is a row vector
        print(v)
        print(beta)
        A[j:m,j:n] = np.matmul((np.eye(m-j)-beta[j]*np.outer(v,v)),A[j:m,j:n])

        if j < m-1:
            A[j+1:m,j] = v[1:m-j]
    [Q,R,rank] = unpack(A,beta)
    return Q,R,rank

def myhouse(x,n):
    sigma = x[1:n] @ x[1:n]
    a1 = np.array([1.])                    
    v = np.concatenate((a1,x[1:n]))
    if abs(sigma) < 1e-10:
        beta = 0
    else:
        mu = math.sqrt(x[0]**2 + sigma)
        if x[0] <= 0:
            v[0] = x[0] - mu
        else:
            v[0] = -sigma/(x[0]+mu)
        beta = 2*v[0]**2/(sigma+v[0]**2)
        v = v/v[0]
    return v,beta

def unpack(H,beta):
    
    m = H.shape[0]
    n = H.shape[1]
    Q = np.eye(m)
    min_dim = min(m,n)
    R = np.zeros((m,n))
    rank = 0
    for i in range(0,min_dim):
        if abs(H[i,i]) > 1e-9:
            R[i,i:n] = H[i,i:n]
            # if 'v1' in locals():
            #     v1.clear()
            # if 'P' in locals():    
            #     P.clear()
            a1 = np.array([1.])                    
            v1 = np.concatenate((a1,H[i+1:m,i]))
            Q[:,i:m] = np.matmul(Q[:,i:m],np.eye(m-i)-beta[i]*np.outer(v1,v1))
            rank = i+1
        else:
            rank = i
            break
    return Q,R,rank
''' 
The main code
'''        
#A = np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.],[10.,11.,12.]])
A = np.array([  [  0.0623,    4.3117,    1.8908],
   [-0.6859,    7.4452,   -1.5081],
    [0.5314,  -11.7046,   -0.9195]])
[Q,R,rank] = QR_householder(A)    
print(np.matmul(Q.T,Q))
print(np.matmul(Q,R))
print(rank)