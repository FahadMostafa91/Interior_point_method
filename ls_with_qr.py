# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 14:53:16 2022

"""
import numpy as np
from MyQR import QR_householder
from Mychol import mychol 

def ls_with_qr(C,b):
    A = C.copy()
    m = A.shape[0]
    n = A.shape[1]

    [Q,R,rank] = QR_householder(A)

    b = np.matmul(Q.T,b)
    min_dim = min(m,n)

    if rank < min_dim: # rank A < min_dim; Minimum norm solution is needed
        ReducedR = R[0:rank,:]
        L = mychol(np.matmul(ReducedR,ReducedR.T)) # L is a lower triangular matrix such that LL' = reducedA

        w = forward(L,b[:rank]) # solve L w = b;; note that b is Q'b.

        z = backward(L.T,w)  # solve L'z = w;
        x = np.matmul(ReducedR.T,z)

    else:      # R is nonsingular and square
        x = backward(R,b[:n])
    return(x)


def forward(A,b):   # forward substitution code
    n = A.shape[1]
    bn = b.shape[0]
    if n != bn:
        print("Number of rows of A  and b must be equal and A must be square");

    x = np.zeros((n,1))

    x[0] = b[0]/A[0,0]

    for i in range(1,n):
        sum = np.matmul(A[i,:i],x[:i])
        x[i] = (b[i] - sum)/A[i,i]
    
    return x

def backward(A,b):   # backward substitution code

    n = A.shape[1]
    bn = b.shape[0]
    if n != bn:
        print("Number of rows of A  and b must be equal and A must be square")

    x = np.zeros((n,1))

    x[n-1] = b[n-1]/A[n-1,n-1]

    for i in range(n-2,-1,-1):
        sum = np.matmul(A[i,i+1:n],x[i+1:n])
        x[i] = (b[i] - sum)/A[i,i]
    return x

''' 
The main code
'''   
def main():     
    # A = np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.],[10.,11.,12.]])
    # print(A)  
    # b = np.reshape(np.array([-1.,-5.,2.,1.]),(4,1))
    # print(b)
    # x = ls_with_qr(A,b)
    # print(x)
    A = np.array([[1.8162 ,   0.7361 ,  -1.6029],
                 [1.7961,    0.3619 ,  -0.6157],
                 [1.4627 ,  -0.0455  ,  0.3997],
                 [4.8991  ,  1.1483  , -2.1063]])
    print(A)  
#    b = np.reshape(np.array([-1.,-5.,2.,1.]),(4,1))
    b = np.reshape(np.array([-0.0301, -0.1649, 0.6277, 1.0933]),(4,1))
    print(b)
    x = ls_with_qr(A,b)
    print(x)
if __name__ == "__main__":  ## This command executes the main function
    main()     