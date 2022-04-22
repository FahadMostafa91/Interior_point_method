'''
Min norm problem
fahad mostafa @ ttu
code for min norm problem in optimization
'''
import numpy as np
import math


#-----------------------------------------------------------------------

def mychol(G):
 #   G = A.copy()

    n = G.shape[1]
    
    for j in range(0,n):
        G[j,j] = math.sqrt(G[j,j])
        G[j+1:n,j] = G[j+1:n,j]/G[j,j]
        for k in range(j+1,n):
            G[k:n,k] = G[k:n,k]-G[k:n,j]*G[k,j]
       
    for i in range(0,n):
        G[i,i+1:n] = np.zeros(n-i-1)  
    return(G)


#-----------------------------------------------------------------------
# forward substitution
def forward(L, b):
    n = L.shape[0]
    x = np.zeros((n,1))
    for i in range(0,n):
        x[i] = b[i]
        for j in range(0,i):
            x[i]=x[i]-(L[i, j]*x[j])
        x[i] = x[i]/L[i, i]
    return x

#-----------------------------------------------------------------------
# backward substitution

def back_sub(Ltraspose,z):
    
    n = Ltraspose.shape[0]
    print('n is:', n)
    x = [0]*n
    for i in range(n-1,-1,-1): #this refers to the rows; i goes 2,1,0
        for j in range(i+1,n): #j goes 1,2 @ i = 0
                               #j goes 2   @ i = 1
            z[i] = z[i] - Ltraspose[i,j]*x[j]
        x[i] = z[i]/Ltraspose[i,i]

    return x


#-----------------------------------------------------------------------
# min-norm problem 

def min_norm(A,b):
    L = mychol(np.matmul(A,A.T))
    z = forward(L,b)
    lam = back_sub(L.T,z)
    x = np.matmul(A.T,lam)

    return x


#-----------------------------------------------------------------------
def main():
    A = np.array([[1.,1.,1.],[1.,2.,3.]])
    b = np.array([[1.],[1.]])

#-----------------------------------------
# solution
    x = min_norm(A,b)
    print(x)
if __name__ == "__main__":  ## This command executes the main function
    main() 
#-------------------------------------------------------------------------
