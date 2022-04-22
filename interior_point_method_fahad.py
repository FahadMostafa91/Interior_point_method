# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 15:40:37 2022

@author: gmostafa
"""
import numpy as np
from ls_with_qr import ls_with_qr

# define the jacbian
def Jacobian(A,Q,x,s):
    m = A.shape[0]
    n = A.shape[1]
    Z12 = np.zeros((m,m))
    Z13 = np.zeros((m,n))
    Z32 = np.zeros((n,m))
    S = np.diagflat(s)
    X = np.diagflat(x)
    I = np.identity(n)
    Jk1 = np.concatenate((A,-Q,S), axis=0)
    Jk2 = np.concatenate((Z12,A.T,Z32), axis=0)
    Jk3 = np.concatenate((Z13,I,X), axis=0)
    Jk = np.concatenate((Jk1,Jk2,Jk3), axis=1)
    return Jk

# problem setting
def problem_fun(x,lambdaa,s,A,b,Q,c,mu):

    n = A.shape[1]
    r0 = np.concatenate((b-np.matmul(A,x),np.matmul(Q,x)+c-s-np.matmul(A.T,lambdaa)),0)
    r = np.concatenate((r0, mu*np.ones((n,1))-np.multiply(s,x)),0)
    return -r

# interior point method
def Interior_point_method(f,A,b,Q,c,x,lambdaa,s,rho,mu,mu_min):
    m = A.shape[0]
    n = A.shape[1]
    rk = -problem_fun(x,lambdaa,s,A,b,Q,c,mu)
    t = (np.linalg.norm(rk))**2
    dk = np.zeros((2*n+m,1))
    k=int(1)
    epsilon = 1e-8
    kmax = int(1000)
    while (t > epsilon and k < kmax and mu > mu_min):
        Jk = Jacobian(A,Q,x,s)

        dk= ls_with_qr(Jk,rk)
        dxk = dk[0:n,0:1]
        dlambdak= dk[n:m+n,0:1]
        dsk=dk[n+m:2*n+m,0:1]
        z= 0.9*np.linalg.norm(np.matmul(rk.T,Jk))*np.linalg.norm(dk)
      
    ############ Find alpha0  #########################################
        zz = -np.divide(x,dxk)
        I = np.argwhere(dxk < 0)       
        J = I[:,0]
    
        zzz = -np.divide(s,dsk)
        II = np.argwhere(dsk < 0)
       
        JJ = II[:,0]       
       
        junk = np.concatenate((zz[J],zzz[JJ],[[1.0]]),0)
    
        alpha0 = 0.995*np.min(junk)     
        L = np.linalg.norm(problem_fun(x+ alpha0*dxk,lambdaa+alpha0*dlambdak,s+alpha0*dsk,A,b,Q,c,mu))**2
        R = t-alpha0*z
        L_min = L
        index = 0
        j = 0
        while L > R and j < 20:
            j = j+1
            L = np.linalg.norm(problem_fun(x+ 2**(-j)*alpha0*dxk,lambdaa+ 2**(-j)*alpha0*dlambdak,s + 2**(-j)*alpha0*dsk,A,b,Q,c,mu))**2
            R = t-2**(-j)*alpha0*z        
            if L < L_min:
                L_min = L
                index = j
                
    # update the solution 
        x = x + 2**(-index)*alpha0*dxk 
        lambdaa = lambdaa + 2**(-index)*alpha0*dlambdak
        s = s + 2**(-index)*alpha0*dsk  
        k =k+1
        mu = rho*mu
        rk = -problem_fun(x,lambdaa,s,A,b,Q,c,mu)
        t = (np.linalg.norm(rk))**2
    return x, lambdaa, s

# input the data from IP problem 01
def problem_no1():
    A = np.array([[1.,2,-1,1],[2,-2,3,3],[1,-1,2,-1]])  # impiut the data matrix
    b = np.array([[0.],[9],[6]]) # right side 
    m = A.shape[0]
    n = A.shape[1]

    ## Phase numbe-1
    Q = np.identity(n)
    c = -np.ones((n,1))
    x = np.ones((n,1))
    lambdaa = np.zeros((m,1))
    s = np.ones((n,1))
    
    rho = 0.5
    mu = 100.0
    mu_min = 1e-8
    
    x, lambdaa, s = Interior_point_method(problem_fun,A,b,Q,c,x,lambdaa,s,rho,mu,mu_min)
    
    ## Phase number 2
    Q = np.zeros((n,n))
    c = np.array([[-3.],[1],[3],[-1]])
    x, lambdaa, s = Interior_point_method(problem_fun,A,b,Q,c,x,lambdaa,s,rho,mu,mu_min)
    
    
    print('solution = ', x)
    print('Constraint satisfaction = ', np.matmul(A,x))
    print('Cost = ', float(np.matmul(c.T,x)))
    return
    
    
def main():
    problem_no1()    
if __name__ == "__main__":
    main()   