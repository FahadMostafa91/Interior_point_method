# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 23:02:01 2022

@author: rvenkata
"""
import math
import numpy as np

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
''' 
Run the program
'''
def main():
    A = np.array([[1.,2.,3.,4.],[2.,5.,6.,7.],[3.,6.,10.,1.],[4.,7.,1.,200.]])  
    print(mychol(A))
