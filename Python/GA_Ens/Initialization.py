# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 02:40:46 2023

@author: morte
"""
import numpy as np
from Random_Creation import Random_Creation
from Random_Selection import Random_Selection
import random as rnd

def Initialization(N,Dim):
    X=np.zeros(shape=(N,Dim))
    for i in range(N):
        rand=rnd.randint(0,3)
#        rand=0
        sol=[]
        flag=1
        while flag==1:
            for j in range(Dim):
                Variable=Random_Selection(rand, N, Dim)
                sol.append(Variable)
            if sum(sol)>0:
                flag=0
                break
        print(sum(sol))
        X[i,:]=np.array(sol)
    return X
        
# X=np.random.randint(0,2,(N,Dim))
    