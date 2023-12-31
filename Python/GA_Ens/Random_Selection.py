# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 18:26:57 2023

@author: morte
"""
import random as rnd
from Random_Creation import Random_Creation
import numpy as np
           
def Random_Selection(rand, N, Dim):
    UB1 = 1
    LB1 = -2.8 
    UB2 = 3
    LB2 = -1

    if rand == 0:
        Variable = np.random.rand() * (UB1 - LB1) + LB1
        if Variable > 0:
            Variable = 1
        else:
            Variable = 0
    elif rand == 1:
        Variable = rnd.random()
        if Variable > 0.5:
            Variable = 1
        else:
            Variable = 0
    elif rand == 2:
        Variable = Random_Creation()
    elif rand == 3:
        Variable = np.random.rand() * (UB2 - LB2) + LB2
        if Variable > 0:
            Variable = 1
        else:
            Variable = 0
            
    return Variable
                
                
                
                
    
    
    