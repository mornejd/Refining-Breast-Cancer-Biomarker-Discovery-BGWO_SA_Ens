# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 04:17:55 2023

@author: morte
"""
import numpy as np

def SinglePointCrossover(p1,p2):
    
    c = np.random.randint(len(p1))
    offspring1 = (np.hstack((p1[:c], p2[c:])))
    offspring1=np.array([offspring1])
    offspring2 = np.array(list(np.hstack((p2[:c], p1[c:]))))
    offspring2=np.array([offspring2])
    
    return offspring1, offspring2