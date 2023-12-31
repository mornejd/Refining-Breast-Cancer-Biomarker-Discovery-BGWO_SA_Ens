# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 04:27:28 2023

@author: morte
"""
import numpy as np

def UniformCrossover(p1,p2):
    alpha=np.random.randint(0,2,(1,len(p1)))
    offspring1=(alpha*p1)+((1-alpha)*p2)
    offspring2=(alpha*p2)+((1-alpha)*p1)

    return offspring1, offspring2