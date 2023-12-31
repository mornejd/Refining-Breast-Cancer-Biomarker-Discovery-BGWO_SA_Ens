# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 18:43:57 2023

@author: morte
"""
import numpy as np
import math
import random


def Mutate(x,mu):
    N=np.size(x)
    nmu=math.ceil(mu*N)
    j=random.sample(range(N), nmu)
    y=x
    y[j]=1-x[j]
    y=np.array([y])
    
    return y
    