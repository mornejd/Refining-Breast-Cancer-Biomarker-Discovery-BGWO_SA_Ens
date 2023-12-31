# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 02:49:02 2023

@author: morte
"""

import random
import numpy as np

def Roulette_Wheel(P):

    random_num = random.random()
    cumulative_prob = np.cumsum(P)
    ind=np.where(random_num<=cumulative_prob)
    first=ind[0][0]
    return first

