# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 03:23:13 2023

@author: morte
"""
import numpy as np
from Roulette_Wheel import Roulette_Wheel
from Single_Point_Crossover import SinglePointCrossover
from Double_Point_Crossover import DoublePointCrossover
from Uniform_Crossover import UniformCrossover
     
def crossover(p1, p2):
    pSinglePoint = 0.35
    pDoublePoint = 0.35
    pUniform = 1 - pSinglePoint - pDoublePoint
    probabilities = np.array([pSinglePoint, pDoublePoint, pUniform])
    Method = Roulette_Wheel(probabilities)
    
    if Method == 0: 
        offspring1, offspring2 = SinglePointCrossover(p1, p2)
    elif Method == 1:
        offspring1, offspring2 = DoublePointCrossover(p1, p2)
    elif Method == 2:
        offspring1, offspring2 = UniformCrossover(p1, p2)
            
    return offspring1, offspring2