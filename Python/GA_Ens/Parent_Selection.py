# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 03:14:43 2023

@author: morte
"""
from Roulette_Wheel import Roulette_Wheel

def Parent_Selection(population, P):
    index_1 = Roulette_Wheel(P)
    Index_2 = Roulette_Wheel(P)
    parent1=population[index_1]
    parent2=population[Index_2]
    while (parent1 == parent2).all():
        Index_2 = Roulette_Wheel(P)
        parent2=population[Index_2]

    return parent1, parent2