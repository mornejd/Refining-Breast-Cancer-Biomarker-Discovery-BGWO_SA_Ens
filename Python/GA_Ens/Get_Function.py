# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 01:13:45 2023

@author: morte
"""
import numpy as np
from OF4 import F4
from OF5 import F5
from OF6 import F6
from OF7 import F7



def Get_F(F):
               
    if F=="F7":
            F_obj = F7
            Dim =10629
        
    
    return F_obj,Dim 
            