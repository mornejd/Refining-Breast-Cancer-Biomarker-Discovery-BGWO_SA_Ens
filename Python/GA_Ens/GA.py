# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 02:38:34 2023

@author: morte
"""
import numpy as np
import math
from Initialization import Initialization
from Parent_Selection import Parent_Selection
from Crossover import crossover
from Mutation import Mutate
import pandas as pd
def GA(N,T,Dim,F_obj,df):
    
    Best_P=np.zeros(shape=(1,Dim))
    Best_FF=-(math.inf)
    Best_tresh=-(math.inf)
    last_weights=[]
    X=Initialization(N,Dim)
    Xnew=X.copy()
    Ffun=np.zeros(shape=(1,X.shape[0]))
    Ffun_new=np.zeros(shape=(1,Xnew.shape[0]))
    for i in range(len(X)):
        Ffun[0,i], Tre, Wei=F_obj(X[i,:],df)
        if Ffun[0,i]>Best_FF:
            Best_FF=Ffun[0,i]
            Best_P[0,:]=X[i,:]
            Best_tresh = Tre
    t=1
        
    conv=[]
    num_generations = 100
    mu = 0.2
    pc=0.8
    nc=2*round(pc*N/2)
    pm=0.3
    nm=round(pm*N)
    beta=8

    
    
    while t<T+1:
        
        P=np.exp(-beta*Ffun/(Best_FF))
        P = P/np.sum(P, dtype=np.float64)
        fitnesses=np.transpose(Ffun.copy())
        pop=X.copy()
        OffspringsPositions=np.zeros(shape=(round(nc/2),2,X.shape[1]))
        OffspringsFitnesses=np.zeros(shape=(round(nc/2),2,1))
        # offsprings=pd.Series()
        for k in range(round(nc/2)):
            parent1, parent2 = Parent_Selection(X, P)
            offspring1, offspring2 = crossover(parent1, parent2)
            OffspringsPositions[k][0]=offspring1
            OffspringsPositions[k][1]=offspring2
            Ffun_offspring1, Tre_offspring1, Wei_offspring1 = F_obj(offspring1[0,:],df)
            Ffun_offspring2, Tre_offspring2, Wei_offspring2 = F_obj(offspring2[0,:],df)
            OffspringsFitnesses[k][0]=Ffun_offspring1
            OffspringsFitnesses[k][1]=Ffun_offspring2
            # offsprings_dict={"offspring1 "+str(k):offspring1[0,:], "Ffun_offspring1 "+str(k):Ffun_offspring1, "offspring2 "+str(k):offspring2[0,:], "Ffun_offspring2 "+str(k): Ffun_offspring2 }
            # TempCSeries=pd.Series(offsprings_dict)
            # offsprings=pd.concat([offsprings,TempCSeries])
            pop=np.concatenate((pop, OffspringsPositions[k]),axis=0)
            fitnesses=np.concatenate((fitnesses, OffspringsFitnesses[k]),axis=0)
            
        MutantsPositions=np.zeros(shape=(nm,1,X.shape[1]))
        MutantsFitnesses=np.zeros(shape=(nm,1,1))
        # Mutants=pd.Series()
        for m in range(nm):
            ind=np.random.randint(0,N)
            p=X[ind,:].copy()
            Mutant=Mutate(p,mu)
            MutantsPositions[m][0]=Mutant
            Ffun_Mutant, Tre_Mutant, Wei_Mutant=F_obj(Mutant[0,:],df)
            last_weights = Wei_Mutant
            MutantsFitnesses[m][0]=Ffun_Mutant
            # Mutants_dict={"Mutant "+str(m):Mutant[0,:], "Ffun_Mutant "+str(k):Ffun_Mutant}
            # TempMSeries=pd.Series(Mutants_dict)
            # Mutants=pd.concat([Mutants,TempMSeries])
            pop=np.concatenate((pop, MutantsPositions[m]),axis=0)
            fitnesses=np.concatenate((fitnesses, MutantsFitnesses[m]),axis=0)

        inds = (-np.transpose(fitnesses)).argsort()
        sortedpop=pop[inds]
        Xnew=sortedpop[0][0:N]
        X=Xnew.copy()
        Ffun_new=fitnesses[inds][0][0:N]
        Ffun=np.transpose(Ffun_new.copy())

        Best_P=sortedpop[0][0]
        Best_FF=Ffun[0][0]

        print("At iteration {0}, the best solution fitness is {1}".format(t,Best_FF))    
        conv.append(Best_FF)
        t=t+1
    
    return Best_FF,Best_P,conv, Best_tresh, last_weights
            
            
            
                

