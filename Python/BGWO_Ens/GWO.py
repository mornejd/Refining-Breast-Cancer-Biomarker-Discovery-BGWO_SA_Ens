# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 16:16:17 2023

@author: morte
"""

import numpy as np
from Obj_fun import Obj_fun

def GWO(train_set_oversampled, val_set, N, max_Iter):
    feat = train_set_oversampled.drop('Health', axis=1)
    label = train_set_oversampled['Health']
    dim = feat.shape[1]
    X = np.zeros((N,dim))
    for i in range(N):
        for d in range(dim):
            if np.random.rand() > 0.5:
                X[i,d] = 1
    
    fit = np.zeros(N)        
    for i in range(N):
        fit[i] = Obj_fun(train_set_oversampled, val_set, X[i,:])
        
    idx = np.argsort(fit)
    Xalpha = X[idx[0],:]
    Xbeta = X[idx[1],:] 
    Xdelta = X[idx[2],:]
    Falpha = fit[idx[0]]
    Fbeta = fit[idx[1]]
    Fdelta = fit[idx[2]]
    
    curve = np.inf * np.ones(max_Iter)
    t = 0
        
    while t < max_Iter:
        a = 2 - 2 * (t / max_Iter)
        for i in range(N):
            for d in range(dim):
                C1 = 2 * np.random.rand()
                C2 = 2 * np.random.rand() 
                C3 = 2 * np.random.rand()
                Dalpha = np.abs(C1 * Xalpha[d] - X[i,d])
                Dbeta = np.abs(C2 * Xbeta[d] - X[i,d])
                Ddelta = np.abs(C3 * Xdelta[d] - X[i,d])
                A1 = 2 * a * np.random.rand() - a
                A2 = 2 * a * np.random.rand() - a
                A3 = 2 * a * np.random.rand() - a
                Bstep1 = jBstepBGWO(A1 * Dalpha)
                Bstep2 = jBstepBGWO(A1 * Dbeta)
                Bstep3 = jBstepBGWO(A1 * Ddelta)
                X1 = jBGWOupdate(Xalpha[d], Bstep1)
                X2 = jBGWOupdate(Xbeta[d], Bstep2)
                X3 = jBGWOupdate(Xdelta[d], Bstep3)
                #X1 = Xalpha[d] - A1 * Dalpha
                #X2 = Xbeta[d] - A2 * Dbeta
                #X3 = Xdelta[d] - A3 * Ddelta
                #Xn = (X1 + X2 + X3) / 3; 
                #TF = 1 / (1 + np.exp(-10 * (Xn - 0.5)))
                r = np.random.rand()
                #if r <= TF:
                #    X[i,d] = 1
                #else:
                #    X[i,d] = 0
                if r < 1/3:
                    X[i, d] = X1
                elif 1/3 <= r <= 2/3:
                    X[i, d] = X2
                else:
                    X[i, d] = X3

        
        for i in range(N):
            fit[i] = Obj_fun(train_set_oversampled, val_set, X[i,:])
            if fit[i] < Falpha:
                Falpha = fit[i]
                Xalpha = X[i,:]
            if Falpha < fit[i] and fit[i] < Fbeta:
                Fbeta = fit[i]
                Xbeta = X[i,:]
            if Fdelta > fit[i] and Fbeta < fit[i] and fit[i] > Falpha:
                Fdelta = fit[i]
                Xdelta = X[i,:]
    
        all_values = [(Falpha, Xalpha), (Fbeta, Xbeta), (Fdelta, Xdelta)]
        
        top_3_values = sorted(all_values, key=lambda x: x[0], reverse=False)[:3]
        
        Falpha, Xalpha = top_3_values[0]
        Fbeta, Xbeta = top_3_values[1]
        Fdelta, Xdelta = top_3_values[2]

                
        curve[t] = Falpha     
        print(f"{t+1} th iteration, Best-so-far Fitness = {curve[t]}")    
        t += 1
        
    Pos = np.arange(dim)
    Sf = Pos[Xalpha==1]
    Nf = len(Sf)
    Gene_names = feat.columns[Sf].tolist()
        
    return Gene_names, Sf, Nf, curve, Xalpha
    
                
def jBstepBGWO(AD):
    Cstep = 1 / (1 + np.exp(-10 * (AD - 0.5)))
    if Cstep >= np.random.rand():
        return 1
    else:
        return 0
    
    
def jBGWOupdate(X, Bstep):
    if (X + Bstep) >= 1:
        return 1
    else:
        return 0