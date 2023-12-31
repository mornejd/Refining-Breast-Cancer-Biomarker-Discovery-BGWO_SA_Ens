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
                
        Xalpha_sa = simulated_annealing(Xalpha, train_set_oversampled, val_set)
        Falpha_sa = Obj_fun(train_set_oversampled, val_set, Xalpha_sa)
        Xbeta_sa = simulated_annealing(Xbeta, train_set_oversampled, val_set)
        Fbeta_sa = Obj_fun(train_set_oversampled, val_set, Xbeta_sa)
        Xdelta_sa = simulated_annealing(Xdelta, train_set_oversampled, val_set)
        Fdelta_sa = Obj_fun(train_set_oversampled, val_set, Xdelta_sa)
    
        all_values = [(Falpha, Xalpha), (Fbeta, Xbeta), (Fdelta, Xdelta), 
                      (Falpha_sa, Xalpha_sa), (Fbeta_sa, Xbeta_sa), (Fdelta_sa, Xdelta_sa)]
        
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
    #sFeat = feat[:,Sf]
        
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
    
# Operators
def swap(solution):
    idx = np.random.choice(len(solution), 2, replace=False)
    solution[idx[0]], solution[idx[1]] = solution[idx[1]], solution[idx[0]]
    return solution

def insertion(solution):
    idx = np.random.choice(len(solution), 2, replace=False)
    solution = np.insert(solution, idx[1], solution[idx[0]])
    solution = np.delete(solution, idx[0] if idx[0] < idx[1] else idx[0] + 1)
    return solution

def reversion(solution):
    idx = np.sort(np.random.choice(len(solution), 2, replace=False))
    solution[idx[0]:idx[1]+1] = solution[idx[0]:idx[1]+1][::-1]
    return solution

def rotate_right(solution):
    return np.roll(solution, 1)

def rotate_left(solution):
    return np.roll(solution, -1)

def select_operator():
    operators = [swap, insertion, reversion, rotate_right, rotate_left]
    weights = [1/len(operators) for _ in operators]
    return np.random.choice(operators, p=weights)

def simulated_annealing(wolf, train_set_oversampled, val_set):
    T = 30
    cooling_rate = 0.01
    threshold = 0.1

    current_solution = wolf.copy()

    while T > threshold:
        operation = select_operator()
        new_solution = operation(current_solution.copy())
        delta_E = Obj_fun(train_set_oversampled, val_set, new_solution) - Obj_fun(train_set_oversampled, val_set, current_solution)
        

        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            current_solution = new_solution

        T *= cooling_rate

    return current_solution