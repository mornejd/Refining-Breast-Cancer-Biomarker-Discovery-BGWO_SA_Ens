# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 17:56:08 2023

@author: morte
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score

def calculate_weights(clf, X_valid, y_valid):
    y_pred = clf.predict(X_valid)
    acc = accuracy_score(y_valid, y_pred)
    pre = precision_score(y_valid, y_pred, average='binary')
    recall = recall_score(y_valid, y_pred, average='binary')
    f1s = f1_score(y_valid, y_pred, average='binary')
    f2s = fbeta_score(y_valid, y_pred, beta=2, average='binary')
    return acc * pre * recall * f1s * f2s

def Obj_fun(Train_set, Val_set, x):
    alpha = 0.8
    pop_binary = x

    FeatureIndices = np.where(pop_binary == 1)
    Xset_train = Train_set.iloc[:, 0:-1]
    X_train = Xset_train.iloc[:, FeatureIndices[0]]
    y_train = Train_set['Health']

    Xset_valid = Val_set.iloc[:, 0:-1]
    X_valid = Xset_valid.iloc[:, FeatureIndices[0]]
    y_valid = Val_set['Health']
    
    clf_xgb = xgb.XGBClassifier(objective='binary:logistic')
    clf_svm = SVC(probability=True)
    clf_rf = RandomForestClassifier()
    clf_dt = DecisionTreeClassifier()

    clf_xgb.fit(X_train, y_train)
    clf_svm.fit(X_train, y_train)
    clf_rf.fit(X_train, y_train)
    clf_dt.fit(X_train, y_train)

    weights = [
        calculate_weights(clf_xgb, X_valid, y_valid),
        calculate_weights(clf_svm, X_valid, y_valid),
        calculate_weights(clf_rf, X_valid, y_valid),
        calculate_weights(clf_dt, X_valid, y_valid)
    ]

    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    eclf = VotingClassifier(estimators=[
        ('xgb', clf_xgb), 
        ('svm', clf_svm), 
        ('rf', clf_rf), 
        ('dt', clf_dt)], 
        voting='soft', weights=normalized_weights
    )

    eclf.fit(X_train, y_train)
    y_pred = eclf.predict(X_valid)

    acc = accuracy_score(y_valid, y_pred)
    pre = precision_score(y_valid, y_pred, average='binary')
    recall = recall_score(y_valid, y_pred, average='binary')
    f1s = f1_score(y_valid, y_pred, average='binary')
    f2s = fbeta_score(y_valid, y_pred, beta=2, average='binary')

    fit = (acc + pre + recall + f1s + f2s) / 5
    fit = 1 - fit
    fit = (alpha * fit) + ((1 - alpha) * (FeatureIndices[0].shape[0] / pop_binary.shape[0]))
    fit_score = fit

    return fit_score