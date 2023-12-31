# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:35:33 2023

@author: morte
"""

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def calculate_weights(clf, X_valid, y_valid):
    y_pred = clf.predict(X_valid)
    acc = accuracy_score(y_valid, y_pred)
    pre = precision_score(y_valid, y_pred, average='binary')
    recall = recall_score(y_valid, y_pred, average='binary')
    f1s = f1_score(y_valid, y_pred, average='binary')
    f2s = fbeta_score(y_valid, y_pred, beta=2, average='binary')
    return acc * pre * recall * f1s * f2s

def OFFF(xtemp, dftr, dfte, tresh, we):
    x = xtemp.copy()
    FeatureIndices = np.where(x == 1)

    Xset = dftr.iloc[:, 0:-1]
    X = Xset.iloc[:, FeatureIndices[0]]
    y = dftr['Health']

    testset = dfte.iloc[:, 0:-1]
    Xtest = testset.iloc[:, FeatureIndices[0]]
    ytest = dfte['Health']

    scaler = MinMaxScaler()
    X_train_normalized = scaler.fit_transform(X)
    X_test_normalized = scaler.transform(Xtest)

    # Initialize Classifiers
    clf_xgb = xgb.XGBClassifier(objective='binary:logistic')
    clf_rf = RandomForestClassifier()
    clf_dt = DecisionTreeClassifier()
    clf_svm = SVC(probability=True)

    # Train classifiers and calculate weights
    clf_xgb.fit(X_train_normalized, y)
    clf_rf.fit(X_train_normalized, y)
    clf_dt.fit(X_train_normalized, y)
    clf_svm.fit(X_train_normalized, y)

    weights = [
        calculate_weights(clf_xgb, X_test_normalized, ytest),
        calculate_weights(clf_rf, X_test_normalized, ytest),
        calculate_weights(clf_dt, X_test_normalized, ytest),
        calculate_weights(clf_svm, X_test_normalized, ytest)
    ]

    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # Ensemble Classifiers using Weighted Voting
    eclf = VotingClassifier(estimators=[
        ('xgb', clf_xgb), 
        ('rf', clf_rf), 
        ('dt', clf_dt), 
        ('svm', clf_svm)], 
        voting='soft', weights=normalized_weights
    )

    eclf.fit(X_train_normalized, y)
    y_pred = eclf.predict(X_test_normalized)

    # Compute performance metrics
    accuracy = accuracy_score(ytest, y_pred)
    precision = precision_score(ytest, y_pred, average='binary')
    recall = recall_score(ytest, y_pred, average='binary')
    f1s = f1_score(ytest, y_pred, average='binary')
    f2s = fbeta_score(ytest, y_pred, beta=2)

    return accuracy, precision, recall, f1s, f2s