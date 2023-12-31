# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 14:36:35 2023

@author: morte
"""
import os
os.chdir('C:\\...\\Analysis\\Feature Selection\\BGWO')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from prepare_dataset import prepare_dataset
from Test_func import Test_func


# Load data
#filepath4 = 'C:\\...\\Analysis\\Feature Selection\\Merged_ML_After_Batch_RemovedExtraCols.xls'
#filepath4 = 'C:\\...\\Analysis\\Feature Selection\\GSE45827_AllGenes_ML Ready.xls'
    
train_set_oversampled, val_set, test_set = prepare_dataset(filepath4)
test_set = pd.concat([val_set, test_set], axis=0)

X = train_set_oversampled.drop(['Health'], axis=1)
y = train_set_oversampled['Health']
classifiers = {
    'DT': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier(),
    'RF': RandomForestClassifier(),
    'SVM': SVC(probability=True)
}

num_iterations = 100
num_features_to_select = 5000
feature_scores = np.zeros(X.shape[1])

for _ in range(num_iterations):
    subset_features = np.random.choice(X.columns, num_features_to_select, replace=False)
    X_subset = X[subset_features]
    model = RandomForestClassifier()
    score = np.mean(cross_val_score(model, X_subset, y, cv=10))
    for feature in subset_features:
        feature_scores[X.columns.get_loc(feature)] += score

feature_scores /= num_iterations
top_features = np.argsort(feature_scores)[::-1][:num_features_to_select]
feature_names = X.columns[top_features]

best_f1 = 0
best_feature_set = None
best_classifier_name = ''

for classifier_name, classifier in classifiers.items():
    for num_features in range(5, len(feature_names) + 1, 5):
        selected_feature_names = feature_names[:num_features]
        X_subset = X[selected_feature_names]
        
        f1_scores = []
        roc_auc_scores = []
        pr_auc_scores = []
        mcc_scores = []
        balanced_acc_scores = []
        
        for train_index, test_index in StratifiedKFold(n_splits=10).split(X_subset, y):
            X_train, X_test = X_subset.iloc[train_index], X_subset.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            y_pred_proba = classifier.predict_proba(X_test)[:, 1] if hasattr(classifier, 'predict_proba') else classifier.decision_function(X_test)
            
            f1_scores.append(f1_score(y_test, y_pred))
            roc_auc_scores.append(roc_auc_score(y_test, y_pred_proba))
            pr_auc_scores.append(average_precision_score(y_test, y_pred_proba))
            mcc_scores.append(matthews_corrcoef(y_test, y_pred))
            balanced_acc_scores.append(balanced_accuracy_score(y_test, y_pred))
        
        avg_f1 = np.mean(f1_scores)
        
        print(f"Classifier: {classifier_name}, Features: {num_features}, Avg F1-Score: {avg_f1:.4f}")

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_feature_set = selected_feature_names
            best_classifier_name = classifier_name

from Test_func import Test_func
Xalpha = np.array([1 if col in best_feature_set else 0 for col in X.columns], dtype='float64')

number_of_ones = np.sum(Xalpha)

results, roc_info, pr_info  = Test_func(Xalpha, train_set_oversampled, test_set)
print(results["XGBoost"]["F1"])
print(results["XGBoost"]["roc_auc"])
print(results["XGBoost"]["prc_auc"])
print(results["XGBoost"]["Balanced Accuracy"])
print(results["XGBoost"]["MCC"])

print(results["Decision Tree"]["F1"])
print(results["Decision Tree"]["roc_auc"])
print(results["Decision Tree"]["prc_auc"])
print(results["Decision Tree"]["Balanced Accuracy"])
print(results["Decision Tree"]["MCC"])

print(results["Random Forest"]["F1"])
print(results["Random Forest"]["roc_auc"])
print(results["Random Forest"]["prc_auc"])
print(results["Random Forest"]["Balanced Accuracy"])
print(results["Random Forest"]["MCC"])

print(results["SVM"]["F1"])
print(results["SVM"]["roc_auc"])
print(results["SVM"]["prc_auc"])
print(results["SVM"]["Balanced Accuracy"])
print(results["SVM"]["MCC"])

print(results["KNN"]["F1"])
print(results["KNN"]["roc_auc"])
print(results["KNN"]["prc_auc"])
print(results["KNN"]["Balanced Accuracy"])
print(results["KNN"]["MCC"])

print(results["Neural Network"]["F1"])
print(results["Neural Network"]["roc_auc"])
print(results["Neural Network"]["prc_auc"])
print(results["Neural Network"]["Balanced Accuracy"])
print(results["Neural Network"]["MCC"])

avg_F1 = sum([results[classifier]['F1'] for classifier in results]) / len(results)
avg_prc_auc = sum([results[classifier]['prc_auc'] for classifier in results]) / len(results)
avg_roc_auc = sum([results[classifier]['roc_auc'] for classifier in results]) / len(results)
avg_BA = sum([results[classifier]['Balanced Accuracy'] for classifier in results]) / len(results)
avg_MCC = sum([results[classifier]['MCC'] for classifier in results]) / len(results)