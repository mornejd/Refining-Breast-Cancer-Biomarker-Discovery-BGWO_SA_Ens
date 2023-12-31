# -*- coding: utf-8 -*-
"""
Created on Wed May  4 02:04:28 2022

@author: morte
"""
import os
import math
os.chdir('C:\\...\\Analysis\\Feature Selection\\BGWO')
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from Test_func import Test_func


# load data Merged Dataset

#dataset = pd.read_csv('C:\\...\\Analysis\\Feature Selection\\Merged_ML_After_Batch_RemovedExtraCols.xls', delimiter="\t")
#X = dataset.iloc[:, 1:10629]
#Y = dataset.iloc[:, 10629]

# load data GSE45827

#dataset = pd.read_csv('C:\\...\\Analysis\\Feature Selection\\GSE45827_AllGenes_ML Ready.xls', delimiter="\t")
# split data into X and y
#X = dataset.iloc[:, 1:11732]
#Y = dataset.iloc[:, 11732]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)
train_set = pd.concat([X_train, y_train.rename('Health')], axis=1)

test_set = pd.concat([X_test, y_test.rename('Health')], axis=1)

pipeline = Pipeline([
                     ('scaler',StandardScaler()),
                     ('model',Lasso())
])

search = GridSearchCV(pipeline,
                      {'model__alpha':np.arange(0.1,10,0.1)},
                      cv = 10, scoring="neg_mean_squared_error",verbose=3
                      )

search.fit(X_train,y_train)
search.best_params_


coefficients = search.best_estimator_.named_steps['model'].coef_

importance = np.abs(coefficients)

DEGenes=X.columns

df1 = pd.DataFrame(DEGenes, columns=['Genes'])
df2 = pd.DataFrame(importance, columns=['Score'])
df= pd.concat([df1, df2], axis=1)
LASSO_Genes=df.loc[df['Score'] > 0]
rslt_df = LASSO_Genes.sort_values(by = 'Score')
print("Number of Selected Features:",rslt_df.shape[0])
rslt_df.plot(kind='bar',x='Genes',y='Score')
Gene_names = rslt_df['Genes'].tolist()

Xalpha = np.array([1 if col in Gene_names else 0 for col in X.columns], dtype='float64')

number_of_ones = np.sum(Xalpha)

results, roc_info, pr_info  = Test_func(Xalpha, train_set, test_set)
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

print(f"Average F1: {avg_F1:.5f}")
print(f"Average PRC AUC: {avg_prc_auc:.5f}")
print(f"Average ROC AUC: {avg_roc_auc:.5f}")
print(f"Average Balanced Accuracy: {avg_BA:.5f}")
print(f"Average MCC: {avg_MCC:.5f}")

