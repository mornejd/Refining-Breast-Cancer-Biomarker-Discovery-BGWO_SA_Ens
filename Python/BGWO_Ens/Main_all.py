# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 01:10:12 2023

@author: morte
"""
import os
import math
os.chdir('C:\\...\\Analysis\\Feature Selection\\BGWO')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from Obj_fun import Obj_fun
from Test_func import Test_func
from GWO import GWO
from GWO import simulated_annealing
from prepare_dataset import prepare_dataset

# Load data
#filepath4 = 'C:\\...\\Analysis\\Feature Selection\\Merged_ML_After_Batch_RemovedExtraCols.xls'
#filepath5 = 'C:\\...\\Analysis\\Feature Selection\\GSE45827_AllGenes_ML Ready.xls'

train_set_oversampled, val_set, test_set = prepare_dataset(filepath5)

N = 50
max_iter = 100

Gene_names, Sf, Nf, curve, Xalpha = GWO(train_set_oversampled, val_set, N, max_iter)

print ('Selected Features:\n',Gene_names)
print('Number of Selected Features:\n',Nf)

import matplotlib.pyplot as plt
plt.plot(range(1,max_iter+1), curve)
plt.xlabel('Number of Iterations')
plt.ylabel('Fitness Value')
plt.title('BGWO-SA')
plt.grid()
plt.savefig("BGWO-SA_Merged_All10.pdf", format='pdf')
plt.show()

Xalpha = np.array([1 if col in Gene_names else 0 for col in train_set_oversampled.columns], dtype='float64')

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

print(f"Average F1: {avg_F1:.5f}")
print(f"Average PRC AUC: {avg_prc_auc:.5f}")
print(f"Average ROC AUC: {avg_roc_auc:.5f}")
print(f"Average Balanced Accuracy: {avg_BA:.5f}")
print(f"Average MCC: {avg_MCC:.5f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

for classifier, (fpr, tpr, roc_auc) in roc_info.items():
    ax1.plot(fpr, tpr, label=f'{classifier} (AUC = {roc_auc:.3f})')
ax1.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal
ax1.set_title('ROC Curves')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.legend(loc='lower right')

for classifier, (precision, recall, pr_auc) in pr_info.items():
    ax2.plot(recall, precision, label=f'{classifier} (AUC = {pr_auc:.3f})')
ax2.set_title('Precision-Recall Curves')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.legend(loc='lower left')

plt.tight_layout()

plt.savefig("Combined_ROC_PR_Curves10.pdf", format='pdf')

plt.show()