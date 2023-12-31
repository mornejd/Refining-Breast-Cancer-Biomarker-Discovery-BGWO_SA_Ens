# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 02:36:07 2023

@author: morte
"""
import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from Get_Function import Get_F
from GA import GA
from imblearn.over_sampling import SMOTE
from OFFF import OFFF
from save_values_to_excel import save_values_to_excel




# Load the sample dataset
#df = pd.read_csv('C:\\...\\Analysis\\Feature Selection\\Merged_ML_After_Batch_RemovedExtraCols.xls', delimiter="\t")
#df = pd.read_csv('C:\\...\\Analysis\\Feature Selection\\GSE45827_AllGenes_ML Ready.xls', delimiter="\t")
df.rename(columns={'Row.names': 'rowname','sample.levels': 'Health'}, inplace=True)
df.drop(['rowname','title','geo_accession','tissue.ch1','sample.labels'],axis=1,inplace=True)

normal_samples = df[df['Health'] == 0]
cancer_samples = df[df['Health'] == 1]

normal_samples = normal_samples.sample(frac=1, random_state=42)
cancer_samples = cancer_samples.sample(frac=1, random_state=42)

normal_train_size = int(0.6 * len(normal_samples))
cancer_train_size = int(0.6 * len(cancer_samples))

normal_train_set = normal_samples[:normal_train_size]
normal_test_set = normal_samples[normal_train_size:]

cancer_train_set = cancer_samples[:cancer_train_size]
cancer_test_set = cancer_samples[cancer_train_size:]

train_set = pd.concat([normal_train_set, cancer_train_set], ignore_index=True)

X_train = train_set.drop('Health', axis=1)
y_train = train_set['Health']

oversampler = SMOTE(sampling_strategy='auto', random_state=42)
X_train_oversampled, y_train_oversampled = oversampler.fit_resample(X_train, y_train)

train_set_oversampled = pd.concat([pd.DataFrame(X_train_oversampled), pd.Series(y_train_oversampled, name='Health')], axis=1)

train_set_oversampled = train_set_oversampled.sample(frac=1, random_state=42)

test_set = pd.concat([normal_test_set, cancer_test_set], ignore_index=True)

Solution_no=30
F_name='F7'
M_Iter=100
F_obj, Dim=Get_F(F_name)
[Best_FF,Best_P,conv, Best_tresh, last_weights]=GA(Solution_no,M_Iter,Dim,F_obj,train_set_oversampled)
FeatureIndices=np.where(Best_P==1)
Features=train_set.columns.values[[FeatureIndices[0]]]
print(Features)
print('Number of features:',Features.shape[1])
print('Tresh',Best_tresh)
[accuracy,precision,recall, f1score, f2score] = OFFF(Best_P, train_set_oversampled, test_set, Best_tresh, last_weights)
print(accuracy)
print(precision)
print(recall)
print(f1score)
print(f2score)

plt.plot(conv, marker=None, color='black', label='Objective Function Values')

plt.xlabel('Iteration')
plt.ylabel('Objective Function Values')

plt.title('Optimization Plot')

plt.legend(loc='best')

plt.show()

