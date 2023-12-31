# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 21:25:35 2023

@author: morte
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def prepare_dataset(filepath, random_state=42):
    # Load the dataset
    df = pd.read_csv(filepath, delimiter="\t")
    df.rename(columns={'Unnamed: 0': 'rowname', 'Label': 'Health'}, inplace=True)
    df.drop(['rowname'], axis=1, inplace=True)

    normal_samples = df[df['Health'] == 0].sample(frac=1, random_state=random_state)
    cancer_samples = df[df['Health'] == 1].sample(frac=1, random_state=random_state)

    normal_train_size = int(0.6 * len(normal_samples))
    cancer_train_size = int(0.6 * len(cancer_samples))

    normal_train_set = normal_samples[:normal_train_size]
    normal_val_set, normal_test_set = train_test_split(normal_samples[normal_train_size:], test_size=0.5, random_state=random_state)

    cancer_train_set = cancer_samples[:cancer_train_size]
    cancer_val_set, cancer_test_set = train_test_split(cancer_samples[cancer_train_size:], test_size=0.5, random_state=random_state)

    train_set = pd.concat([normal_train_set, cancer_train_set], ignore_index=True)
    val_set = pd.concat([normal_val_set, cancer_val_set], ignore_index=True)
    test_set = pd.concat([normal_test_set, cancer_test_set], ignore_index=True)

    X_train = train_set.drop('Health', axis=1)
    y_train = train_set['Health']

    oversampler = SMOTE(sampling_strategy='auto', random_state=random_state)
    X_train_oversampled, y_train_oversampled = oversampler.fit_resample(X_train, y_train)

    train_set_oversampled = pd.concat([pd.DataFrame(X_train_oversampled), pd.Series(y_train_oversampled, name='Health')], axis=1)
    train_set_oversampled = train_set_oversampled.sample(frac=1, random_state=random_state)

    return train_set_oversampled, val_set, test_set
