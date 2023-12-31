import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_curve, precision_recall_curve, auc, matthews_corrcoef, balanced_accuracy_score

def Test_func(x, Train_set, Test_set):
    FeatureIndices = np.where(x == 1)
    
    X_train = Train_set.iloc[:, FeatureIndices[0]]
    y_train = Train_set['Health']
    
    X_test = Test_set.iloc[:, FeatureIndices[0]]
    y_test = Test_set['Health']
    
    classifiers = {
        "XGBoost": xgb.XGBClassifier(objective='binary:logistic'),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(),
        "Neural Network": MLPClassifier(max_iter=1000)
    }

    results = {}
    roc_info = {}
    pr_info = {}

    for name, clf in classifiers.items():
        f1s = []
        rocs = []
        prcs = []
        mccs = []
        baccs = []
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        precisions = []
        recalls = []
        pr_aucs = []

        cv = StratifiedKFold(n_splits=10)
        for train_idx, test_idx in cv.split(X_train, y_train):
            X_train_cv, X_test_cv = X_train.iloc[train_idx], X_train.iloc[test_idx]
            y_train_cv, y_test_cv = y_train.iloc[train_idx], y_train.iloc[test_idx]

            clf.fit(X_train_cv, y_train_cv)
            y_pred_cv = clf.predict(X_test_cv)
            y_pred_proba_cv = clf.predict_proba(X_test_cv)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test_cv)
            
            f1s.append(f1_score(y_test_cv, y_pred_cv, average='binary'))
            fpr, tpr, _ = roc_curve(y_test_cv, y_pred_proba_cv)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            rocs.append(auc(fpr, tpr))

            precision, recall, _ = precision_recall_curve(y_test_cv, y_pred_proba_cv)
            precisions.append(np.interp(mean_fpr, recall[::-1], precision[::-1]))
            prcs.append(auc(recall, precision))

            mccs.append(matthews_corrcoef(y_test_cv, y_pred_cv))
            baccs.append(balanced_accuracy_score(y_test_cv, y_pred_cv))

        # Compute mean and std of metrics
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        roc_info[name] = (mean_fpr, mean_tpr, mean_auc)

        mean_precision = np.mean(precisions, axis=0)
        mean_recall = mean_fpr
        mean_pr_auc = auc(mean_recall, mean_precision)
        pr_info[name] = (mean_precision, mean_recall, mean_pr_auc)

        results[name] = {
            "F1": np.mean(f1s),
            "roc_auc": np.mean(rocs),
            "prc_auc": np.mean(prcs),
            "MCC": np.mean(mccs),
            "Balanced Accuracy": np.mean(baccs)
        }
    
    return results, roc_info, pr_info
