import numpy as np
from sklearn import metrics

def accuracy_multilabel(y_true, y_pred_weights, total_only=False):
    y_pred = np.rint(y_pred_weights)
    acc_total = sum((y_true == y_pred).all(axis=1)) / y_true.shape[0]
    if total_only:
        return acc_total
    acc_v = []
    for i in range(y_true.shape[1]):
        acc_v.append(metrics.accuracy_score(y_true[:, i], y_pred[:, i]))
    return np.array(acc_v + [acc_total]) # by label and average

def f1_multilabel(y_true, y_pred_weights):
    y_pred = np.rint(y_pred_weights)
    f1_v = []
    for i in range(y_true.shape[1]):
        f1_v.append(metrics.f1_score(y_true[:, i], y_pred[:, i]))
    return np.array(f1_v + [np.mean(f1_v)]) # by label and average

def roc_auc_multilabel(y_true, y_pred_weights):
    roc_auc_v = []
    for i in range(y_true.shape[1]):
        roc_auc_v.append(metrics.roc_auc_score(y_true[:, i], y_pred_weights[:, i]))
    return np.array(roc_auc_v + [np.mean(roc_auc_v)]) # by label and average

# Recall, TPR = TP/(TP+FN)
def sensitivity_multilabel(y_true, y_pred_weights):
    y_pred = np.rint(y_pred_weights)
    TPR_v = []
    for i in range(y_true.shape[1]):
        CM = metrics.confusion_matrix(y_true[:, i], y_pred[:, i])
        TP = CM[1][1]
        FN = CM[1][0]
        TPR_v.append(TP/(TP+FN))
    return np.array(TPR_v + [np.mean(TPR_v)]) # by label and average

# TNR = TN/(TN+FP)
def specificity_multilabel(y_true, y_pred_weights):
    y_pred = np.rint(y_pred_weights)
    TNR_v = []
    for i in range(y_true.shape[1]):
        CM = metrics.confusion_matrix(y_true[:, i], y_pred[:, i])
        TN = CM[0][0]
        FP = CM[0][1]
        TNR_v.append(TN/(TN+FP))
    return np.array(TNR_v + [np.mean(TNR_v)]) # by label and average

def get_metrics(predictions, metrics_f):
    n_folds = len(predictions)
    n_labels = predictions[0][0].shape[1]
    n_metrics = len(metrics_f)
    # [folds, labels + total, metrics]
    result = np.zeros([n_folds, n_labels + 1, len(metrics_f)])
    for fold_i in range(n_folds):
        # n_samples x n_labels
        y_true = predictions[fold_i][0]
        y_pred_weights = predictions[fold_i][1]
        for metric_i in range(n_metrics):
            result[fold_i, :, metric_i] = metrics_f[metric_i](y_true, y_pred_weights)
    return result

def print_metrics(labels, metrics_labels, metrics_v):
    for label_i, label in enumerate(labels):
        print(label)
        parts = []
        for metric_i, metric in enumerate(metrics_labels):
            parts.append(f'{metric}: {np.round(metrics_v[:, label_i, metric_i].mean(), 4)}')
        print('\t' + '  '.join(parts))