
# ERROR METRICS: ACCURACY, PRECISION, RECALL, F1-SCORE, IOU, AND DICE

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def error_metrics(y_true, y_pred, plot_confusion_matrix=True):
    
    num_classes = y_true.shape[1]
    y_true_flat = y_true.argmax(axis=1).flatten()
    y_pred_flat = y_pred.argmax(axis=1).flatten()
    
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=range(num_classes))
    
    accuracy = np.zeros(num_classes)
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    iou = np.zeros(num_classes)
    dice = np.zeros(num_classes)
    
    for i in range(num_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        
        accuracy[i] = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
        precision[i] = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall[i] = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
        iou[i] = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
        dice[i] = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0
    
    if plot_confusion_matrix:
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=num_classes, yticklabels=num_classes)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Confusion Matrix')
        plt.show()
    
    return accuracy, precision, recall, f1, iou, dice