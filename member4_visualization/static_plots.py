import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import numpy as np

def plot_pr_curve(y_true, y_scores, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    
    plt.figure()
    plt.plot(recall, precision, color='b', alpha=0.8)
    plt.fill_between(recall, precision, alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.grid(True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_scores, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, out_path):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_disaster_f1(disaster_stats, out_path):
    disasters = list(disaster_stats.keys())
    f1s = list(disaster_stats.values())
    
    plt.figure(figsize=(10, 6))
    plt.barh(disasters, f1s, color='skyblue')
    plt.xlabel('Mean F1 Score')
    plt.title('Performance by Disaster Zone')
    plt.xlim([0.0, 1.0])
    plt.gca().invert_yaxis()
    plt.grid(axis='x')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
