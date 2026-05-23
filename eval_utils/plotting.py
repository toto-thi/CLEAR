import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_confusion_matrix(y_true, y_pred, class_labels=None, title='Confusion Matrix', figsize=(10, 8), save_path=None):
    """
    Generates and plots a confusion matrix.
    """
    if class_labels is None:
        class_labels = sorted(list(set(y_true) | set(y_pred)))
        
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, 
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)
    
    plt.xlabel('Predicted Label', fontsize=13)
    plt.ylabel('True Label', fontsize=13)
    plt.title(title, fontsize=16)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
        
    plt.show()
